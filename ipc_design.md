# Accelerated Inference Process

## Redesigning IPC and inference with C++ actors

[Valentin Pollet](mailto:vpollet@mozilla.com), Jul 10, 2026

# Context

In an effort to bring hardware acceleration to llama.cpp backed features, [this revision](https://phabricator.services.mozilla.com/D268403) proposes a new process type: HWInference. The new process is a utility process operating within a much tighter sandbox than the existing Inference process: it doesn't run scripts, and its dedicated sandbox policy (landing in a follow-up, D268412) grants no network access.

Moving llama.cpp there means building two new pieces no matter what: a C++ IPC protocol for inference, and a way to drive it from chrome JS. What separates the options below is how much of the existing stack survives around those two pieces. This document weighs the options for bringing llama.cpp inference to the new process type.

## Existing infrastructure

Currently, llama.cpp is executed in a headless content process with remote type inference (shared with the Translations engine). The process is managed by a JSProcessActor pair, MLEngineParent and MLEngineChild. Inference runs in a worker thread in that process: LlamaCppPipeline (JS) drives LlamaRunner, a C++ WebIDL abstraction wrapping llama.cpp operations, which in turn dispatches the actual compute to a background thread and llama.cpp's threadpool.

Features like Link Preview call createEngine, which hands them an engine proxy (hiding the IPC and thread dispatching) to call text generation on. Control messages and model downloads flow over the actor pair; generation requests and streamed tokens flow over a MessagePort transferred to the child when the engine is set up.

```shell
[Parent, JS]
  -(actor IPC / MessagePort)-> [Child main, JS EngineDispatcher]
  -(worker postMessage)->      [Child worker, JS LlamaCppPipeline]
  -(WebIDL)->                  [Child worker, C++ LlamaRunner]
  -(dispatch)->                [Child background thread, C++ LlamaBackend]
```

On the parent side, the JS actor is wrapped for convenience, and ultimately consumer code interacts with a high-level API hiding all that complex machinery.

```javascript
// PipelineOptions -> MLEngine ready to run
export async function createEngine(options);

// Internal: MLEngine wraps calls to MLEngineParent, which posts (IPC) to MLEngineChild, which posts (dispatch) to the engine's worker Pipeline.
export class MLEngine{
	run(params); 		  // One shot
	runWithGenerator(params); // Streamed
}
```

The child side is three layers, and none of them survives the move as-is:

- MLEngineChild is a thin proxy. It handles a few lifecycle messages like ForceShutdown and defers everything stateful (backend selection, resource fetching, model notification) to the parent via sendQuery.
- EngineDispatcher (child main thread) holds the real logic: it reconciles RemoteSettings with pipeline defaults, resolves the backend, keeps sessions alive via keepAlive timeouts, and speaks the internal message protocol with engine instances.
- MLEngineWorker \+ LlamaCppPipeline (worker thread) does the worker plumbing: turning high-level options into llama.cpp parameters (nCtx, mlock), formatting chat, driving the streaming loop, and computing the JS-side metrics like TTFT and throughput.

# Design goals

### Integrate in MLEngine

The llama.cpp engine will run a new protocol in the new process but keep the surface: createEngine, run and runWithGenerator. This will limit churn at call sites and make it easy to check that the new engine passes the old tests without changing them.

### Same behavior as the old engine

The new engine must behave exactly like the old. This should be validated by making sure the new engine emits the same telemetry as before and passes the same tests (perftests and golden samples).

### OpenAIPipeline interoperability

The new engine will be used in SmartWindow in paths currently using the OpenAI MLEngine (OpenAIPipeline). Interoperability with this engine is nice-to-have.

### Configuration surface cleanup

The current code has several layers that forward options: Link Preview sets fields in PipelineOptions. LlamaCppPipeline converts those options to WebIDL LlamaContextOptions which converts them to llama.cpp options. There are two issues here: the surface is huge, we’re forwarding a lot of llama.cpp context/model options; each conversion layer is not mechanical and can decide to drop options or change defaults in the process.

The rollout resolves this with one rule: the surface is what is served. An option in the surface is honored verbatim; an option that cannot be served is removed from the surface and from callsites rather than detected or warned about at runtime; defaults live in exactly one place, the WebIDL dictionaries, and the wire carries the WebIDL surface enums directly, so one enum family runs from chrome JS to the backend. See the configuration principle in [the rollout doc](./rollout.md).

### Future proofing

Link Preview is the only existing consumer and its workflows are “create engine, generate, tear down engine” shaped. Our current llama.cpp engine does not support:

- \[P0\] **Accelerator discovery and placement.** The new engine/infras must provide a way to query the acceleration capabilities.
- \[P1\] **Constrained generation**. The new engine should support constraining models using JSON schemas. This is also important if tool calls ever become a thing.
- \[P1\] **Reasoning API**. The only consumer uses SmolLM2 360M which does not use reasoning, but bigger models likely will. The way models output reasoning is heterogeneous and architecture-dependent, and the new engine will need to handle that cleanly: toggling, separating “reasoning content” and “content” etc.
- \[P2\] **Batching**. Smaller models struggle with attention more, big extraction tasks could be broken down into N smaller tasks running on a long shared prefix. This has the potential to improve quality as tasks are more focused, and speed as batching
- \[P3\] **Stateful sessions** (chat, interactive) with stateful KV-caching and sequence handling. Chat is an unlikely scenario, but question-answering is one that could be interesting: prefill a context window with the contents of N pages, have a session with a small model that retrieves information in the pages. Small agentic loops could also exist in that setup.
- \[P3\] **Interruptability**. This could prove helpful in running long background tasks in low power usage mode e.g. computing only when memory-pressure is low or on few threads with low priority

# Proposal

## IPC protocol

A few ground rules shape the protocol. Everything is asynchronous, and all control traffic starts at the parent. The utility process can only act on what the parent hands it. Backend-wide operations, like asking what accelerators the process can see, live on the browser manager (PHWInferenceBrowserManager), which we own. Each generator is a PTextGeneration actor the manager hands out, and its compute stays off the IPC thread. The manager is the factory. We add a generator constructor to the manager protocol; it takes the model and the generator options, and loads the model as part of construction.

```c
// The browser-side manager.
protocol PHWInferenceBrowserManager
manages PTextGeneration;
child:
  async PTextGeneration(FileDescriptor model, TextGenerationOptions opts);
  // Backend-wide operations live here too, e.g. accelerator discovery (P0):
  // async GetCapabilities() returns (Capabilities);
```

```c
// One generator per model. The manager constructs it with the model fd and its
// options; loading the model is part of construction (IPDL forbids returns on
// async constructors, so the one-shot Ready message reports the outcome, per
// PRemoteWorker's Created/Error) and the model is freed when the actor dies.
// Messages are handled on the manager thread; the compute runs on a
// per-generator TaskQueue.
protocol PTextGeneration
parent:
  async Ready(LoadResult result);
  async Delta(nsCString text);
child:
  async Generate(GenerateRequest req) returns (GenerateResponse);
  async Clear();
  async Cancel();

// Room to grow by adding messages. Batch(GenerateRequest[]) runs N sequences
// over a shared prefix. Suspend/Resume moves a generator's KV state to and
// from an fd.
```

Each payload is its own type and carries only what it needs:

```c
// ChatMessage, not Message: the generated headers already use IPC::Message.
// Role is the WebIDL surface enum (TextGenerationRole), carried directly.
struct ChatMessage { Role role; nsCString content; };   // Role: System | User | Assistant

// Streamed text is the one channel a consumer reads today, so it is the one
// message: Delta(text). A future channel — reasoning for models that think,
// tool calls — is its own parent-directed message beside Delta; adding a
// channel means adding a message the receiver has to handle before the code
// compiles.

// Appended to the generator's context; call Clear() first to start fresh. Holds
// what a caller varies per generation. Future fields would slot in here the
// same way, like a reasoning toggle or a JSON schema for constrained output.
// They feed sampling and formatting, and wait until a consumer needs them.
struct GenerateRequest {
  ChatMessage[] messages;
  uint32_t maxTokens;
  uint32_t bufferLength;   // tokens to batch into one ContentDelta
  Sampler[] samplers;
  int32_t[] stopTokens;    // llama's token type
  bool stopOnEndOfGenerationTokens;
};

// The reply carries the whole result, so a caller can ignore the event stream
// and still get everything. Streaming just shows output early. A reasoning
// model would return its reasoning text here, alongside content.
struct GenerateResult {
  nsCString content;     // the full generated text (the ContentDeltas joined)
  FinishReason reason;   // Eos | Length | StopToken | Cancelled
  Usage usage;
};

// Runtime failures (invalid sampler chain, ...) resolve the reply with a
// message the caller can surface; crashes reject through ActorDestroy.
struct GenerateError { nsCString message; };
union GenerateResponse { GenerateResult; GenerateError; };

// Counts and timings the child measures directly, laid out so the parent can
// project them into the OpenAI usage shape. A reasoningTokens count would join
// these for reasoning models, where most of the budget goes.
struct Usage {
  uint32_t promptTokens;
  uint32_t generatedTokens;
  Timings timings;       // prefill, decode; the model-load time belongs to
                         // construction and travels in Ready's LoadResult
};

// The options the generator is constructed with, alongside the model fd. Holds
// only knobs that are actually served (see Configuration surface cleanup): an
// option in this struct is honored verbatim. A KV-cache/flash-attention knob
// joins when the acceleration phase serves one for real; device placement (P0)
// lands here too, defaulting to CPU today.
struct TextGenerationOptions {
  uint32_t contextSize;
  uint32_t numThreads;   // 0 picks the same physical-core count as the old path
  uint32_t batchSize;
  uint32_t ubatchSize;
};
```

The parent resolves the model's file descriptor from ModelHub, which runs in the privileged parent, and the manager constructs the generator with it. Loading runs on the generator's TaskQueue as part of construction, and Generate runs on that same queue, so it starts once loading finishes and needs no separate ready check; the one-shot Ready message reports the load outcome, and the parent side does not hand the generator out until it resolves, so a load failure is a creation failure. Generate appends its messages to the context, so a warm generator accumulates turns on its own; a caller that wants a clean slate, like the one-shot Link Preview case, calls Clear first. The model is freed when its generator dies.

There is one manager per browser session: GetOrCreate() hands every creator the same actor, launching the process on first use and joining a launch already in flight, so the manager's managed set is the process-wide generator count. The manager holds a keep-alive on the browser instance key (UtilityProcessManager's Acquire/ReleaseHWInferenceKeepAlive, from D314420) and releases it when its last generator dies; UtilityProcessManager then shuts the process down unless another consumer of that key still holds one. A creator that arrives while that release decision is pending takes a reservation the decision honors, and once the decision falls the cached manager retires immediately, so a late creator launches a fresh process instead of joining a dying one.

A Delta and the Generate reply are both messages on the same actor, and IPDL delivers messages in the order the child sent them. The child sends every Delta and then the reply, so the reply always lands after the tokens. Completion is the Generate reply itself, which is why the protocol needs no separate done event. One generation is in flight at a time — the chrome surface rejects an overlapping generate() with InvalidStateError — so every Delta belongs to the single pending Generate and the wire carries no request index. One rule keeps this working: the child issues every Delta and the reply from the actor's thread, and issues the reply last. Compute runs on a separate TaskQueue, so each token hops back to the actor thread to be sent, and the reply is queued behind the last token. That rule is the whole concurrency contract, and it lets us delete the hand-written SPSC queue and double-buffered promise in LlamaRunner, which did the same serialization inside one process.

Cancel is best-effort and targets the pending Generate through a flag on the generation itself, so a Cancel can never bleed into the request that follows: an in-flight Generate always resolves, and a cancelled one resolves with the tokens produced so far and a cancellation flag. Crashes and shutdowns come for free from the actor model: every pending return rejects through ActorDestroy, the parent-side owner maps that to the engine's failure path and respawns the process on the next request, and a crash on a given device configuration feeds the blocklist.

The protocol is expressive enough to have interesting workflows, starting with the basic use-case that would cover a Link Preview key-points style of feature with a warm engine.

```shell
-> Construct PTextGeneration(model, { contextSize, numThreads, batchSize, ubatchSize })
-> Generate(
messages = [{.role = Role.System, .content = "You are helpful."},
{.role = Role.User, .content = "Summarize this [...]" }],
options = { .maxTokens = 64, .bufferLength = 16})
<- Delta: "The page argues that [...]"   # 16 tokens, buffered
<- Delta: "[...] in conclusion."         # 12 tokens, reached eos
<- Generate resolves: full text, finish reason, usage (prefill/decode timings)
# Next generation, if the engine is still warm
-> Clear()
-> Generate(...) [...]
-> Actor destroyed
```

Fancier scenarios are also cleanly expressible like a turn-based flow:

```shell
-> Generate(messages = [{.role = system, "You are helpful."},
 		{.role = user, "Hello"}],
options = options)
<- "Hi! What can I help you with?"
# Default behavior is append messages
-> Generate(messages = [{.role = user, "Who are you?"}],
      options = options)
<- "I'm powered by Gemini"
```

The child actor, TextGenerationChild, is a thin wrapper: it owns a LlamaBackend and drives it on the generator's TaskQueue. The parent actor, TextGenerationParent, is a C++ object exposed to chrome JS through WebIDL, so the parent process drives it directly.

## Embed new process in MLEngine eco-system

The MLEngine machinery is built around very different actors and IPC. The fork happens early: createEngine must route llama.cpp to the new process type before EngineProcess spins up the "inference" content process. Three ways to do it, from most to least conservative:

1. Lift LlamaCppPipeline to the parent, refactored to own a TextGenerationParent instead of a LlamaRunner. Least API churn on paper, but everything that justified the class dies (OPFS reads, wasm plumbing, JS-side metrics)
2. Drop the pipeline object. A small parent-side class implements the MLEngine surface (run, runWithGenerator, terminate) directly on top of TextGenerationParent, and createEngine returns it when the pref routes to llama.cpp. Consumers, telemetry and the Nimbus switch don't move.
3. Don't embed at all: consumers hold a TextGenerationParent directly or a thin wrapper around it, skipping all MLEngine.

Option 2 is less work than adapting LlamaCppPipeline and puts us in good shape to depart from PipelineOptions which is starting to feel crammed. Option 3 has the benefit of untangling the onnxruntime backed features from the llama.cpp backed ones, at the cost of rewriting the telemetry and perf reporting layer.

### Swapping an OpenAI consumer to a local engine

This is cheaper than it sounds, and it makes on-device a realistic option for other teams. SmartWindow talks to a remote OpenAI endpoint through OpenAIPipeline, and it measures what that endpoint reports back: the usage numbers (prompt tokens, completion tokens, and for reasoning models, reasoning tokens). The interop that matters is reporting those same numbers from a local engine. If our result struct carries the same token counts, SmartWindow's telemetry keeps working whether the tokens came from the network or from this process, and the single parent-side Glean point gives us that for free (see below).

### Perf telemetry

The telemetry could use a brush up. Measurement is JS-side and coarse today: throughput computed in the pipeline, the isPrompt plumbing exists largely to derive one prompt-vs-decode timestamp, no model-load/prefill/decode breakdown, no memory high-water mark. The C++ child owns the ground truth for every boundary worth measuring, including the CPU and memory usage the sandbox no longer lets us sample from JS. Make the wire carry the measurements, a timings/usage struct on GenerateResult and on generator creation, and keep Glean emission in exactly one parent-side place. Events become backend-agnostic by construction and the Nimbus branch is the only differentiator.
