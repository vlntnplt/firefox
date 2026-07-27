(hwinference-architecture)=

# HWInference: the on-device hardware-accelerated inference process

`HWInference` is a utility process that runs native, hardware-accelerated
inference libraries (currently `parakeet.cpp`, backed by `libggml`) outside of
any content or the main process. Unlike the [Firefox AI
Runtime](inference-architecture) inference process, it does not run
JavaScript: its job is purely computational, receiving some input, running it
against a model file, and producing some output.

This document describes the generic facility: the process itself, how a
consumer connects to it, and the security properties that hold regardless of
who the consumer is. It does not cover the specifics of any one consumer.
Two consumers exist today (with one being an example not intended for landing),
demonstrating the two ways `HWInference` can be used:

- **Content-initiated**: [`SpeechRecognition`](/media/SpeechRecognition),
  implementing the on-device recognition side of the Web Speech API. A web
  page, running in a content process, drives this through DOM calls.
- **Browser-initiated**: the HWInference end-to-end example
  (`toolkit/components/ml/backends/browser-inference-example/`), reachable
  from `Tools > HWInference Smoke Test` in the browser chrome. Privileged,
  parent-process JavaScript drives this directly, with no content process
  involved at all.

## The `HWInference` process

Its sandbox policy resembles that of the GPU process, but it doesn't have
access to the display server, or to things like fonts, or other special
system calls or capabilities related to rendering. It only does computations:
it receives some input (e.g. text, image, audio data) and uses a model file and
a library to perform inference, and produces some output (e.g. timed text
fragments, summary). It is generally only started when needed, and closed
quickly when not needed anymore, but lifetime is in the hands of the _user_ of
the `HWInference` system.

It delegates all model management tasks to the
[ModelHub](https://searchfox.org/firefox-main/source/toolkit/components/ml/content/ModelHub.sys.mjs),
which it calls via IPC. This includes model availability checks and download
(`ModelHub` handles caching). It can also acquire a handle to a model file
using a `FileDescriptor` passed via IPC, without copy, important because model
files can be quite big. This also allows `mmap`ing some models (notably
mixture-of-experts models), for significant memory footprint gains.

Since it doesn't run JavaScript, it will eventually be possible to tighten the
sandbox further on macOS by making it a different executable, relinquishing
the capability to mark pages as executable for JITing code.

`ONNX Runtime` (for non-LLM type inference) and `llama.cpp` (for LLM-type
inference on text) are eventually expected to also run inside `HWInference`, to
be able to use hardware acceleration for tasks unrelated to speech recognition.

## A flexible user:process relationship

Currently, `HWInferenceParent.h` defines two distinct, hardcoded
process-instance keys:

```cpp
#define HWINFERENCE_BROWSER_INSTANCE_KEY "browser"_ns
#define HWINFERENCE_CONTENT_INSTANCE_KEY "content"_ns
```

`UtilityProcessManager` keys processes by the `(SandboxingKind,
instanceKey)` pair (see `GetProcess`/`StartProcess` in
`ipc/glue/UtilityProcessManager.cpp`), so `HW_INFERENCE` +
`HWINFERENCE_CONTENT_INSTANCE_KEY` and `HW_INFERENCE` +
`HWINFERENCE_BROWSER_INSTANCE_KEY` are **two separate OS processes with two
separate PIDs**.

This is exercised directly by `TestUtilityProcess.HWInferenceInstances`
(`ipc/glue/test/gtest/TestUtilityProcess.cpp`), which asserts the two PIDs
differ and that shutting down one instance doesn't affect the other.

Content-driven inference (`SpeechRecognition`) always uses
`HWINFERENCE_CONTENT_INSTANCE_KEY`
(`UtilityProcessManager::StartContentHWInferenceManager`). Privileged,
parent-process-triggered inference (the smoke test, and future "browser AI"
features) uses `HWINFERENCE_BROWSER_INSTANCE_KEY`
(`UtilityProcessManager::StartHWInference`). Content-driven inference never
runs in the same OS process as browser-driven inference — no code changes to
this isolation are needed when a new privileged caller is added, since the
split already exists at the `UtilityProcessManager` layer.

We can imagine devising more advanced isolation policy, per origin, per feature,
etc., by keying differently.

## Process lifetime

Consumers decide how long an `HWInference` instance lives, not the process
itself. `UtilityProcessManager` holds a keep-alive count per instance key
(`AcquireHWInferenceKeepAlive`/`ReleaseHWInferenceKeepAlive`, main thread
only); when the last one is released the instance is shut down with
`CleanShutdown` rather than lingering until browser shutdown. This mirrors what
the `WINDOWS_FILE_DIALOG` utility process does with `sOpenDialogActors`.

The count lives in the main process because that is where
`UtilityProcessManager` is. `HWInferenceManagerParent` — the thing that
represents "a content process is connected" — lives in the utility process and
cannot shut its own process down.

- **Content-process consumers** go through `PContent`:
  `RequestHWInferenceConnection` acquires a keep-alive for the requesting
  content process when it succeeds, and `ReleaseHWInferenceConnection` drops
  it. `HWInferenceManagerChild` sends exactly one release per successful
  request: from `ActorDestroy` for the endpoint it adopted, and immediately for
  an endpoint it had to discard (two concurrent requests can race, and only the
  first is bound). `ContentParent` counts how many that process holds and
  releases the remainder in its own `ActorDestroy`, so a crashed content
  process cannot pin the utility process forever.
- **Parent-process consumers** call
  `Acquire`/`ReleaseHWInferenceKeepAlive` directly, with no IPC involved.

Keep-alives are per instance key, so releasing the content instance never
affects the browser one.

`UtilityProcessManager` is a plain refcount with no policy of its own: it shuts
the instance down the moment the count reaches zero. Any "keep it warm for a
bit" behaviour belongs to the consumer holding the keep-alive — see the grace
period in the [SpeechRecognition
docs](/dom/media/docs/SpeechRecognition.md) for one such policy.

`TestUtilityProcess.HWInferenceKeepAlive{,PerInstance}` covers the refcount and
its per-instance isolation.

The topology this produces today (plus where a second, planned browser-initiated
feature would land) looks like this: **every** content process shares the one
content instance, and **every** browser-initiated feature, regardless of how
many exist, shares the one browser instance — two OS processes total, split
solely along the content/browser trust boundary:

```{mermaid}
%%{init: {"flowchart": {"htmlLabels": false}}}%%
flowchart LR
  subgraph CP1[Content Process A]
    SR1[SpeechRecognition]
  end
  subgraph CP2[Content Process B]
    SR2[SpeechRecognition]
  end
  subgraph HWC["HWInference<br/>content"]
    SRP1[SpeechRecognitionParent for A]
    SRP2[SpeechRecognitionParent for B]
  end
  subgraph MP[Main Process]
    MLU[Toy example<br/>nsIMLUtils]
    FUT["summarization"]
  end
  subgraph HWB["HWInference<br/>browser"]
    BIE[BrowserInference<br/>ExampleParent]
    LLAMA["<br/>llama.cpp summarization"]
  end

  SR1 --> SRP1
  SR2 --> SRP2
  MLU --> BIE
  FUT -. planned .-> LLAMA
```

### Content-initiated: connecting directly to the utility process

`SpeechRecognition` obtains a direct channel to the utility
process: `ContentParent::RecvRequestHWInferenceConnection` (the only
speech-recognition-related method on `ContentParent`) brokers an endpoint via
`UtilityProcessManager::StartContentHWInferenceManager`, and the content
process binds it as `HWInferenceManagerChild`. Audio and results then flow
directly between the content and utility processes, without going through the
main process on every message. Model install/consent still routes through the
main process (see below) — see [SpeechRecognition.md](/media/SpeechRecognition)
for the full call flow.

### Browser-initiated: direct Parent - Utility communication

The example is driven from privileged chrome JavaScript
(`BrowserCommands.runHWInferenceSmokeTest` in `browser-commands.js`) calling
`nsIMLUtils.runHWInferenceSmokeTest()`, implemented by `MLUtils::RunHWInferenceSmokeTest`.
Unlike the content case, there is no content process at all: the main process
itself is the consumer.

- `MLUtils` starts (or reuses) the browser-instance `HWInference` process via
  `UtilityProcessManager::StartHWInference(HWINFERENCE_BROWSER_INSTANCE_KEY)`
  and gets back an `HWInferenceParent` (the main-process side of `PHWInference`).
- `HWInferenceParent::RunBrowserSmokeTest()` creates a `PBrowserHWInferenceManager`
  endpoint pair and hands the utility-process-side endpoint over via
  `PHWInference::NewBrowserHWInferenceManager`; the utility process binds
  `BrowserHWInferenceManagerParent`, the main process binds
  `BrowserHWInferenceManagerChild` locally.
- `BrowserHWInferenceManagerChild` (main process) then creates a
  `PBrowserInferenceExample` actor pair and calls `RunScalar()`.
  `BrowserInferenceExampleParent` (utility process) first exercises the
  generic model-provisioning path — `InstallModel`/`IsModelInstalled` over
  `PHWInference` — using its own registered resolver/gate (see below), then
  runs a small `libggml` compute graph via `LlamaRuntimeLinker` and returns the
  result.

```{mermaid}
sequenceDiagram
  autonumber

  box Main Process
    participant JS as Chrome JS
    participant MLU as MLUtils
    participant UPM as UtilityProcessManager
    participant HWP as HWInferenceParent
    participant BHC as BrowserHWInferenceManagerChild
  end

  box "HWInference (browser instance)"
    participant HWC as HWInferenceChild
    participant BHP as BrowserHWInferenceManagerParent
    participant BIE as BrowserInferenceExampleParent
    participant LIB as libggml (LlamaRuntimeLinker)
  end

  JS->>MLU: runHWInferenceSmokeTest()
  MLU->>UPM: StartHWInference(BROWSER_INSTANCE_KEY)
  UPM-->>MLU: HWInferenceParent
  MLU->>HWP: RunBrowserSmokeTest(input)
  HWP->>HWC: PHWInference::NewBrowserHWInferenceManager(endpoint)
  HWC->>BHP: bind
  HWP->>BHC: bind (local endpoint)
  BHC->>BHP: PBrowserInferenceExample(alloc)
  BHP->>BIE: alloc
  BHC->>BIE: RunScalar(input)
  Note over BIE: exercise the generic InstallModel/<br/>IsModelInstalled path first (see below)
  BIE->>LIB: build + compute graph
  LIB-->>BIE: value
  BIE-->>BHC: BrowserInferenceExampleResult
  BHC-->>HWP: result
  HWP-->>MLU: value
  MLU-->>JS: Promise resolves(value)
```

## Generic model provisioning: resolver, gate, and `ModelHub`

Every task using `HWInference` model management registers up to two XPCOM
components under its task name, and uses one shared XPCOM component:

- `nsIMLModelResolver` (contract id `@mozilla.org/ml/model-resolver;1?task=<task>`):
  expands an opaque, task-defined `id` string into concrete
  `model`/`revision`/`filename` coordinates. `SpeechModelResolver` expands a
  language-derived id against the compiled-in speech model table;
  `BrowserInferenceExampleModelResolver` recognizes a single hardcoded
  `"demo-model"` id. As an example, Speech Recognition currently has identifiers
  `english` and `multilingual`, mapping to two models of the Parakeet family.
- `nsIMLModelDownloadGate` (contract id `@mozilla.org/ml/model-download-gate;1?task=<task>`):
  decides whether a download may proceed, given the resolved model coordinates
  and the identity of the requester (`aInnerWindowId`/`aContentId`, below).
  `SpeechModelDownloadGate` shows the model-download doorhanger;
  `BrowserInferenceExampleModelDownloadGate` authorizes immediately, since its
  caller is already-trusted parent-process code, but any policy is possible.
- `nsIMLModelHub`: a thin XPCOM component wrapping `ModelHub`, used by
  `HWInferenceParent` for both the download and the local-cache check. Shared
  by every task, pre-existing.

`HWInferenceParent` (main-process side of `PHWInference`) is the only place
that calls these. Every `InstallModel` request that reaches it — whether
relayed from a content-initiated session (via `HWInferenceManagerParent`/the
task's actor) or issued directly for a browser-initiated one — carries only a
`task`, an `id`, an `aInnerWindowId`, and an `aContentId`.
`HWInferenceParent` first expands the id to concrete
`model`/`revision`/`filename` coordinates via the task's `nsIMLModelResolver`
(an id it doesn't recognize fails the install immediately, before any gate or
`ModelHub` call), then does the following.

### 1. Consult the gate

With coordinates in hand, `HWInferenceParent` looks up the task's
`nsIMLModelDownloadGate` by contract id. A task with no registered gate is
ungated: the download proceeds unconditionally. A task with a gate defers to
its `ShouldAllowDownload` decision, passing along the coordinates and the
requester's `innerWindowId`/`contentId` — this is the only point where a
content-initiated gate can authenticate the requester (see
[Security](#security-what-a-compromised-content-process-can-and-cannot-do),
below).

```{mermaid}
sequenceDiagram
  autonumber

  box Main Process
    participant HWP as HWInferenceParent
    participant Gate as nsIMLModelDownloadGate
  end

  HWP->>Gate: do_GetService(model-download-gate for this task)
  alt no gate registered for this task
    Note over HWP: ungated: download proceeds unconditionally
  else gate registered
    HWP->>Gate: ShouldAllowDownload(task, model, revision,<br/>filename, innerWindowId, contentId,<br/>progressToken, callback)
    Note over Gate: a content-initiated gate resolves innerWindowId<br/>to a WindowGlobalParent and checks its<br/>ContentParentId() against the stamped contentId
    Gate-->>HWP: callback resolves(allow)
  end
```

### 2. Download (or not)

Only now does `HWInferenceParent` touch `ModelHub`, and only if step 1 ended
in "ungated" or "allow". A denial resolves the install `false` without any
`ModelHub` call.

```{mermaid}
sequenceDiagram
  autonumber

  box Utility Process
    participant TA as Task actor
  end

  box Main Process
    participant HWP as HWInferenceParent
    participant MH as nsIMLModelHub (ModelHub)
  end

  alt allowed (ungated or gate said allow)
    HWP->>MH: DownloadModel(task, model, revision, files,<br/>progressCallback, completionCallback)
    MH--)HWP: progress callback(s)
    MH-->>HWP: download success/fail
    HWP-->>TA: true/false
  else denied
    Note over HWP: nothing downloaded
    HWP-->>TA: false
  end
```

This omits two pre-download short-circuits for clarity: `RecvInstallModel`
itself resolves straight to `true` under the `browser.ml.modelHub.testing`
mock (used by tests, bypassing the gate and `ModelHub` entirely), and each gate
implementation may check whether the model is already cached before deciding
whether to prompt — `SpeechModelDownloadGate` does this via `ModelHub`'s local
cache, resolving `true` with no prompt when there is nothing to download.

## Security: what a compromised content process can and cannot do

A review concern for the content-initiated case was whether a compromised
content process could install or read an arbitrary model file by supplying
its own `model`/`revision`/`filename`, or trigger a model download without
the user's consent.

No, not because the values get sanitized, but because content has no path to
even attempt it: the consent decision plus the download both live entirely in
the trusted parent (main) process:

- Content-facing uses (e.g. `PSpeechRecognition`) never mentions
  `model`/`revision`/`filename` on the IPC call. They only ever use
  task-specific, opaque identifiers (for `SpeechRecognition`, BCP-47 language
  tags), and not a file name or other specific identifier anymore.
- The id itself (e.g. `dom::LanguagesToSpeechModelId` for speech recognition)
  reads only a table generated at build time and compiled into the binary; it
  is not loaded from anything runtime, or attacker-writable. That mapping
  from e.g. a language to an id happens wherever the task's actor runs (for
  `SpeechRecognition`, in the Utility process), never in content.
- `HWInferenceParent`, on the main-process side, resolves the id back to
  the `ModelHub` slug by calling the task's `nsIMLModelResolver`, which
  reads the very same compiled-in table.

So the only attacker-controlled input anywhere on this path is a
task-specific opaque identifier, matched against a static compiled-in table, and
this id is the only things that gets send over IPC.

### Consent to a model download cannot be faked by content

A model download requires the download gate's authorization, decided **and
enforced in the parent (main) process**, never in content or in the utility
process:

- In the case of content, it can only *ask*. It sends its request for a model
  along with the **inner window id** of its requesting document (not a
  `BrowsingContext` id) to the task's Utility-process actor. It never sees, and
  cannot tweak, a consent answer.
- The task's Utility-process actor maps the request to a model id, mints a
  progress token (to distinguish concurrent requests), and relays
  `PHWInference::InstallModel` to the main process, stamping the **trusted**
  `ContentParentId` of the content process that owns the connection (in the case
  of an initiation by a content process) — never a value content supplies.
- `HWInferenceParent::RecvInstallModel` (main process) resolves the id via the
  task's `nsIMLModelResolver`, then consults the task's `nsIMLModelDownloadGate`.
  A content-initiated gate (`SpeechModelDownloadGate`) resolves the
  content-supplied inner window id to a `WindowGlobalParent`
  (`WindowGlobalParent::GetByInnerWindowId`) and checks that its
  `ContentParentId()` matches the stamped, trusted `aContentId` before trusting
  it — a compromised content process cannot name a window it does not own to
  anchor the prompt on another tab or act for another origin. A browser-initiated
  gate has no such check to make: the caller is parent-process code, so
  `aContentId`/`aInnerWindowId` are `0`, and
  `BrowserInferenceExampleModelDownloadGate` can e.g. authorize immediately, or
  show a custom prompt, or any other policy.
- The gate then checks whether the model is already cached (`nsIMLModelHub`) —
  allowing with no prompt if so. Otherwise, for a content-initiated task,
  Speech Recognition's current policy is to show a doorhanger. Only on a real
  Allow does `HWInferenceParent` start the download via
  `nsIMLModelHub::DownloadModel`.

## Testing

The end-to-end example (not intended to be landed) can be compiled, and
`MOZ_LOG=HWInference:5` traces `RecvInstallModel`/`RecvIsModelInstalled` and
related IPC across both the content and browser instances. The browser
smoke-test consumer is exercised end-to-end by
`toolkit/components/ml/tests/browser/browser_ml_hwinference_browser_smoke.js`,
run with:

```
./mach mochitest --headless \
  toolkit/components/ml/tests/browser/browser_ml_hwinference_browser_smoke.js
```

or by running the browser and clicking the menu entry under `Tools`.
