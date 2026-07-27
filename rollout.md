# Rolling out the accelerated inference engine

[Valentin Pollet](mailto:vpollet@mozilla.com), Jul 2026. Kept current with
the stack; the per-commit messages carry the fine detail.

## Goal

Move llama.cpp text generation to the HWInference utility process, behind
the C++ protocol described in [the design doc](./ipc_design.md), while
holding two properties at every step:

- Link Preview and the existing generation tests produce the same
  observable results on the new engine as on the old one. This is measured
  by tests, not assumed.
- Every commit is small enough that one reviewer group can check it without
  holding the rest of the stack in their head.

Most of the work is plumbing: IPDL, actor wiring, a parent-side surface,
telemetry. That is the kind of work to hand to Claude, provided the process
around it is strict enough that the result can be trusted and audited. This
doc describes that process and the commit stack that came out of it.

## Working method

The rollout applies [Greg's principles](./high_quality_principles.md) as
follows:

- **Tests check behavior visible from outside, never the internals of the
  code being replaced.** The tests describe what Link Preview does and what
  the perftests measure. We reuse the ones already in the tree, add cases
  in the same style where coverage was thin, and require the new engine to
  pass all of them unchanged. There is nothing to game: the tests do not
  know which engine serves them.
- **Every new pattern is grounded in tree precedent.** Before writing a
  protocol, actor, or binding, we find how an existing module does the same
  job, cite it in the commit message (file and line, verified against the
  tree), and explain any departure. The first commit using a pattern
  becomes the example the rest of the stack follows.
- **Small commits, one reviewer domain each.** IPDL goes to IPC reviewers,
  bindings to WebIDL peers, the Glean point through data review.
- **Machine feedback first.** The IPDL compiler, WebIDL codegen, linting,
  and the test harness catch problems before a human reads the diff.

## The configuration rule: the surface is what is served

The old option plumbing had layers that silently dropped options or changed
defaults. Concretely: the old pipeline computed KV-cache dtypes,
flash-attention and batch settings from PipelineOptions, then forwarded
none of them into initialization (`LlamaCppPipeline.initialize` passed only
`useMmap`, `useMlock`, `nCtx` and the thread counts to the runner). Every
consumer always ran on the runner defaults; the options callsites set were
fiction.

The stack replaces that with one rule: an option that is in the surface is
honored verbatim, and an option that cannot be honored is removed from the
surface. There is no detection or warning machinery for unhonored options —
that would be another layer of the same problem. In practice:

- Callsites were fixed first, on the old code, to stop requesting options
  that were never honored (Bug 02). No observable change, which the golden
  tests confirm.
- The wire and the WebIDL surface carry only knobs the backend actually
  reads: `contextSize`, `numThreads`, `batchSize`, `ubatchSize`, plus the
  per-request generation knobs. There is no KV-cache or flash-attention
  knob until the acceleration work honors one for real. (Requesting
  quantized caches with flash attention crashes the vendored llama.cpp on
  qwen3-class models today; that finding is an input to the backend
  rewrite.)
- Defaults live in exactly one place, the WebIDL dictionaries, set to what
  the old engine effectively ran (batch 2048/512; `numThreads = 0` means
  the same physical-core count MLEngineChild picks). Wrappers forward what
  the consumer set and default nothing themselves.
- The deeper PipelineOptions cleanup — dropping the fields no remaining
  path honors, and honoring the ones that stay end to end — is follow-up
  work after the old path is deleted; it is deliberately not part of this
  stack.

wllama is decommissioned, which is what made this clean: the batch/cache
knobs documented as wllama-only lost their last consumer, and `best-llama`
resolution collapsed to llama.cpp.

## The tests that gate the move

A tests-only commit (Bug 03) sits near the base of the stack, ahead of
everything that moves code. It had to build and pass on the old engine
first, and it is what makes the rest of the stack checkable. What it
covers:

- **Golden samples.** Two tiers. The strong one is
  `browser_ml_llama_smollm2_smoke.js`: the real Link Preview model
  (SmolLM2-360M Q8) driven greedily through the real engine, output pinned
  as text plus SHA-256 — pins that hold across CPU architectures, per the
  test's own notes on SmolLM2's top-1 margins — plus determinism,
  prompt-sensitivity, and structural checks. It runs under the perftest
  harness (the GGUF is served from MOZ_FETCHES_DIR by the local-hub hooks),
  so it covers all three desktop platforms on perf try rather than every
  push. The every-push tier is its mochitest sibling
  `browser_ml_native.js`, the same checks against the in-tree TinyStories
  model. Bug 03 extends both and pins the run options (sampler, thread
  count, nCtx) explicitly — options Bug 02 first made honest — so a default
  drifting during the migration fails a pinned test.
- **Link Preview end to end.** The existing Link Preview browser tests stub
  the engine out (`sinon.stub` on `createEngine` / `generateTextAI`), so
  they pass whether or not the engine works. Bug 03 adds the missing case:
  real `createEngine`, checked-in model, assert a preview is produced. Once
  the routing pref exists (Bug 08), this one test runs against both engines
  with no edits.
- **Telemetry shape.** Reset FOG, drive the canonical workflow (create,
  `run`, `runWithGenerator`, a deliberate failure, terminate) against the
  real engine, then walk the `firefox_ai_runtime` metrics and
  `testGetValue()` each one. Coverage is kept as one table (metric,
  workflow step, shape assert) so a reviewer can diff it against
  `metrics.yaml` at a glance. It asserts presence and shape only:
  distribution counts match the number of runs, event extras parse sane,
  labels carry the expected taxonomy — the label set staying identical
  across engines is itself part of the goal. Timing values are not
  asserted: JS-measured and C++-measured timings will never be equal, so
  value drift shows up as small percentage deltas in the perf compare,
  where it is reviewed deliberately. Both run modes are exercised, since
  chunk metrics only exist when streaming.
- **Crash handling.** The existing crash test becomes process-agnostic:
  the serving process differs by engine, so it races
  `ipc:content-shutdown` against `ipc:utility-shutdown` (precedent:
  `crashSomeUtility` in `ipc/glue/test/browser/head.js`) with identical
  assertions on both. No assertion anywhere in the suite depends on which
  process serves the request — that is what lets the flip commit change no
  tests.

One pre-existing assertion changed, with the reason in Bug 03's message:
the invalid-sampler rejection now matches the sampler name rather than the
name of the rejecting class, which the engine swap replaces. It is the only
pre-existing expectation the stack re-targets.

Performance has its own gate: `browser_ml_llama_summarizer_perf.js`, the
perfherder test measuring latency, memory, and token and character speed
over tiny/medium/big articles. It was skipped in the manifest (Bug
1952456); Bug 03 revives it. The comparison is `mach try perf` between the
base of the stack and its tip: we expect no regression, and the expected
tiny telemetry-value deltas get eyeballed there too. There is no stored
baseline number to maintain.

Building these tests found a real bug before anything moved: the
cross-mode check caught `LlamaCppPipeline.run()` returning
`[object Object]` chunks as `finalOutput` — the only shipping consumer
streams, so nothing had ever exercised `run()`'s return value. That product
fix is Bug 01, ahead of the tests.

## Building on Bug 2051970

Paul's stack introduces the HWInference process and its model machinery:
D268403 (process introduction), D314420 (process lifetime), D268408
(ModelHub XPCOM interface), D268412 (sandbox policy), D309866/D309867
(model IPC), D268411 (model-file transfer), D311101 (Windows DLL preload),
D313816 (documentation). It is in review, not landed. We do not wait for
it: the base of this branch is Paul's stack itself, pulled from Phabricator
at pinned diffs, excluding only his WIP smoke-test prototype (D313227) —
the WIP never enters our base. When Paul pushes an update, we re-pull and
rebase; a fix we need in his code is a separate `PROVISIONAL` commit, to
be proposed on his revision and dropped when he takes it.

Where that convergence stands (Jul 27): Paul's latest push took the
relaunch fix — a cached `HWInferenceParent` kept reporting `CanSend()` for
a dead process, blocking every respawn; D268403 now evicts stale instances,
with a gtest — and added D314420, which makes process lifetime a
per-instance-key keep-alive count on UtilityProcessManager. Our manager
holds one of those keep-alives (see Bug 05). One provisional commit
remains: `PHWInferenceManager.ipdl` is listed as preprocessed IPDL but
contains no preprocessor directives, which fails the build. Fixed only
inside the excluded WIP; to be proposed on D268403.

One more change to Paul's code rides inside Bug 07 rather than as its own
commit: the Blob-to-`FileDescriptor` extraction is a file-local static in
`HWInferenceParent.cpp`, and TextGenerator's `create()` needs the same
extraction, so Bug 07 hoists it unchanged into a shared
`ml/ipc/ModelFileUtils` that both callers use. To be proposed on D268411;
that part of Bug 07 drops when the revision takes it.

A cherry-pick of D177649 (profiler shutdown data on utility processes) also
rides at the base until it lands.

Two design positions in this stack are ours to defend in review:

- The browser side gets its own manager protocol,
  `PHWInferenceBrowserManager`, with the main process holding the parent
  side. Paul's WIP prototypes the opposite orientation (a utility-parented
  browser manager, matching his content manager). The question is open;
  whichever way it settles, the managed `PTextGeneration` subtree is
  unaffected, and Bug 04 is the only commit that rebases non-trivially.
- Model files reach the process as a `FileDescriptor` pushed by the parent
  at generator construction. The utility process never requests files by
  name.

One finding travels as a review note rather than a commit: D268408's
`MLModelHubService._getModelHub()` passes
`{MODEL_HUB_ROOT_URL, MODEL_HUB_URL_TEMPLATE}` to a ModelHub constructor
that expects `rootUrl`/`urlTemplate`, silently dropping both prefs.

## The stack

Layout follows tree precedent and separates the wire from what drives it:
everything IPC — protocols, ipdlh types, both actors, the fd helper — lives
in `toolkit/components/ml/ipc/` (precedent: `dom/media/ipc`,
`dom/webgpu/ipc`); the chrome-facing WebIDL driver `TextGenerator.{cpp,h}`
lives with the backend it drives in `backends/llama/`, where LlamaRunner
lived; and the MLEngine-shaped JS drivers live one per capability in
`engines/` (`TextGenerationEngine.sys.mjs` now, an embeddings engine over
its own protocol later — there is deliberately no engine-to-rule-them-all).
Naming splits the same way: the protocol is the capability
(`PTextGeneration`, actors `TextGenerationParent/Child`), the WebIDL
interface is the object a caller holds (`TextGenerator`). Our C++ uses the
`mozilla::hwinference` namespace, matching the base.

Bug numbers are placeholders (01, 02, ...) to be substituted with Bugzilla
IDs at submission. Commits are ordered so each one is green in CI and
nothing user-facing changes until the flip.

**Bug 01 - LlamaCppPipeline.run() returns text again.** `run()` accumulated
chunk objects instead of their text. A product fix on existing code; lands
now. Reviewer: ML team.

**Bug 02 - Callsites request only options the engine honors.** Link
Preview and the ML test callsites stop requesting `kvCacheDtype`,
`flashAttn`, `numBatch`/`numUbatch` — options the old engine provably never
honored (see the configuration rule). No observable change, which the
golden tests confirm. Touches `LinkPreviewModel.sys.mjs` and the ML test
callsites. Reviewer: ML team.

**Bug 02b - Two pre-existing test bugs.** The crash test's only assertion
sat inside its own catch block, so it could never fail; and the golden pin
keyed on macOS when it always meant the CPU architecture. Both fixes on
existing code, ahead of the new work. Reviewer: ML team.

**Bug 03 - The tests described above.** Green on the old engine, so the
perf and golden comparisons have a base to run against. Grounded in
`browser_ml_llama_smollm2_smoke.js`, `browser_ml_native.js`, and
`browser_ml_telemetry.js`. Touches `toolkit/components/ml/tests` and
`browser/components/genai/tests`. Reviewer: ML team.

**Bug 04 - Browser-side HWInference manager.** `PHWInferenceBrowserManager`,
bootstrapped the way Paul bootstraps his content manager — the main process
creates the endpoint pair and hands one end over `PHWInference`, mirroring
`NewContentHWInferenceManager` — but with the orientation flipped: the main
process keeps the parent side (precedent: PSocketProcess, whose `child:`
section declares `PHttpConnectionMgr`, `PDNSRequest` and friends — the main
process constructing actors into a utility-type process). Bootstrap only;
the lifecycle policy lives with Bug 05. Touches the IPDL directory and the
actor registration. Reviewer: #ipc-reviewers.

**Bug 05 - PTextGeneration, echo-stubbed, and the process lifecycle.** The
protocol under the manager (`manages PTextGeneration`, constructor on
`child:`) and its payload structs and unions, with no llama logic behind it
yet: the actors compile and echo. Payload shape per the design doc
(precedent: PGamepadEventChannel's event union; PRemoteMediaManager is
cited only for its manager-hands-out-workers tree — its orientation is
inverted, so it is never the example for message direction). The wire
carries the full served option set from birth and nothing else.

Construction is acquisition: loading the model is part of constructing the
actor, and the one-shot `Ready(LoadResult)` reports the outcome (precedent:
PRemoteWorker's `Created`/`Error` pair, collapsed into one result union
since runtime errors already travel in the `Generate` reply). IPDL forbids
async constructors with return values, so the child-pushed one-shot is the
closest expressible form; the parent side exposes it as `WhenReady()` and
never hands a generator out before it resolves.

One generation is in flight per generator, enforced where the hazard lives:
the chrome surface rejects an overlapping `generate()` with
`InvalidStateError` and sets its delta handler per call. The wire matches
that contract instead of hedging against its absence: no request index, and
the event union collapsed to `Delta(text)` because nothing consumed any
other channel (the surface-is-served rule, applied to the protocol itself).
Cancel targets the pending Generate through a flag on the generation, not
on the actor, so it cannot bleed into the request that follows.

The manager is a browser-session singleton and expresses the process
lifetime through the keep-alive API from D314420: `Create()` acquires a
keep-alive on the browser instance key, and when the last generator dies
the manager releases it, which shuts the process down unless another
consumer of that key still holds one. `GetOrCreate()` hands every creator
the same actor and joins a launch already in flight, so the manager's
managed set is the process-wide generator count. Two races are closed
deterministically: a create arriving while the release decision is pending
takes a reservation the decision honors, and the cached singleton retires
at the decision rather than at ActorDestroy, so a late create launches a
fresh process instead of joining a dying one. ActorDestroy also releases
the keep-alive, so a crashed process cannot leak the count, and bootstrap
failures after launch release it rather than leaking an idle process.

The commit ships with a gtest that round-trips a real browser-keyed
process, which is also what exercises Bug 04's bootstrap end to end.
Reviewer: #ipc-reviewers.

**Bug 06 - LlamaBackend behind TextGenerationChild.** The child owns a
`LlamaBackend` and drives it on a per-generator TaskQueue: the model loads
at construction from the fd (reported through `Ready`),
`Generate`/`Clear`/`Cancel`, deltas sent from the actor thread. The child
is structured, not monolithic: wire-to-backend conversion is pure free
functions, each in-flight request is a `Generation` object with one job per
method, and thread hops go through a single helper owning the `CanSend()`
guard. The threading contract is grounded in RemoteDecoderParent
(per-instance TaskQueue handed out by the manager, Recv on the manager
thread, results hopping back via `->Then(mManagerThread, ...)` under
`CanSend()` guards).

The child hard-sets `useMmap=false` for v1 — there is no mmap knob in the
surface, which keeps the seccomp question out of this rollout entirely; the
golden outputs are unaffected since mmap changes model loading, not logits.
Streamed deltas and the reply are valid UTF-8 by construction, using the
same streaming `mozilla::Decoder` shape LlamaRunner used (the contract from
bug 2043430). No new llama.cpp API calls were needed, so
`config/external/mozinference/mozinference.symbols` is untouched; any
future call must be added there or it compiles clean and fails only at
runtime — a trap the backend rewrite needs to watch for. This is where
LlamaRunner's hand-written SPSC queue and double-buffered promise stop
being needed: message order on the actor gives the same serialization.

Exercised by direct-drive gtests, which also pin the clean failure paths
(an invalid descriptor; a readable file that is not GGUF, rejected by the
magic check). These run unsandboxed; the first sandboxed exercise is Bug
08. Touches `backends/llama` and the child actor. Reviewer: ML team plus
IPC for the threading contract.

**Bug 07 - TextGenerator, the WebIDL surface.** The small parent-side class
that implements the MLEngine surface (`run`, `runWithGenerator`,
`terminate`) on top of `TextGenerationParent`, exposed to chrome JS through
WebIDL. Grounded in WindowGlobalParent (`WindowGlobalActors.webidl`,
ChromeOnly) — with the honest caveat that no in-tree ChromeOnly interface
wraps a utility-process-backed parent actor today, so this commit is a
first and the WebIDL review is a design conversation, not pattern-matching.
The dictionaries are the single defaults authority, set to the effective
old-engine configuration; the `numThreads = 0` (auto) default resolves here
— and only here — to the same physical-core count MLEngineChild picks, so
the wire always carries a concrete count and the sandboxed child never
samples core topology. `create()` resolves only once the construction-time
load reported `Ready` and rejects with the load error otherwise, so load
failures are creation failures, as on the old engine. There is no
process-lifetime API on the surface: teardown follows the last generator
structurally (Bug 05), not through consumer bookkeeping. `create()` turns
the ModelHub blob into the file descriptor the generator constructor
carries, using HWInferenceParent.cpp's extraction helper — this commit
hoists it into the shared `ml/ipc/ModelFileUtils` both callers use (see
the coordination section; to be proposed on D268411). Built but not
wired into `createEngine` yet. Touches the WebIDL bindings and the parent
actor (impl in `backends/llama/`). Reviewer: WebIDL peers for the binding,
ML for the surface.

**Bug 08 - Route llama.cpp to HWInference behind a pref.** `createEngine`
forks llama.cpp to the new process when the capability is present and the
pref is on — the fork point exists today in
`MLEngineParent.chooseBestBackend`. Default off, old path still serves. The
routed path is `engines/TextGenerationEngine.sys.mjs`, an MLEngine-shaped
adapter over the `TextGenerator` surface: it reads exactly the
PipelineOptions fields it honors, defaults nothing, does no engineId reuse
(every `createEngine` constructs a fresh engine; process lifetime is the
manager's, not the wrapper's), and builds its ModelHub through the one
shared factory, so the allow/deny list, the caller's abort signal, and the
download notifications behave on this path exactly as on MLEngine's.

With the pref in place, the Bug 03 suite runs against both engines from the
same test files, including the telemetry walk, and this is the first time
the new path runs under the real sandbox in CI. Two tests pin the failure
story: kill the utility process mid-`Generate` and assert the failure
reaches the consumer and the next request respawns the process; browser
shutdown during a generate exercises the same ActorDestroy path.

Measurement moves to the C++ side: the wire carries the child's
`Usage`/`Timings` and the wrapper projects them into the MLEngine metrics
shape; Glean emission stays in the single parent-side point. This commit
also had to teach the perf harness about the new process, because it
silently measured nothing otherwise: the shared process lookup only matched
the inference content process, and the speed series counted per-chunk token
arrays that text-only deltas never carry — mozperftest records zeros
without failing, so a passing perftest does not imply a measuring one. The
lookup now takes an explicit process type (an ONNX engine in the inference
process and llama in HWInference can be alive at once, and first-match
attribution would charge one path with the other's resources) and the speed
series fall back to the engine's C++-measured token counts.
`prompt-charactersSpeed` exists only where the old engine echoes the
prompt, the one series the new path does not report. Additive telemetry the
design doc wants (load breakdown, memory high-water) waits on data review
as a follow-up. Touches `MLEngineParent`, the engine routing, and the new
wrapper. Reviewer: ML team.

**Bug 09 - Make HWInference the only llama.cpp path (held).** The default
flip and the deletion of the old path travel together, because separately
neither tells a reviewable story: the routing pref goes away,
`createEngine` sends llama.cpp to HWInference unconditionally, and the code
the pref kept alive goes with it — `LlamaCppPipeline`, `LlamaRunner` with
its SPSC queue and double-buffered promise, and `LlamaRunner.webidl`. The
dictionaries LlamaBackend consumes as its C++ options types move verbatim
to `dom/chrome-webidl/TextGenerator.webidl`, next to their only remaining
surface, so nothing Runner-flavored survives the deletion. The Bug 03 suite
is engine-agnostic by construction, so test changes are deletion-driven
only. wllama-declared PipelineOptions fields keep their declarations; their
removal belongs to the wllama decommission's own cleanup.

Written now so reviewers can read the end state; held until the routed path
has soaked on Nightly behind Bug 08's pref, watched on the existing Glean
dashboards and crash reports. V1 lifecycle stays minimal: Link Preview
keeps its create-generate-teardown shape, there is no idle timeout, and the
process goes away when the manager releases its keep-alive after the last
generator — process spawn is measured to be cheap against this workload, so
keeping it warm buys nothing yet. The protocol's warm paths (`Clear`,
appended turns) stay as headroom for later workflows, unused by v1.
Reviewer: ML team.

## What runs before a push

Before anything goes to try, everything a `mach try` or `mach try perf` run
would exercise runs locally at the stack tip, with the logs kept in
`artifacts/`: the full ML browser suite on both engines, the SmolLM2 smoke
perftest through the local hub, the summarizer perftest for completion
(local perf numbers are not the gate; CI reference hardware is), a full
build, and lint on every touched file. The user-triggered try push then
confirms platform coverage — the Windows sandbox, the mac policy,
debug/asan, the perf compare on reference hardware — rather than
discovering basic breakage.

## Landing order and risk

- Bugs 01-03 stand on existing code and can land now.
- Bugs 04-08 are reviewable now; landing stays serialized behind Paul's
  stack. Bug 09 additionally needs D311101 on Windows: without
  the `mozinference.dll` preload, the process cannot load the backend under
  the sandbox at all.
- Development notes while the base is unlanded: on Linux, D268403's generic
  utility sandbox policy is enough to develop against; on Windows, develop
  with the utility sandbox disabled until D311101 lands.
- The two design positions in the coordination section (browser-side
  manager orientation; parent-pushed fds) are ours to defend in review.
- The LlamaBackend rewrite is a separate stack, sequenced after this one
  matches the old engine on CPU and before the acceleration and batching
  work. It reuses the same test gate. Inputs to it recorded here: the qwen3
  small-ubatch crash; the quantized-cache-plus-flash-attention crash; and a
  corrupt GGUF crashing the vendored llama.cpp (`std::map::at`) instead of
  returning an error — which is why Bug 07 ships no browser-level
  corrupt-model rejection test, while the clean failure paths that do exist
  are pinned by the Bug 06 gtests.
- One previously undocumented contract is now stated: Link Preview keys its
  flush on the terminal empty chunk the old engine emits at end of stream.
  The cross-engine test (byte-identical streamed text and `finalOutput`
  across both engines in one run) forced `TextGenerationEngine` to
  reproduce it, and it is now part of the MLEngine contract the adapter
  implements.
