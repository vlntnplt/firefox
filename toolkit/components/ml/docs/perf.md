# How to perftest a model

For each model running inside Firefox, we want to determine its performance
in terms of speed and memory usage and track it over time.

To do so, we use the [Perfherder](https://wiki.mozilla.org/Perfherder) infrastructure
to gather the performance metrics.

Adding a new performance test is done in two steps:
1\. making it work locally
2\. add it in perfherder

## Run locally

To test the performance of a model, you can add in the `tests/browser` a new file
with the following structure and adapt it to your needs:

```javascript
"use strict";

// unfortunately we have to write a full static structure here
// see https://bugzilla.mozilla.org/show_bug.cgi?id=1930955
const perfMetadata = {
  owner: "GenAI Team",
  name: "ML Test Model",
  description: "Template test for latency for ml models",
  options: {
    default: {
      perfherder: true,
      perfherder_metrics: [
        { name: "pipeline-ready-latency", unit: "ms", shouldAlert: true },
        { name: "initialization-latency", unit: "ms", shouldAlert: true },
        { name: "model-run-latency", unit: "ms", shouldAlert: true },
        { name: "pipeline-ready-memory", unit: "MB", shouldAlert: true },
        { name: "initialization-memory", unit: "MB", shouldAlert: true },
        { name: "model-run-memory", unit: "MB", shouldAlert: true },
        { name: "total-memory-usage", unit: "MB", shouldAlert: true },
      ],
      verbose: true,
      manifest: "perftest.toml",
      manifest_flavor: "browser-chrome",
      try_platform: ["linux", "mac", "win"],
    },
  },
};

requestLongerTimeout(10);

add_task(async function test_ml_generic_pipeline() {
  const options = {
    taskName: "feature-extraction",
    modelId: "Xenova/all-MiniLM-L6-v2",
    modelHubUrlTemplate: "{model}/{revision}",
    modelRevision: "main",
  };

  const args = ["The quick brown fox jumps over the lazy dog."];
  await runMLPerfTest("example", options, args);
});
```

Then add the file in `perftest.toml` and rebuild with `./mach build`.

The test downloads models it uses from the local disk, so you need to prepare them.

We provide a script to automate this.

```bash
$ mach python toolkit/components/ml/tests/tools/create_local_hub.py --list-models
Available git-based models from the YAML:

- xenova-all-minilm-l6-v2 -> path-prefix: onnx-models/Xenova/all-MiniLM-L6-v2/main/
- mozilla-ner -> path-prefix: onnx-models/Mozilla/distilbert-uncased-NER-LoRA/main/
- mozilla-intent -> path-prefix: onnx-models/Mozilla/mobilebert-uncased-finetuned-LoRA-intent-classifier/main/
- mozilla-autofill -> path-prefix: onnx-models/Mozilla/tinybert-uncased-autofill/main/
- mozilla-smart-tab-topic -> path-prefix: onnx-models/Mozilla/smart-tab-topic/main/
- mozilla-smart-tab-emb -> path-prefix: onnx-models/Mozilla/smart-tab-embedding/main/

(Use `--model <key>` to clone one of these repositories.)
```

You can then use `--model` to download locally models, by specifying the local
`MOZ_ML_LOCAL_DIR` directory, via the env var or command line argument :

```bash
$ mach python toolkit/components/ml/tests/tools/create_local_hub.py --model mozilla-smart-tab-emb --fetches-dir ~/ml-fetches
Found existing file /Users/tarekziade/Dev/fetches/ort-wasm-simd-threaded.jsep.wasm, verifying checksum...
Existing file's checksum matches. Skipping download.
Updated Git hooks.
Git LFS initialized.
Cloning https://huggingface.co/Mozilla/smart-tab-embedding into '/Users/tarekziade/Dev/fetches/onnx-models/Mozilla/smart-tab-embedding/main...
Cloning in '/Users/tarekziade/Dev/fetches/onnx-models/Mozilla/smart-tab-embedding/main'...
Checked out revision '2278e76f67ada584cfd3149fd2661dad03674e4d' in '/Users/tarekziade/Dev/fetches/onnx-models/Mozilla/smart-tab-embedding/main'.
```

Once done, you should then be able to run it locally with :

```bash
MOZ_ML_LOCAL_DIR=~/ml-fetches ./mach perftest toolkit/components/ml/tests/browser/browser_ml_engine_perf.js --mochitest-extra-args=headless
```

Notice that `MOZ_ML_LOCAL_DIR` is an absolute path to the `root` directory.

## Add in the CI

To add the test in the CI you need to add an entry in

- `taskcluster/kinds/perftest/linux.yml`
- `taskcluster/kinds/perftest/windows11.yml`
- `taskcluster/kinds/perftest/macos.yml`

With a unique name that starts with `ml-perf`.

A test that measures an ONNX model gets **two** entries per platform file, one
per backend: `<name>-wasm` holding the full definition (and a YAML anchor),
and `<name>-native` merging it with `<<:` to override the treeherder symbol
and set `MOZ_ML_BACKENDS: onnx-native`. Copy an existing pair such as
`ml-perf-autofill-wasm` / `ml-perf-autofill-native`.

Example for Linux:

```yaml
ml-perf:
    fetches:
        fetch:
            - ort.wasm
            - ort.jsep.wasm
            - ort-training.wasm
            - xenova-all-minilm-l6-v2
    description: Run ML Models Perf Tests
    treeherder:
        symbol: perftest(linux-ml-perf)
        tier: 2
    attributes:
        batch: false
        cron: false
    run-on-projects: [autoland, mozilla-central]
    run:
        command: >-
            mkdir -p $MOZ_FETCHES_DIR/../artifacts &&
            cd $MOZ_FETCHES_DIR &&
            python3 python/mozperftest/mozperftest/runner.py
            --mochitest-binary ${MOZ_FETCHES_DIR}/firefox/firefox-bin
            --flavor mochitest
            --output $MOZ_FETCHES_DIR/../artifacts
            toolkit/components/ml/tests/browser/browser_ml_engine_perf.js
```

You also need to add the models your test uses (like the ones you've downloaded locally) by adding entries in
`taskcluster/kinds/fetch/onnxruntime-web-fetch.yaml` and adapting the `fetches` section.

Once this is done, try it out with:

```bash
./mach try perf --single-run --full --artifact
```

You should then see the results in treeherder.

## Backends: measuring native ONNX and WASM

Most features request the `best-onnx` backend, which resolves to `onnx-native`
where the platform bundles `libonnxruntime` and silently falls back to the WASM
`onnx` backend everywhere else. Measuring only one of the two therefore tells
you nothing about what a meaningful share of the fleet actually runs.

In CI, every ONNX perf task exists twice: `<task>-native` and `<task>-wasm`,
each pinning one backend through the `MOZ_ML_BACKENDS` environment variable in
`taskcluster/kinds/perftest/*.yml`. One job measures one backend, so a backend
that cannot run **fails its own job** — orange in Treeherder, no soft-error
metric to cross-reference — without costing the other backend's numbers. A
green job is a job that measured what its name says.

Metric names carry the backend tag, so each backend is its own perfherder
series and the two can be compared directly:

```
SMART-TAB-TOPIC-NATIVE-model-run-latency
SMART-TAB-TOPIC-WASM-model-run-latency
```

Locally, without `MOZ_ML_BACKENDS`, `runMLPerfTest()` probes the runtime once
and measures the most preferred backend that can run: native where the runtime
loads, the wasm fallback otherwise. To measure both in one session, or to
reproduce a specific CI job, set the variable yourself:

```bash
MOZ_ML_BACKENDS=onnx-native,onnx ./mach perftest <test.js> ...
```

Do not set `backend` in your `PipelineOptions` — `runMLPerfTest()` overrides it per
iteration. For a test whose backend is genuinely fixed (llama.cpp, OpenAI,
static embeddings), pin it instead:

```javascript
await runMLPerfTest({ name, options, request, backends: ["llama.cpp"] });
```

`MOZ_ML_BACKENDS` overrides even a pinned list: the CI task name is the
contract for what its job measures.

### Tests that drive a feature rather than a pipeline

`runMLPerfTest()` owns engine creation, so it only fits a test that measures one
pipeline. A test that drives a whole feature (MLSuggest, semantic history
search, smart tab clustering) calls `runMLPerfTestForEachBackend()` directly,
passes `backend` down into whatever options it builds, and prefixes its own
metric names with `tag`:

```javascript
add_task(async function test_my_feature() {
  await runMLPerfTestForEachBackend({ name: "MY-FEATURE", run: measureMyFeature });
});

async function measureMyFeature({ backend, tag }) {
  MyFeature.OPTIONS = { ...MY_OPTIONS, backend };
  // ...
  reportMetrics({ [`MY-FEATURE-${tag}-model-run-latency`]: latencies });
}
```

Watch out for features that carry their own native→wasm fallback (MLSuggest
does): pin the fallback options to the same backend, otherwise a native failure
quietly reports wasm numbers under the `NATIVE` tag.

Declare the metric names in `perfherder_metrics` **without** the feature
prefix. mozperftest keeps a subtest only if a declared name appears in it as a
substring, and the backend tag sits between the feature name and the metric, so
`AUTOFILL-model-run-latency` no longer matches `AUTOFILL-NATIVE-model-run-latency`
while `model-run-latency` does.

### When a backend cannot run a model

The job fails. There is no soft-failure path: mozperftest stops before
emitting `PERFHERDER_DATA` when a test fails, and since the job only measures
that one backend, there are no other numbers to protect — the orange job *is*
the report.

If a backend can **never** run a given model, do not schedule that variant:
drop it from the kind.yml with a comment saying why, and add the same reason
to `DECLARED_SKIPS` in `ml_perf_report.py` so the report shows the hole as
deliberate instead of flagging it. An undeclared hole is treated as a bug.

Two known native gaps to expect if you add a test using them: the native
backend cannot load models that ship their weights as external data
(`use_external_data_format`, i.e. a `.onnx_data` file — it has no way to hand
those bytes to the runtime), and text-to-speech fails with
`Unsupported device: "wasm"` because the vocoder is loaded with the WASM
default device.

To pin a whole run to one backend, for example to reproduce a WASM-only
platform on a machine that has the native runtime:

```bash
MOZ_ML_BACKENDS=onnx ./mach perftest <test.js> \
    --hooks toolkit/components/ml/tests/tools/hooks_local_hub.py \
    --mochitest-extra-args headless
```

### Where the runtime actually loads

`browser_ml_ort_availability_probe.js` is a **regular** mochitest, deliberately
without any `skip-if`, so it runs on every CI configuration — including the
asan, tsan, debug, x86 and Android jobs that the native ONNX tests exclude. It
asserts that `AppConstants.MOZ_ONNX_RUNTIME` (what the build claims) matches
`InferenceSession.isAvailable()` (what a real `dlopen` does), and emits one
machine-readable `ML_ORT_PROBE` line per job.

If it fails on a configuration, that is a finding. Do not add a `skip-if`.

### Getting a report out of a try push

```bash
./mach try preset onnx-perf
./mach python toolkit/components/ml/tests/tools/ml_perf_report.py \
    -- --revision <rev> --output ml-perf-report.md
```

The report leads with a coverage verdict: every scheduled ML job must have
reported metrics, and every feature must have both a NATIVE and a WASM series
wherever it ran. Anything that violates either rule is listed first, before
any number — a failed job, a green job with no data, or a variant that was
never scheduled. The native-vs-WASM comparison tables come after. The same
tool reads local logs, which is the quickest way to check your test before
pushing:

```bash
./mach python toolkit/components/ml/tests/tools/ml_perf_report.py -- --logs <logfile>
```
