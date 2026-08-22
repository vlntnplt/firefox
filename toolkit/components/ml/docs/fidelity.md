# Model fidelity tests

A model is trained and evaluated in Python, then exported, quantized and run
in Firefox by a different runtime, on hardware the evaluation never saw. A
fidelity test checks that the model Firefox ships, in the configuration it
ships in, still produces the outputs the evaluation produced. The evaluation
measures how good the model is; the fidelity test measures whether Firefox
reproduces the evaluation.

Each test replays a small sample of the evaluation, called a trace, through
the feature that uses the model, and compares the model's outputs and the
feature's conclusions with what the trace recorded.

The team that trains and exports a model owns its trace and writes it out of
the evaluation run, in the same configuration the evaluation used. Firefox
defines the trace format and reads the file. Traces currently ship as test
support files; the plan is to host them next to the models and fetch them
together.

## Trace format

A trace is one JSON document with `snake_case` keys and vectors as plain
arrays. Everything that belongs to a model is keyed by a role name that the
test chooses, such as `encoder` or `classifier`. A feature that ships two
models has two roles under `models`, under `tolerances`, and in every
example's `outputs`.

```json
{
  "schema_version": 1,

  "models": {
    "encoder": { "model_id": "mozilla/form-autofill-embed", "revision": "v0.3.1" },
    "classifier": { "model_id": "mozilla/form-autofill-head", "revision": "v0.3.1" }
  },

  "reference": {
    "runtime": "onnxruntime",
    "runtime_version": "1.29.0",
    "device": "cpu",
    "dtype": "fp32",
    "revision": "v0.3.0"
  },

  "tolerances": {
    "embedding_cosine_min": { "encoder": 0.995 },
    "embedding_norm_rel_max": { "encoder": 0.01 },
    "result_accuracy_min": 1.0
  },

  "request_options": {
    "encoder": { "pooling": "mean", "normalize": false }
  },

  "examples": [
    {
      "id": "address.firstname",
      "input": {
        "text": { "current": "firstname first name enter your first name",
                  "previous": "",
                  "next": "lastname last name surname family name" }
      },
      "outputs": {
        "encoder": { "current": [-0.2663476, -0.2527834, "..."],
                     "previous": ["..."],
                     "next": ["..."] }
      },
      "result": "given-name"
    }
  ]
}
```

### `models`

The models the reference outputs were produced with, keyed by role. The test
compares each model id and revision with what Remote Settings ships. A model
bump without a new trace fails the test.

### `reference`

The configuration that produced the reference outputs:

- `runtime` is `onnxruntime` with a `runtime_version` when the model was run
  outside the browser, or `firefox` with the `backend` that served it when
  the trace was produced by driving Firefox.
- `dtype` is the weight precision, in the transformers.js spelling: `fp32`,
  `q8`, and so on.
- `revision`, when present, names the model revision the reference weights
  came from. It is used when the reference is an older fp32 export of the
  quantized revision that ships.
- `note` is free text for anything else a reader needs to know.

The reference decides what the trace can answer. A reference produced
outside the browser measures the export, the quantization, the runtime and the
hardware together. A reference produced by driving Firefox measures what
differs between two Firefox configurations, for example quantized weights
against fp32 weights, or one machine against another; the runtime is on both
sides of that comparison and its own cost stays invisible. Use an external
reference when one can be produced, and explain in `note` when it cannot.

### `tolerances`

How far the shipped configuration may drift from the reference. The team that
evaluated the model sets these, because they know what the downstream task
absorbs.

A tolerance on a model's output is keyed by role, as `"kld_max": { "intent":
0.05 }`. A tolerance on what the feature concluded is a plain number, as
`"result_accuracy_min": 1.0`. A name ending in `_min` is a floor and one
ending in `_max` a ceiling. The test reports each measurement under the name
of its tolerance.

The test lists every tolerance it reads. The harness fails a trace that is
missing one of them, and a trace that declares a tolerance outside the list.

### `request_options`

Per role, the options the feature passes with each run that are part of the
model contract, for the tests that check them. For the autofill encoder that
is mean pooling without L2 normalization, because the classifier was trained
on unnormalized vectors. The test compares each option with what the feature
requested on the recorded run.

### `examples`

Each example has:

- `id`, a short unique name used in log messages;
- `input`, what the feature was driven with, in the form the model consumes.
  For a text model this is the text; for the pdf.js captioner it is an image
  as a PNG data URL, at the size the feature would hand to the model.
- `outputs`, the reference output of each role that served the example;
- `result`, what the feature concluded: a field type, an intent, a group
  name, a caption.

An output takes one of three shapes:

- a vector, compared by cosine similarity and by magnitude. When one role
  embeds several inputs per example, as the autofill encoder does with the
  current, previous and next field, the vectors are keyed by input slot. A
  text shared by several examples is stored once per example.
- logits, as `{ "dims": [1, 8], "logits": [...] }`, compared by KL
  divergence. Give this shape when the feature's pipeline returns a
  post-processed result such as a top-1 label and score. When the winning
  class differs between two configurations, their scores describe different
  classes and cannot be compared; the logits can.
- for a vocabulary too large to store, the reference's top-k logits plus the
  full log-partition, as `{ "vocab": 32100, "top_indices": [...],
  "top_logits": [...], "log_partition": 9.41 }`. The divergence computed from
  this is exact for the terms retained, and the terms left out have no mass
  in the reference and contribute nothing.

A role with more than one stage nests them. The topic model and the
captioner are encoder-decoders, and their output holds `encoder_pooled`, the
encoder output averaged over positions, and `step0`, the first decoder step
in the truncated shape.

The test asserts `result` when the trace's examples were filtered for a
clear margin (see "Choosing examples") and reports it otherwise.

### Several models for one task

A feature can ship several models and pick one at runtime. The Smart Window
intent classifier picks by home region. Each of its examples carries the
region in its input and an output for the one model that served it:

```json
{ "id": "fr.essence",
  "input": { "query": "prix du carburant aujourd hui", "region": "FR" },
  "outputs": { "intent_en_fr": { "dims": [1, 2], "logits": ["..."] } },
  "result": "search" }
```

The test sets the region, lets the feature choose, and fails if a different
model answers.

## Metrics

Embeddings are compared by cosine similarity and by magnitude. Cosine ignores
scale. A pooling or normalization setting applied differently than the trace
records can change the length of a vector and keep its direction, and the
magnitude check catches that.

Distributions are compared by KL divergence, in nats, computed from the
softmax of the logits. It compares whole distributions and has the same unit
for every model.

## Choosing examples

A fidelity trace wants examples the model decides with a clear margin. An
example whose top two classes are close flips its label on a numeric
difference too small to indicate a problem, and the test becomes
intermittent.

Reject examples whose top two logits are close before writing the trace. The
autofill trace was filtered at a margin of 2.0, against a largest observed
logit movement of 0.31 between fp32 and int8. The filtering happens before
the trace is written; it is what makes `result` reliable enough to assert.

A trace does not need to be large. Every example is an inference on CI
hardware, and a handful of well-chosen ones is enough.

For the autofill trace, use `""` for `previous` or `next` at a form boundary.
The encoder maps the empty string to a fixed non-zero embedding, which is
what the classifier was trained on.

## Writing a test

Tests live in `toolkit/components/ml/tests/browser_models/fidelity/`, one per
slot, named `browser_ml_fidelity_<slot>.js`, with the trace next to it as
`<slot>_trace.json`. A slot is a feature's code path together with the models
Remote Settings ships for it. The tab strip and Smart Window run the same
topic-model code against separate Remote Settings entries, and each has its
own test; a model bump on one side leaves the other unchanged.

1. Write the trace out of the evaluation run.
2. Add the trace to `support-files` in `fidelity.toml` and add the test file.
3. Call `recordEngineRuns()` before the feature creates its engines. From then
   on every engine keeps what it was asked and what the model returned.
4. Call `loadTrace(file, { tolerances })`, listing every tolerance the test
   reads.
5. Drive the feature's own entry point, to cover the code between the browser
   and the model. The autofill test calls `FormAutofillML.detectFields`, which
   covers the context split, the windowed feature layout and both engine hops.
6. Take `runCount(engineId)` before the call and read the runs back with
   `runsSince(engineId, mark, ...)` after it. Pull tensors out of a run with
   `modelOutputOf(run, ...)`.
7. Compare with `compareEmbeddings`, `compareDistributions`,
   `compareTruncatedDistribution` or `compareSeq2SeqRun`.
8. Finish with `reportFidelity`. Key each checked measurement by its
   tolerance and wrap it in `bounded(trace, path, value)`. A plain number
   under a name of its own is reported without being checked.

A second slot for a model shape that already has a test is shorter. A second
topic model is one call to `runTopicModelFidelity` with its own trace, engine
and entry point.

Two constraints of run recording:

- Only the ONNX pipeline records model outputs and reports resolved options.
  A feature on another backend can compare `result` only.
- A recorded run keeps the first run of each session the model declares. For
  an encoder-decoder that is the encoder and the first decoder step. Later
  decoder steps depend on the tokens generated before them, and once two
  configurations pick different tokens their later steps answer different
  questions. Two recorded runs cannot be in flight on the same engine; drive
  examples one at a time.

## Running a test locally

```
./mach mochitest --headless \
  toolkit/components/ml/tests/browser_models/fidelity/browser_ml_fidelity_formfill.js
```

The test downloads the shipped models from `model-hub.mozilla.org` on first
use, up to about 200 MB for a slot, and reads the live Remote Settings
collection. With `MOZ_UPLOAD_DIR` unset the report is logged.

To validate a model that has not shipped yet, set
`MOZ_ML_FIDELITY_UNPINNED=1`. The trace may then name models Remote Settings
does not serve; everything else is still checked. This is the order for a
model bump: validate the trace and the test unpinned, ship the model, land
the trace.

## CI

The tests run in the `mochitest-browser-chrome-ml-models` suite on hardware
pools, on every platform Firefox ships on. Which kernels onnxruntime
dispatches to depends on the CPU, and a docker pool is a single machine
specification. Coverage of architectures, instruction sets and core counts
comes from the set of machines the suite runs on.

Each run:

1. records the resolved options of every engine. A backend reports only the
   options it reads. A requested option the backend does not read is listed
   as `dropped`, and one served with a different value as `transformed`. The
   usual transformation is backend resolution, such as `best-onnx` becoming
   `onnx-native`;
2. compares the model's outputs with the reference;
3. compares what the feature concluded with the trace's `result`.

`request_options` are checked on both sides. The request side compares each
recorded option with what the feature passed on the recorded run. The effect
side is the vector comparison: a different pooling changes the direction of
the vector, which cosine catches, and a different normalization changes its
length, which magnitude catches.

The Treeherder log carries one summary line per test:

```
Resolved backend on this machine: onnx-native
FIDELITY onnx-native: embedding_cosine_min.encoder=0.999040 embedding_norm_rel_max.encoder=0.001850 result_accuracy_min=1.000000 results_matched=19
```

The machine-readable output is a task artifact, `ml-fidelity-<feature>.json`,
written to `MOZ_UPLOAD_DIR` before any assertion runs; a failing run still
leaves its numbers behind. Its top level holds:

- `schema_version` and `feature`;
- `trace`: the models the trace referenced, its `reference` block, and the
  number of examples;
- `host`: `os`, `arch`, a `cpu` block with name, vendor, family, model,
  stepping and core counts, and `cpu_features`;
- `backend`: the backend the engines resolved to;
- `engines`: per role, `requested`, `resolved`, `effective`, `dropped` and
  `transformed`;
- `metrics`: keyed like the trace's tolerances, each a `value` with the
  `bound` it was checked against, or a bare number for a measurement that is
  only reported.

## Known gaps

- The native ONNX backend returns a fresh copy from the `Tensor.data` getter
  on every read, and transformers.js builds all of its tensors through the
  backend's `Tensor` class. Every in-place write transformers.js makes is
  lost on `onnx-native`. This affects `normalize: true`, which the embedding
  request carries and the wasm backend honours, and it affects the logits
  processors declared in a model's `generation_config.json`: the repetition
  penalty, no-repeat n-gram and bad-words list of the pdf.js captioner, and
  the bad-words lists of both topic models. The embeddings test reports the
  magnitude check as a `todo` on `onnx-native`; the pdf.js test shows the
  problem as captions that diverge from the reference at the first repeated
  word. The topic traces were recorded from `onnx-native` and cannot see it.
- The unified autofill classifier publishes no fp32 export from v0.2.5 on.
  Its trace uses the shipped q8 weights run under Python onnxruntime, which
  measures the runtime and the hardware; the quantized weights are on both
  sides of that comparison. The two-head autofill encoder was re-quantized as
  v0.3.1 without a new fp32 export; its trace keeps the v0.3.0 fp32 weights
  as the reference, recorded in `reference.revision`.
- The Smart Window topic model publishes no fp32 export, and the smart tab
  topic model cannot be reproduced outside the browser because transformers.js
  tokenises T5 input differently from the Python `tokenizers` library. Both
  traces use Firefox's own output as the reference. The Smart Window intent
  trace does too, without such a blocker, and should be regenerated
  externally.
- pdf.js answers alt text guesses with a fixed string under automation, to
  keep its own tests offline. The fidelity test clears
  `PdfJsParent.stubAIEngine` and drives `_mlGuess` on the actor of a loaded
  PDF, with the request the image editor would have built.
- transformers.js resizes images through canvas `drawImage` and ignores the
  image processor's `resample` setting. The pdf.js reference was resized with
  a bilinear filter, and the trace's images come in several sizes so that the
  encoder comparison includes the resize.
- Only the ONNX pipeline implements `getResolvedOptions` and
  `takeModelOutput`. A feature that moves to another backend loses the
  configuration report and the output comparison, and its test fails with a
  message saying so.
