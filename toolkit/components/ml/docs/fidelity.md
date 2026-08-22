# Fidelity traces

A fidelity trace pins a small sample of a model's evaluation, so CI can check
that the model we ship, in the configuration we ship it in, still behaves like
the one that was evaluated. It does not measure whether the model is good; that
is what the evaluation itself is for.

Firefox defines the format. Producing a trace is up to whoever trains and
exports the model.

## Format

One JSON document, `snake_case` keys, vectors as plain arrays.

```json
{
  "schema_version": 1,

  "models": {
    "encoder": {
      "model_id": "mozilla/form-autofill-embed",
      "revision": "v0.3.0",
      "embedding_dim": 384
    },
    "classifier": {
      "model_id": "mozilla/form-autofill-head",
      "revision": "v0.3.0"
    }
  },

  "preprocessing": { "pooling": "mean", "normalize": false },

  "reference": {
    "runtime": "onnxruntime",
    "runtime_version": "1.29.0",
    "device": "cpu",
    "dtype": "float32"
  },

  "tolerances": {
    "embedding_cosine_min": 0.995,
    "embedding_norm_rel_max": 0.01,
    "label_accuracy_min": 1.0
  },

  "embeddings": [
    { "text": "firstname first name enter your first name",
      "vector": [-0.2663476, -0.2527834, "..."] }
  ],

  "examples": [
    { "id": "address.firstname",
      "text": { "current": "firstname first name enter your first name",
                "previous": "",
                "next": "lastname last name surname family name" },
      "expected_label": "given-name" }
  ]
}
```

`models` records what the reference outputs were produced from. CI checks the
model ids and revisions against what Remote Settings ships, so a model bump
without a new trace fails rather than measuring a pairing that no longer
exists. `embedding_dim` is used to un-nest the encoder output.

`preprocessing` holds the pipeline settings that are part of the model
contract. For the autofill encoder that is mean pooling without L2
normalization, because the classifier was trained on un-normalized vectors.
Only settings the test can check belong here.

`reference` records what produced the reference outputs. The reference is
itself a configuration, not ground truth, so it is recorded as fields rather
than a formatted string. A producer running the pre-conversion model would put
its own framework and version here.

`tolerances` are how far the shipped configuration may drift. Whoever evaluated
the model sets these, since they know what the downstream task absorbs.

`embeddings` are deduplicated by text: one text is `current` for one example
and `previous` or `next` for its neighbours.

`examples` give inputs as the text the model consumes. However a feature
carries that text around inside the browser is a transport detail and stays out
of the trace.

## Choosing examples

An evaluation set wants hard, ambiguous inputs. A fidelity trace wants the
opposite: an example whose top two classes are close will flip its label on a
numeric difference too small to indicate a real problem, and the test becomes
intermittent.

So a producer should reject examples whose top two logits are close. The
autofill trace was filtered at a margin of 2.0; the smallest margin that
survived is 3.9, against a worst-case logit movement of 0.31 from quantizing
float32 to int8. That filtering happens in the producer and does not appear in
the trace; it is what makes label accuracy usable as an oracle.

Use `""` for `previous`/`next` at a form boundary. The encoder turns the empty
string into a fixed non-zero embedding, which is what the classifier was
trained against.

## What CI does with it

Fidelity tests live in `toolkit/components/ml/tests/browser_models/fidelity/`,
one per feature, named `browser_ml_fidelity_<feature>.js` with its trace
alongside as `<feature>_trace.json`. Anything reusable across features is in
`head.js`. They run in the `mochitest-browser-chrome-ml-models` suite, which
downloads real models from the model hub.

It runs the shipped configuration on whatever the machine resolves to. Coverage
of architecture, instruction set and thread count comes from the set of CI
machines the suite runs on, not from forcing configurations in the test. Each
run:

1. records the resolved options of every engine. Each backend reports only the
   options it reads, so options belonging to another backend show up as
   `dropped` rather than as served. Requested options that were served
   differently are listed under `transformed`, which is normally resolution
   working, e.g. `best-onnx` becoming `onnx-native`;
2. compares the encoder's embeddings against the reference vectors, by cosine
   and by magnitude;
3. calls `FormAutofillML.detectFields` and compares the field types it assigns
   against `expected_label`.

Step 3 drives production rather than reassembling the pipeline, so the context
split, the windowed feature layout and both engine hops are covered. Flipping
the sign of the difference blocks in `detectFields` drops label accuracy from
100% to 10.5%.

`preprocessing` is checked on both sides, because asking for a setting and
getting it are different things.

The request side compares `pooling` and `normalize` against the options
production actually passed on the recorded encoder run, so a change in
`FormAutofillML` fails here.

The effect side falls out of the two vector comparisons. Wrong pooling
produces a different vector, which cosine catches. Normalization keeps the
direction and changes only the length, so cosine is blind to it -- normalizing
a vector leaves its cosine against the original at 1 -- and magnitude catches
it instead. That case is not hypothetical: the `onnx` and `onnx-native`
backends have been observed to disagree about whether they honour `normalize`
for the same request.

The trace stores the text the models consume, so the test converts back into
the `mlData` form `detectFields` expects. That conversion is checked against
the production splitter for every example.

Treeherder output:

```
Resolved backend on this machine: onnx-native
FIDELITY onnx-native: embedding-cosine-min=0.99904 embedding-norm-rel-delta=0.00185 label-accuracy=1.0000 (19/19)
```

The machine-readable output is a task artifact, `ml-fidelity-formfill.json`,
written to `MOZ_UPLOAD_DIR`. Top level:

- `schema_version`, `feature`
- `trace` — the models the trace referenced, and how many examples it held
- `host` — `os`, `arch`, and a `cpu` block (name, vendor, family, model,
  stepping, core counts) plus `cpu_features`
- `backend` — what the machine resolved to
- `engines` — per engine: `requested`, `resolved`, `effective`, `dropped`,
  `transformed`
- `metrics` and `tolerances`

Each artifact states the architecture, instruction set, core count,
quantization and backend a measurement was taken under, so coverage can be read
from collected artifacts rather than assumed.

## Comparing model output

A feature whose engines return a post-processed result has nothing stable to
compare. The transformers.js task heads reduce a model's output to a top-1
label and score, and those are not comparable across configurations: when the
argmax differs, the two scores describe different classes.

So `MLEngine.recordRuns` also records what the model returned before the head
ran, and a trace for such a feature holds reference logits per example:

```json
"tolerances": {
  "output_cosine_min": { "intent": 0.99, "ner": 0.97 },
  "output_prob_max_delta": { "intent": 0.10, "ner": 0.60 }
},
"examples": [
  { "id": "travel.nyc", "query": "flights to new york",
    "intent": { "dims": [1, 8], "logits": ["..."] },
    "ner": { "dims": [1, 7, 11], "logits": ["..."] } }
]
```

Two metrics, both scale-free so they read the same way for any model. Cosine is
the shape of the output vector. Probability delta is how far the prediction
moved, measured after softmax, which is the space the task head works in. A
difference in logit space would not be comparable across models, since logit
scale is a property of the model: quantizing float32 to int8 moves the urlbar
intent model by 0.82 and the NER model by 1.97, but in readable terms that is
cosine 0.996 against 0.984.

Reporting both separates two things that a single number confuses. NER at
cosine 0.984 with a 0.33 probability swing means the output vector barely
rotated while a near-tie changed sides.

## Known gaps

- `onnx-native` ignores `normalize: true`. The request carries it and the
  vectors come back unnormalized, while the wasm backend applies it. No
  consumer is affected today, they all compare with a scale-invariant cosine,
  so the embeddings test reports the magnitude check with `todo` rather than
  asserting it.
- The trace ships as a test support file. It belongs next to the models in the
  model repository, so that it is versioned and fetched with them.
- Coverage is not declared. The autofill trace exercises 17 of 66 labels, and
  nothing in the file says so.
- `cpu_features` reports what `nsSystemInfo.cpp` exposes, which covers SSE
  through AVX-512 and VNNI on x86 but has no AMX, and on aarch64 only NEON, no
  dotprod or i8mm. Those affect int8 kernel dispatch, and models ship
  quantized, so the CPU identity fields are currently the way to work out which
  kernel ran. Ground truth would have to come from onnxruntime reporting what
  it dispatched.
