/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the form autofill classifier against its reference trace.
// See toolkit/components/ml/docs/fidelity.md.

const { EngineProcess, FEATURES } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);

const { FormAutofillML, splitContext } = ChromeUtils.importESModule(
  "resource://gre/modules/shared/FormAutofillML.sys.mjs"
);

// Cold workers download ~15 MB of ONNX first.
requestLongerTimeout(4);

add_task(async function test_formfill_fidelity() {
  // detectFields only starts downloads on first use, it does not infer.
  await SpecialPowers.pushPrefEnv({
    set: [
      ["extensions.formautofill.useml.successful", true],
      ["extensions.formautofill.useml.twoHead", true],
    ],
  });
  recordEngineRuns();

  const trace = await loadTrace("formfill_trace.json");

  for (const example of trace.examples) {
    Assert.deepEqual(
      splitContext(toMlData(example.text)),
      [example.text.current, example.text.previous, example.text.next],
      `${example.id}: the mlData adapter round-trips through splitContext.`
    );
  }

  const texts = trace.embeddings.map(entry => entry.text);
  const referenceVectors = new Map(
    trace.embeddings.map(entry => [entry.text, entry.vector])
  );

  try {
    const fieldDetails = trace.examples.map(example => ({
      mlData: toMlData(example.text),
    }));
    await new FormAutofillML().detectFields(fieldDetails);

    const { encoder, classifier } = assertDeploymentRan(
      {
        encoder: FEATURES["formfill-encoder"].engineId,
        classifier: FEATURES["formfill-head"].engineId,
      },
      [FEATURES["formfill-classification"].engineId]
    );

    await assertTraceMatchesShippedModels(trace, {
      encoder: encoder.resolvedOptions.options,
      classifier: classifier.resolvedOptions.options,
    });

    const backend = resolvedBackend(encoder);
    info(`Resolved backend on this machine: ${backend}`);

    const engines = reportEngineConfigs({
      encoder: { engine: encoder },
      classifier: { engine: classifier },
    });

    Assert.equal(
      encoder.recordedRuns.length,
      1,
      "detectFields embedded every section in a single encoder run."
    );
    const [encoderRun] = encoder.recordedRuns;
    const encoded = encoderRun.request.args[0];
    let embeddings = encoderRun.response;

    const { pooling, normalize } = trace.preprocessing;
    Assert.deepEqual(
      {
        pooling: encoderRun.request.options?.pooling,
        normalize: encoderRun.request.options?.normalize,
      },
      { pooling, normalize },
      "The encoder was run with the pooling the trace records."
    );

    // feature-extraction triple-nests a singleton batch.
    if (
      Array.isArray(embeddings) &&
      embeddings.length === 1 &&
      Array.isArray(embeddings[0]) &&
      embeddings[0].length !== trace.models.encoder.embedding_dim
    ) {
      embeddings = embeddings[0];
    }

    Assert.deepEqual(
      [...encoded].sort(),
      [...texts].sort(),
      "detectFields encoded exactly the texts the trace has vectors for."
    );

    const { minCosine, normDelta } = compareEmbeddings(
      encoded,
      embeddings,
      referenceVectors
    );

    let correct = 0;
    for (let i = 0; i < trace.examples.length; i++) {
      const example = trace.examples[i];
      const expected =
        example.expected_label === "other" ? undefined : example.expected_label;
      if (fieldDetails[i].fieldName === expected) {
        correct++;
      } else {
        info(
          `${backend} label changed for ${example.id}: ` +
            `got ${fieldDetails[i].fieldName}, trace says ${example.expected_label}`
        );
      }
    }
    const accuracy = correct / trace.examples.length;

    await writeFidelityArtifact("formfill", {
      trace: {
        models: trace.models,
        example_count: trace.examples.length,
        text_count: texts.length,
      },
      backend,
      engines,
      metrics: {
        embedding_cosine_min: minCosine,
        embedding_norm_rel_delta: normDelta,
        label_accuracy: accuracy,
      },
      tolerances: trace.tolerances,
    });

    info(
      `FIDELITY ${backend}: embedding-cosine-min=${minCosine.toFixed(5)} ` +
        `embedding-norm-rel-delta=${normDelta.toFixed(5)} ` +
        `label-accuracy=${accuracy.toFixed(4)} ` +
        `(${correct}/${trace.examples.length})`
    );

    Assert.greaterOrEqual(
      minCosine,
      trace.tolerances.embedding_cosine_min,
      `${backend}: embedding cosine meets the trace tolerance.`
    );
    Assert.lessOrEqual(
      normDelta,
      trace.tolerances.embedding_norm_rel_max,
      `${backend}: embedding magnitudes match the trace.`
    );
    Assert.greaterOrEqual(
      accuracy,
      trace.tolerances.label_accuracy_min,
      `${backend}: label accuracy meets the trace tolerance.`
    );
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
