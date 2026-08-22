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
  // Without useml.successful, the first detectFields call only downloads the
  // models.
  await SpecialPowers.pushPrefEnv({
    set: [
      ["extensions.formautofill.useml.successful", true],
      ["extensions.formautofill.useml.twoHead", true],
    ],
  });
  recordEngineRuns();

  const trace = await loadTrace("formfill_trace.json", {
    tolerances: [
      "embedding_cosine_min.encoder",
      "embedding_norm_rel_max.encoder",
      "result_accuracy_min",
    ],
  });

  const CONTEXT = ["current", "previous", "next"];
  for (const example of trace.examples) {
    Assert.deepEqual(
      splitContext(toMlData(example.input.text)),
      CONTEXT.map(slot => example.input.text[slot]),
      `${example.id}: the mlData adapter round-trips through splitContext.`
    );
  }

  // detectFields embeds each distinct text once; live vectors are matched to
  // examples by text.
  const texts = [
    ...new Set(
      trace.examples.flatMap(example =>
        CONTEXT.map(slot => example.input.text[slot])
      )
    ),
  ];
  const encoderId = FEATURES["formfill-encoder"].engineId;

  try {
    const fieldDetails = trace.examples.map(example => ({
      mlData: toMlData(example.input.text),
    }));
    const mark = runCount(encoderId);
    await new FormAutofillML().detectFields(fieldDetails);

    const { encoder, classifier } = assertDeploymentRan(
      {
        encoder: encoderId,
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

    const [encoderRun] = runsSince(
      encoderId,
      mark,
      "detectFields embedded every section in a single encoder run.",
      1
    );
    const encoded = encoderRun.request.args[0];
    let embeddings = encoderRun.response;

    assertRequestOptions(
      encoderRun,
      trace,
      "encoder",
      "The encoder was run with the pooling the trace records."
    );

    // feature-extraction triple-nests a singleton batch.
    if (embeddings.length === 1 && Array.isArray(embeddings[0]?.[0])) {
      embeddings = embeddings[0];
    }

    Assert.deepEqual(
      [...encoded].sort(),
      [...texts].sort(),
      "detectFields encoded exactly the texts the trace has vectors for."
    );
    const live = new Map(encoded.map((text, i) => [text, embeddings[i]]));

    const { minCosine, normDelta, worstId } = compareEmbeddings(
      trace.examples.flatMap(example =>
        CONTEXT.map(slot => ({
          id: `${example.id}.${slot}`,
          live: live.get(example.input.text[slot]),
          reference: example.outputs.encoder[slot],
        }))
      )
    );
    info(`Worst text: ${worstId}`);

    let correct = 0;
    for (let i = 0; i < trace.examples.length; i++) {
      const example = trace.examples[i];
      const expected = example.result === "other" ? undefined : example.result;
      if (fieldDetails[i].fieldName === expected) {
        correct++;
      } else {
        info(
          `${backend} label changed for ${example.id}: ` +
            `got ${fieldDetails[i].fieldName}, trace says ${example.result}`
        );
      }
    }

    await reportFidelity({
      feature: "formfill",
      trace,
      backend,
      engines,
      metrics: {
        "embedding_cosine_min.encoder": bounded(
          trace,
          "embedding_cosine_min.encoder",
          minCosine
        ),
        "embedding_norm_rel_max.encoder": bounded(
          trace,
          "embedding_norm_rel_max.encoder",
          normDelta
        ),
        result_accuracy_min: bounded(
          trace,
          "result_accuracy_min",
          correct / trace.examples.length
        ),
        results_matched: correct,
      },
    });
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
