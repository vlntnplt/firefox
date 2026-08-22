/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the unified form autofill classifier -- the single
// text-classification model detectFields runs when
// `extensions.formautofill.useml.twoHead` is off -- against its reference
// trace. The two-head deployment is covered by
// browser_ml_fidelity_formfill.js; which of the two a profile gets is a
// runtime decision, so both are pinned and checked rather than whichever one
// this build happens to default to.
// See toolkit/components/ml/docs/fidelity.md.

const { EngineProcess, FEATURES } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);

const { FormAutofillML } = ChromeUtils.importESModule(
  "resource://gre/modules/shared/FormAutofillML.sys.mjs"
);

// Cold workers download ~14 MB of ONNX first.
requestLongerTimeout(4);

add_task(async function test_formfill_unified_fidelity() {
  // detectFields only starts downloads on first use, it does not infer.
  await SpecialPowers.pushPrefEnv({
    set: [
      ["extensions.formautofill.useml.successful", true],
      ["extensions.formautofill.useml.twoHead", false],
    ],
  });
  recordEngineRuns();

  const trace = await loadTrace("formfill_unified_trace.json");

  try {
    const fieldDetails = trace.examples.map(example => ({
      mlData: toMlData(example.text),
    }));
    await new FormAutofillML().detectFields(fieldDetails);

    const { classifier } = assertDeploymentRan(
      { classifier: FEATURES["formfill-classification"].engineId },
      [
        FEATURES["formfill-encoder"].engineId,
        FEATURES["formfill-head"].engineId,
      ]
    );

    await assertTraceMatchesShippedModels(trace, {
      classifier: classifier.resolvedOptions.options,
    });

    const backend = resolvedBackend(classifier);
    info(`Resolved backend on this machine: ${backend}`);
    const engines = reportEngineConfigs({ classifier: { engine: classifier } });

    Assert.equal(
      classifier.recordedRuns.length,
      1,
      "detectFields classified every field in a single run."
    );
    const [run] = classifier.recordedRuns;
    const output = run.modelOutput?.[0]?.outputs?.logits;
    Assert.ok(output, "The classifier reported its output.");

    const classes = trace.examples[0].classifier.dims.at(-1);
    Assert.deepEqual(
      output.dims,
      [trace.examples.length, classes],
      "The classifier scored every field in one batch."
    );

    let kldMax = 0;
    let kldSum = 0;
    let worstId = null;
    let correct = 0;
    for (let i = 0; i < trace.examples.length; i++) {
      const example = trace.examples[i];
      const row = {
        dims: [1, classes],
        data: output.data.slice(i * classes, (i + 1) * classes),
      };
      const agreement = compareDistributions(
        row,
        example.classifier,
        `${example.id}: the classifier produced the reference shape.`
      );
      kldSum += agreement.kldMean;
      if (agreement.kldMax > kldMax) {
        kldMax = agreement.kldMax;
        worstId = example.id;
      }

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
    const kldMean = kldSum / trace.examples.length;

    await writeFidelityArtifact("formfill-unified", {
      trace: {
        models: trace.models,
        reference: trace.reference,
        example_count: trace.examples.length,
      },
      backend,
      engines,
      metrics: {
        classifier_kld_max: kldMax,
        classifier_kld_mean: kldMean,
        label_accuracy: accuracy,
      },
      tolerances: trace.tolerances,
    });

    info(
      `FIDELITY ${backend}: classifier-kld-max=${kldMax.toFixed(6)} ` +
        `classifier-kld-mean=${kldMean.toFixed(6)} nats ` +
        `(worst example ${worstId}) ` +
        `label-accuracy=${accuracy.toFixed(4)} ` +
        `(${correct}/${trace.examples.length})`
    );

    Assert.lessOrEqual(
      kldMax,
      trace.tolerances.kld_max,
      `${backend}: predicted distribution is within the trace tolerance.`
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
