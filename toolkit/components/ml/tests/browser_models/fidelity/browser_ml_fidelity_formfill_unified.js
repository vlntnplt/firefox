/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the unified form autofill classifier, the single text-classification
// model detectFields runs when `extensions.formautofill.useml.twoHead` is
// off, against its reference trace. browser_ml_fidelity_formfill.js covers
// the two-head deployment; the pref is set at runtime and both deployments
// are tested.
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
  // Without useml.successful, the first detectFields call only downloads the
  // models.
  await SpecialPowers.pushPrefEnv({
    set: [
      ["extensions.formautofill.useml.successful", true],
      ["extensions.formautofill.useml.twoHead", false],
    ],
  });
  recordEngineRuns();

  const trace = await loadTrace("formfill_unified_trace.json", {
    tolerances: ["kld_max.classifier", "result_accuracy_min"],
  });
  const classifierId = FEATURES["formfill-classification"].engineId;

  try {
    const fieldDetails = trace.examples.map(example => ({
      mlData: toMlData(example.input.text),
    }));
    const mark = runCount(classifierId);
    await new FormAutofillML().detectFields(fieldDetails);

    const { classifier } = assertDeploymentRan({ classifier: classifierId }, [
      FEATURES["formfill-encoder"].engineId,
      FEATURES["formfill-head"].engineId,
    ]);

    await assertTraceMatchesShippedModels(trace, {
      classifier: classifier.resolvedOptions.options,
    });

    const backend = resolvedBackend(classifier);
    info(`Resolved backend on this machine: ${backend}`);
    const engines = reportEngineConfigs({ classifier: { engine: classifier } });

    const [run] = runsSince(
      classifierId,
      mark,
      "detectFields classified every field in a single run.",
      1
    );
    const output = modelOutputOf(run, "The classifier reported its output.");

    const classes = trace.examples[0].outputs.classifier.dims.at(-1);
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
        example.outputs.classifier,
        `${example.id}: the classifier produced the reference shape.`
      );
      kldSum += agreement.kldMean;
      if (agreement.kldMax > kldMax) {
        kldMax = agreement.kldMax;
        worstId = example.id;
      }

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
    info(`Worst example: ${worstId}`);

    await reportFidelity({
      feature: "formfill-unified",
      trace,
      backend,
      engines,
      metrics: {
        "kld_max.classifier": bounded(trace, "kld_max.classifier", kldMax),
        "kld_mean.classifier": kldSum / trace.examples.length,
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
