/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the urlbar ML suggest models against their reference trace.
// See toolkit/components/ml/docs/fidelity.md.

const { EngineProcess, FEATURES } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);

const { MLSuggest } = ChromeUtils.importESModule(
  "moz-src:///browser/components/urlbar/private/MLSuggest.sys.mjs"
);

// Cold workers download ~90 MB of ONNX first.
requestLongerTimeout(10);

const ROLES = {
  intent: {
    engineId: FEATURES["suggest-intent-classification"].engineId,
    config: MLSuggest.INTENT_OPTIONS,
  },
  ner: {
    engineId: FEATURES["suggest-NER"].engineId,
    config: MLSuggest.NER_OPTIONS,
  },
};

add_task(async function test_mlsuggest_fidelity() {
  recordEngineRuns();

  const trace = await loadTrace("mlsuggest_trace.json", {
    tolerances: ["kld_max.intent", "kld_max.ner"],
  });
  await assertTraceMatchesShippedModels(trace, {
    intent: ROLES.intent.config,
    ner: ROLES.ner.config,
  });

  try {
    await MLSuggest.initialize();

    const marks = {};
    for (const [role, { engineId }] of Object.entries(ROLES)) {
      marks[role] = runCount(engineId);
    }
    for (const example of trace.examples) {
      await MLSuggest.makeSuggestions(example.input.query);
    }

    const engines = {};
    for (const [role, { engineId }] of Object.entries(ROLES)) {
      engines[role] = MLEngine.getInstance(engineId);
    }

    const backend = resolvedBackend(engines.intent);
    info(`Resolved backend on this machine: ${backend}`);

    const configReport = reportEngineConfigs({
      intent: { engine: engines.intent, requested: ROLES.intent.config },
      ner: { engine: engines.ner, requested: ROLES.ner.config },
    });

    const metrics = {};
    for (const [role, { engineId }] of Object.entries(ROLES)) {
      const runs = runsSince(
        engineId,
        marks[role],
        `${role} ran once per example.`,
        trace.examples.length
      );

      let kldMax = 0;
      let kldSum = 0;
      let worstId = null;
      for (let i = 0; i < trace.examples.length; i++) {
        const example = trace.examples[i];
        const output = modelOutputOf(
          runs[i],
          `${role} reported model output for ${example.id}.`
        );
        const agreement = compareDistributions(
          output,
          example.outputs[role],
          `${role} produced the reference shape for ${example.id}.`
        );
        kldSum += agreement.kldMean;
        if (agreement.kldMax > kldMax) {
          kldMax = agreement.kldMax;
          worstId = example.id;
        }
      }

      metrics[`kld_max.${role}`] = bounded(trace, `kld_max.${role}`, kldMax);
      metrics[`kld_mean.${role}`] = kldSum / trace.examples.length;
      info(`Worst ${role} example: ${worstId}`);
    }

    await reportFidelity({
      feature: "mlsuggest",
      trace,
      backend,
      engines: configReport,
      metrics,
    });
  } finally {
    await MLSuggest.shutdown();
    await EngineProcess.destroyMLEngine();
  }
});
