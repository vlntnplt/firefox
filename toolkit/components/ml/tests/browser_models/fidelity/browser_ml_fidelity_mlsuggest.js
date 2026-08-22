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

  const trace = await loadTrace("mlsuggest_trace.json");
  await assertTraceMatchesShippedModels(trace, {
    intent: ROLES.intent.config,
    ner: ROLES.ner.config,
  });

  try {
    await MLSuggest.initialize();
    for (const example of trace.examples) {
      await MLSuggest.makeSuggestions(example.query);
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
    for (const role of Object.keys(ROLES)) {
      const runs = engines[role].recordedRuns;
      Assert.equal(
        runs.length,
        trace.examples.length,
        `${role} ran once per example.`
      );

      let kldMax = 0;
      let kldSum = 0;
      let worstId = null;
      for (let i = 0; i < trace.examples.length; i++) {
        const example = trace.examples[i];
        const output = runs[i].modelOutput?.[0]?.outputs?.logits;
        Assert.ok(output, `${role} reported model output for ${example.id}.`);
        const agreement = compareDistributions(
          output,
          example[role],
          `${role} produced the reference shape for ${example.id}.`
        );
        kldSum += agreement.kldMean;
        if (agreement.kldMax > kldMax) {
          kldMax = agreement.kldMax;
          worstId = example.id;
        }
      }

      metrics[`${role}_kld_max`] = kldMax;
      metrics[`${role}_kld_mean`] = kldSum / trace.examples.length;
      info(
        `FIDELITY ${backend} ${role}: kld-max=${kldMax.toFixed(6)} ` +
          `kld-mean=${metrics[`${role}_kld_mean`].toFixed(6)} nats ` +
          `(worst example ${worstId})`
      );
    }

    await writeFidelityArtifact("mlsuggest", {
      trace: { models: trace.models, example_count: trace.examples.length },
      backend,
      engines: configReport,
      metrics,
      tolerances: trace.tolerances,
    });

    for (const role of Object.keys(ROLES)) {
      Assert.lessOrEqual(
        metrics[`${role}_kld_max`],
        trace.tolerances.kld_max[role],
        `${backend}: ${role} predicted distribution is within tolerance.`
      );
    }
  } finally {
    await MLSuggest.shutdown();
    await EngineProcess.destroyMLEngine();
  }
});
