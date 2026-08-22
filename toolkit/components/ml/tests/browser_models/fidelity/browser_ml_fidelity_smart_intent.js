/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the Smart Window intent classifier against its reference trace. Two
// models ship for this feature and the home region picks between them, so the
// trace records the region each query was recorded under and the test sets
// that region rather than naming a model.
// See toolkit/components/ml/docs/fidelity.md.

const { EngineProcess, FEATURES } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);

const { IntentClassifier } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/models/IntentClassifier.sys.mjs"
);

const { Region } = ChromeUtils.importESModule(
  "resource://gre/modules/Region.sys.mjs"
);

// Cold workers download ~110 MB of ONNX first.
requestLongerTimeout(10);

const REGION_PREF = "browser.search.region";

const ENGINE_IDS = {
  intent: FEATURES["smart-intent"].engineId,
  intent_en_fr: FEATURES["smart-intent-en-fr"].engineId,
};

/**
 * @param {string} role
 * @returns {number} How many runs that role's engine has recorded so far.
 */
function runCount(role) {
  return MLEngine.getInstance(ENGINE_IDS[role])?.recordedRuns.length ?? 0;
}

add_task(async function test_intent_fidelity() {
  recordEngineRuns();

  const trace = await loadTrace("smart_intent_trace.json");
  // Other components read the region, so put it back the way it was. Writing
  // an empty string is not the same as never having had a user value.
  const hadRegion = Services.prefs.prefHasUserValue(REGION_PREF);
  const originalRegion = Region.home;
  registerCleanupFunction(() =>
    hadRegion
      ? Region._setHomeRegion(originalRegion, false)
      : Services.prefs.clearUserPref(REGION_PREF)
  );

  try {
    const served = { intent: 0, intent_en_fr: 0 };
    const measurements = { intent: [], intent_en_fr: [] };
    let intentMatches = 0;

    for (const example of trace.examples) {
      Region._setHomeRegion(example.region, false);
      const intent = await IntentClassifier.getPromptIntent(example.query);

      // The region picks the model, so a query recorded against one model must
      // not come back served by the other.
      const ran = Object.keys(ENGINE_IDS).filter(
        role => runCount(role) > served[role]
      );
      Assert.deepEqual(
        ran,
        [example.model],
        `${example.id}: region ${example.region} routed the query to the ` +
          `${example.model} model.`
      );
      served[example.model] = runCount(example.model);

      const engine = MLEngine.getInstance(ENGINE_IDS[example.model]);
      const run = engine.recordedRuns.at(-1);
      const output = run.modelOutput?.[0]?.outputs?.logits;
      Assert.ok(output, `${example.id}: the model reported its output.`);
      measurements[example.model].push({
        id: example.id,
        ...compareDistributions(
          output,
          example.output,
          `${example.id}: the model produced the reference shape.`
        ),
      });

      if (intent === example.intent) {
        intentMatches++;
      } else {
        info(
          `${example.id}: intent ${JSON.stringify(intent)}, ` +
            `trace says ${JSON.stringify(example.intent)}`
        );
      }
    }

    const engines = {};
    const configs = {};
    for (const [role, engineId] of Object.entries(ENGINE_IDS)) {
      const engine = MLEngine.getInstance(engineId);
      // The feature keeps its options to itself, so what it asked for is read
      // off the engine it created. `backend` there is already the one the
      // request resolved to, so it never shows up as transformed.
      engines[role] = {
        engine,
        requested: engine.pipelineOptions.getOptions(),
      };
      configs[role] = engine.resolvedOptions.options;
    }
    await assertTraceMatchesShippedModels(trace, configs);

    const backend = resolvedBackend(engines.intent.engine);
    info(`Resolved backend on this machine: ${backend}`);
    const configReport = reportEngineConfigs(engines);

    const metrics = { intent_matches: intentMatches };
    for (const [role, runs] of Object.entries(measurements)) {
      const worst = runs.reduce((a, b) => (b.kldMax > a.kldMax ? b : a));
      metrics[`${role}_kld_max`] = worst.kldMax;
      metrics[`${role}_kld_mean`] =
        runs.reduce((sum, run) => sum + run.kldMean, 0) / runs.length;
      info(
        `FIDELITY ${backend} ${role}: kld-max=${worst.kldMax.toFixed(6)} ` +
          `kld-mean=${metrics[`${role}_kld_mean`].toFixed(6)} nats ` +
          `(worst example ${worst.id})`
      );
    }
    info(
      `FIDELITY ${backend}: intent-matches=` +
        `${intentMatches}/${trace.examples.length}`
    );

    await writeFidelityArtifact("smart-intent", {
      trace: { models: trace.models, example_count: trace.examples.length },
      backend,
      engines: configReport,
      metrics,
      tolerances: trace.tolerances,
    });

    for (const role of Object.keys(ENGINE_IDS)) {
      Assert.lessOrEqual(
        metrics[`${role}_kld_max`],
        trace.tolerances.kld_max[role],
        `${backend}: ${role} predicted distribution is within tolerance.`
      );
    }
    Assert.greaterOrEqual(
      intentMatches / trace.examples.length,
      trace.tolerances.intent_accuracy_min,
      `${backend}: the intent every query resolves to matches the trace.`
    );
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
