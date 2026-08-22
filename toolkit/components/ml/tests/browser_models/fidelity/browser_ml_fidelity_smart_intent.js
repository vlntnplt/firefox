/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the Smart Window intent classifier against its reference trace. Two
// models ship for this feature and the home region picks between them. Each
// example carries the region it was recorded under and an output for the
// model that served it; the test sets that region.
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

add_task(async function test_intent_fidelity() {
  recordEngineRuns();

  const trace = await loadTrace("smart_intent_trace.json", {
    tolerances: [
      "kld_max.intent",
      "kld_max.intent_en_fr",
      "result_accuracy_min",
    ],
  });
  // Restore the region afterwards; other components read it. A pref with an
  // empty user value differs from a pref with no user value.
  const hadRegion = Services.prefs.prefHasUserValue(REGION_PREF);
  const originalRegion = Region.home;
  registerCleanupFunction(() =>
    hadRegion
      ? Region._setHomeRegion(originalRegion, false)
      : Services.prefs.clearUserPref(REGION_PREF)
  );

  try {
    const measurements = { intent: [], intent_en_fr: [] };
    let intentMatches = 0;

    for (const example of trace.examples) {
      const { query, region } = example.input;
      const [served, ...others] = Object.keys(example.outputs);
      Assert.deepEqual(
        others,
        [],
        `${example.id}: the trace records one model's output.`
      );

      const marks = Object.fromEntries(
        Object.entries(ENGINE_IDS).map(([role, id]) => [role, runCount(id)])
      );
      Region._setHomeRegion(region, false);
      const intent = await IntentClassifier.getPromptIntent(query);

      // The region picks the model; the run must come from the model the trace
      // recorded.
      const ran = Object.keys(ENGINE_IDS).filter(
        role => runCount(ENGINE_IDS[role]) > marks[role]
      );
      Assert.deepEqual(
        ran,
        [served],
        `${example.id}: region ${region} routed the query to the ${served} model.`
      );

      const [run] = runsSince(
        ENGINE_IDS[served],
        marks[served],
        `${example.id}: the query took one run.`,
        1
      );
      const output = modelOutputOf(
        run,
        `${example.id}: the model reported its output.`
      );
      measurements[served].push({
        id: example.id,
        ...compareDistributions(
          output,
          example.outputs[served],
          `${example.id}: the model produced the reference shape.`
        ),
      });

      if (intent === example.result) {
        intentMatches++;
      } else {
        info(
          `${example.id}: intent ${JSON.stringify(intent)}, ` +
            `trace says ${JSON.stringify(example.result)}`
        );
      }
    }

    const engines = {};
    const configs = {};
    for (const [role, engineId] of Object.entries(ENGINE_IDS)) {
      const engine = MLEngine.getInstance(engineId);
      // The requested options are read off the engine the feature created.
      // Its `backend` is already the resolved one.
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

    const metrics = {
      result_accuracy_min: bounded(
        trace,
        "result_accuracy_min",
        intentMatches / trace.examples.length
      ),
      results_matched: intentMatches,
    };
    for (const [role, runs] of Object.entries(measurements)) {
      const worst = runs.reduce((a, b) => (b.kldMax > a.kldMax ? b : a));
      metrics[`kld_max.${role}`] = bounded(
        trace,
        `kld_max.${role}`,
        worst.kldMax
      );
      metrics[`kld_mean.${role}`] =
        runs.reduce((sum, run) => sum + run.kldMean, 0) / runs.length;
      info(`Worst ${role} example: ${worst.id}`);
    }

    await reportFidelity({
      feature: "smart-intent",
      trace,
      backend,
      engines: configReport,
      metrics,
    });
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
