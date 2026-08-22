/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the smart tab grouping topic model against its reference trace. The
// embeddings half of the feature is covered by browser_ml_fidelity_embeddings.
// See toolkit/components/ml/docs/fidelity.md.

const { EngineProcess, FEATURES } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);

const { SmartTabGroupingManager, SMART_TAB_GROUPING_CONFIG } =
  ChromeUtils.importESModule(
    "moz-src:///browser/components/tabbrowser/SmartTabGrouping.sys.mjs"
  );

// Cold workers download ~55 MB of ONNX first.
requestLongerTimeout(10);

add_task(async function test_smarttab_topic_fidelity() {
  recordEngineRuns();

  const trace = await loadTrace("smarttab_topic_trace.json");
  const config = SMART_TAB_GROUPING_CONFIG.topicGeneration;
  await assertTraceMatchesShippedModels(trace, { topic: config });

  const manager = new SmartTabGroupingManager();
  try {
    let encoderCosineMin = 1;
    let kldMax = 0;
    let worstId = null;
    let labelMatches = 0;
    let engine = null;

    for (let i = 0; i < trace.examples.length; i++) {
      const example = trace.examples[i];
      // Drive production: this builds the prompt from the tabs, applies the
      // model revision override, runs the engine and post-processes the label.
      const label = await manager.getPredictedLabelForGroup(
        example.tabs,
        example.other_tabs
      );

      engine = MLEngine.getInstance(FEATURES["smart-tab-topic"].engineId);
      const { encoderCosine, kld } = compareTopicModelRun(
        engine.recordedRuns[i].modelOutput,
        example
      );
      encoderCosineMin = Math.min(encoderCosineMin, encoderCosine);
      if (kld > kldMax) {
        kldMax = kld;
        worstId = example.id;
      }

      if (label === example.predicted_label) {
        labelMatches++;
      } else {
        info(
          `${example.id}: label ${JSON.stringify(label)}, ` +
            `trace says ${JSON.stringify(example.predicted_label)}`
        );
      }
    }

    const backend = resolvedBackend(engine);
    info(`Resolved backend on this machine: ${backend}`);
    const engines = reportEngineConfigs({
      topic: { engine, requested: config },
    });

    await writeFidelityArtifact("smarttab-topic", {
      trace: { models: trace.models, example_count: trace.examples.length },
      backend,
      engines,
      metrics: {
        encoder_cosine_min: encoderCosineMin,
        step0_kld_max: kldMax,
        predicted_label_matches: labelMatches,
      },
      tolerances: trace.tolerances,
    });

    info(
      `FIDELITY ${backend}: encoder-cosine-min=${encoderCosineMin.toFixed(6)} ` +
        `step0-kld-max=${kldMax.toFixed(6)} nats (worst ${worstId}) ` +
        `label-matches=${labelMatches}/${trace.examples.length}`
    );

    Assert.greaterOrEqual(
      encoderCosineMin,
      trace.tolerances.encoder_cosine_min,
      `${backend}: encoder output agrees with the trace.`
    );
    Assert.lessOrEqual(
      kldMax,
      trace.tolerances.step0_kld_max,
      `${backend}: the decoder's predicted distribution is within tolerance.`
    );
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
