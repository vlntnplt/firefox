/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the Smart Window topic model against its reference trace. It runs the
// same code as the tab strip's topic model but resolves through its own Remote
// Settings slot, so a bump on one side leaves the other where it was and
// browser_ml_fidelity_smarttab_topic says nothing about this pairing.
// See toolkit/components/ml/docs/fidelity.md.

const { EngineProcess, FEATURES } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);

const { AutoTabGroupingSuggestions } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/ui/modules/AutoTabGroupingSuggestions.sys.mjs"
);

// Cold workers download ~55 MB of ONNX first.
requestLongerTimeout(10);

add_task(async function test_smartwindow_topic_fidelity() {
  recordEngineRuns();

  const trace = await loadTrace("smartwindow_topic_trace.json");
  // The manager Smart Window builds carries the config it points the topic
  // model at, so the test names no feature of its own.
  const manager = AutoTabGroupingSuggestions.manager;
  const config = manager.config.topicGeneration;
  await assertTraceMatchesShippedModels(trace, { topic: config });

  try {
    let encoderCosineMin = 1;
    let kldMax = 0;
    let worstId = null;
    let labelMatches = 0;
    let engine = null;

    for (let i = 0; i < trace.examples.length; i++) {
      const example = trace.examples[i];
      // Drive production: this builds the prompt from the tabs, runs the
      // engine and post-processes the label.
      const label = await manager.getPredictedLabelForGroup(
        example.tabs,
        example.other_tabs
      );

      engine = MLEngine.getInstance(FEATURES[config.featureId].engineId);
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

    await writeFidelityArtifact("smartwindow-topic", {
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
