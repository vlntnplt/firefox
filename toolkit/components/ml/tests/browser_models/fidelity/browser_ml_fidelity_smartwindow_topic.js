/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the Smart Window topic model against its reference trace. It runs the
// same code as the tab strip's topic model through its own Remote Settings
// entry; a model bump on one side leaves the other unchanged.
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

  // The manager Smart Window builds carries the topic model config.
  const manager = AutoTabGroupingSuggestions.manager;
  const config = manager.config.topicGeneration;

  try {
    await runTopicModelFidelity({
      feature: "smartwindow-topic",
      traceFile: "smartwindow_topic_trace.json",
      engineId: FEATURES[config.featureId].engineId,
      config,
      // Builds the prompt from the tabs, runs the engine and post-processes
      // the label.
      predict: ({ input }) =>
        manager.getPredictedLabelForGroup(input.tabs, input.other_tabs),
    });
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
