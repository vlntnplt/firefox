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

  const manager = new SmartTabGroupingManager();
  try {
    await runTopicModelFidelity({
      feature: "smarttab-topic",
      traceFile: "smarttab_topic_trace.json",
      engineId: FEATURES["smart-tab-topic"].engineId,
      config: SMART_TAB_GROUPING_CONFIG.topicGeneration,
      // Builds the prompt from the tabs, applies the model revision override,
      // runs the engine and post-processes the label.
      predict: ({ input }) =>
        manager.getPredictedLabelForGroup(input.tabs, input.other_tabs),
    });
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
