/* Any copyright is dedicated to the Public Domain.
http://creativecommons.org/publicdomain/zero/1.0/ */
"use strict";

// Latency test for the AI Window intent classifier
// (browser/components/aiwindow/models/IntentClassifier.sys.mjs). Measures
// both models getIntentModelInfoForLocale() can pick: they are different
// architectures, so one set of numbers does not predict the other.

const INTENT_ENGINES = [
  {
    perfName: "intent-default",
    featureId: "smart-intent",
    modelId: "Mozilla/mobilebert-query-intent-detection",
  },
  {
    // Selected when Region.home == "FR".
    perfName: "intent-en-fr",
    featureId: "smart-intent-en-fr",
    modelId: "Mozilla/query-intent-detection-en-fr",
  },
];

// Representative of what reaches the classifier after _preprocessQuery: short,
// lowercase-ish, punctuation stripped. Mixed search-leaning and chat-leaning
// phrasings so the run is not dominated by one branch.
const QUERIES = [
  "weather in lisbon tomorrow",
  "how do i convince my landlord to fix the boiler",
  "cheap flights to reykjavik in march",
  "write me a haiku about compilers",
];

const perfMetadata = {
  owner: "GenAI Team",
  name: "browser_ml_intent_perf.js",
  description: "Latency for the AI Window intent classification models",
  options: {
    default: {
      perfherder: true,
      perfherder_metrics: [
        {
          name: "latency",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "memory",
          unit: "MiB",
          shouldAlert: false,
        },
      ],
      verbose: true,
      manifest: "perftest.toml",
      manifest_flavor: "browser-chrome",
      try_platform: ["linux", "mac", "win"],
    },
  },
};

requestLongerTimeout(10);

async function runIntentModel({ perfName, featureId, modelId }) {
  const options = new PipelineOptions({
    taskName: "text-classification",
    featureId,
    modelId,
    modelHubUrlTemplate: "{model}/{revision}",
    modelRevision: "main",
    dtype: "q8",
    numThreads: 2,
    timeoutMS: -1,
  });

  const request = { args: [QUERIES] };

  // Cold start matters more here than for most features: the engine is created
  // lazily inside getPromptIntent() on the first prompt of a session, so the
  // first user interaction pays the full initialization cost.
  await runMLPerfTest({
    name: perfName,
    options,
    request,
    addColdStart: true,
  });
}

add_task(async function test_ml_intent_default() {
  await runIntentModel(INTENT_ENGINES[0]);
});

add_task(async function test_ml_intent_en_fr() {
  await runIntentModel(INTENT_ENGINES[1]);
});
