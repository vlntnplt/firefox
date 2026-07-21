/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/// <reference path="head.js" />

// Telemetry shape test for the native llama.cpp engine workflow.
//
// Drives the canonical workflow against the real engine and the in-tree
// TinyStories model -- create, run, runWithGenerator, a deliberate run
// failure, a deliberate creation failure, terminate -- and asserts that
// every firefox_ai_runtime metric the workflow touches fires with the
// expected shape: counts, label taxonomy, extras present and parsing as
// sane numbers. Timing values are deliberately not asserted; they differ
// between measurement sources and are reviewed as perf-compare deltas.
//
// The tables below are the coverage contract: diff them against
// toolkit/components/ml/metrics.yaml when reviewing. Metrics not listed:
// model_deletion (cache management, not the engine workflow),
// session_start/session_end (require an explicit sessionId/flow setup,
// covered by browser_ml_telemetry.js), model_download (covered on the
// llama path by browser_ml_engine_rs_hub.js-style download tests; with
// the chrome:// hub the fetch is a cache path and does not reliably
// record here).

requestLongerTimeout(120);

const TELEMETRY_ENGINE_ID = "ml-native-telemetry";

// Fixtures (TinyStories options, greedy samplers, storyteller prompt,
// stream drain) live in head.js.
const TELEMETRY_LLAMA_OPTIONS = tinyStoriesOptions({
  engineId: TELEMETRY_ENGINE_ID,
  featureId: "test-feature",
});

function labeledTimingCount(metricName, label = TELEMETRY_ENGINE_ID) {
  return Glean.firefoxAiRuntime[metricName][label]?.testGetValue()?.count || 0;
}

function eventCount(metricName) {
  return Glean.firefoxAiRuntime[metricName].testGetValue()?.length || 0;
}

// Asserts an event extra is present and parses as a number >= 0. Glean
// stores quantities as strings.
function assertNumericExtra(extra, key, metricName) {
  Assert.notEqual(
    extra[key],
    undefined,
    `${metricName} extra ${key} should be present`
  );
  const number = Number(extra[key]);
  Assert.ok(
    !Number.isNaN(number),
    `${metricName} extra ${key} should be a number`
  );
  Assert.greaterOrEqual(number, 0, `${metricName} extra ${key} should be >= 0`);
}

// The success-path contract: one createEngine, then one run() and one
// runWithGenerator() on the same engine. Deltas are exact so a metric
// that stops firing, or fires twice, fails the walk.
const SUCCESS_METRICS = [
  { name: "engineCreationSuccess", kind: "labeled-timing", delta: 1 },
  { name: "runInferenceSuccess", kind: "labeled-timing", delta: 2 },
  {
    name: "runInferenceSuccessFlow",
    kind: "event",
    delta: 2,
    numericExtras: [
      "input_tokens",
      "output_tokens",
      "time_to_first_token",
      "tokens_per_second",
    ],
  },
  {
    name: "engineRun",
    kind: "event",
    delta: 2,
    numericExtras: ["wall_milliseconds", "cores", "memory_bytes"],
    identityExtras: {
      engine_id: TELEMETRY_ENGINE_ID,
      feature_id: "test-feature",
      model_id: "Mozilla/test-llama",
      backend: "llama.cpp",
    },
  },
];

// The failure-path contract: a run that rejects and a creation that
// rejects must each record their failure event, and must not record a
// success.
const FAILURE_METRICS = [
  { name: "runInferenceFailure", kind: "event", delta: 1 },
  { name: "engineCreationFailure", kind: "event", delta: 1 },
  { name: "runInferenceSuccess", kind: "labeled-timing", delta: 0 },
];

function snapshotCounts(table) {
  const counts = {};
  for (const row of table) {
    counts[row.name] =
      row.kind === "event"
        ? eventCount(row.name)
        : labeledTimingCount(row.name);
  }
  return counts;
}

function assertDeltas(table, before) {
  for (const row of table) {
    const after =
      row.kind === "event"
        ? eventCount(row.name)
        : labeledTimingCount(row.name);
    Assert.equal(
      after,
      before[row.name] + row.delta,
      `${row.name} should have recorded exactly ${row.delta} new ${
        row.kind === "event" ? "event(s)" : "sample(s)"
      }`
    );
  }
}

async function runSuccessTelemetryWalk() {
  const { cleanup } = await setup();
  try {
    const before = snapshotCounts(SUCCESS_METRICS);

    const engine = await createEngine(TELEMETRY_LLAMA_OPTIONS);
    try {
      await engine.run({
        prompt: TINYSTORIES_STORYTELLER_PROMPT,
        samplers: TINYSTORIES_GREEDY_SAMPLERS,
        nPredict: 16,
      });
      const { text } = await drainGenerator(engine, { nPredict: 16 });
      Assert.greater(text.length, 0, "The generator produced text");
    } finally {
      await engine.terminate?.();
    }

    assertDeltas(SUCCESS_METRICS, before);

    for (const row of SUCCESS_METRICS) {
      if (row.kind !== "event") {
        continue;
      }
      const events = Glean.firefoxAiRuntime[row.name].testGetValue();
      const { extra } = events.at(-1);
      for (const key of row.numericExtras ?? []) {
        assertNumericExtra(extra, key, row.name);
      }
      for (const [key, expected] of Object.entries(row.identityExtras ?? {})) {
        Assert.equal(
          extra[key],
          expected,
          `${row.name} extra ${key} should carry the expected label`
        );
      }
    }
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
}

add_task(async function test_llama_workflow_success_telemetry() {
  await runSuccessTelemetryWalk();
});

add_task(async function test_llama_workflow_failure_telemetry() {
  const { cleanup } = await setup();
  try {
    const before = snapshotCounts(FAILURE_METRICS);

    const engine = await createEngine(TELEMETRY_LLAMA_OPTIONS);
    try {
      await Assert.rejects(
        engine.run({
          prompt: TINYSTORIES_STORYTELLER_PROMPT,
          samplers: [{ type: "dist-invalid" }],
        }),
        /dist-invalid/,
        "A run with an invalid sampler rejects"
      );
    } finally {
      await engine.terminate?.();
    }

    await Assert.rejects(
      createEngine({
        ...TELEMETRY_LLAMA_OPTIONS,
        engineId: "ml-native-telemetry-missing",
        modelFile: "does-not-exist.gguf",
      }),
      /Failed to fetch the model file/,
      "Engine creation with a missing model file rejects with the hub's " +
        "fetch error"
    );

    assertDeltas(FAILURE_METRICS, before);
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});
