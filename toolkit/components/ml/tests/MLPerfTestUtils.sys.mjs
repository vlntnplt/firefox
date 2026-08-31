/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { getInferenceProcessInfo } from "chrome://global/content/ml/Utils.sys.mjs";
import { EngineProcess } from "chrome://global/content/ml/EngineProcess.sys.mjs";
import {
  MLEngine,
  MLEngineParent,
} from "moz-src:///toolkit/components/ml/actors/MLEngineParent.sys.mjs";
import { TestUtils } from "resource://testing-common/TestUtils.sys.mjs";
import {
  setInterval,
  clearInterval,
} from "resource://gre/modules/Timer.sys.mjs";

const ONE_MIB = 1024 * 1024;

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2) {
    return sorted[middle];
  }
  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function kebabCase(name) {
  return name.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`);
}

// Sets prefs on the user branch and returns a restore function.
function applyPrefs(prefs) {
  const setPref = (name, value) =>
    typeof value == "boolean"
      ? Services.prefs.setBoolPref(name, value)
      : Services.prefs.setStringPref(name, value);

  const getPref = (name, value) =>
    typeof value == "boolean"
      ? Services.prefs.getBoolPref(name)
      : Services.prefs.getStringPref(name);

  const saved = prefs.map(([name, value]) => {
    const had = Services.prefs.prefHasUserValue(name);
    const old = had ? getPref(name, value) : null;
    setPref(name, value);
    return { name, had, old };
  });

  return () => {
    for (const { name, had, old } of saved) {
      if (had) {
        setPref(name, old);
      } else {
        Services.prefs.clearUserPref(name);
      }
    }
  };
}

async function destroyEngines() {
  await EngineProcess.destroyMLEngine();
  await TestUtils.waitForCondition(
    () => EngineProcess.areAllEnginesTerminated(),
    "Waiting for all ML engines to terminate"
  );
}

function modelHubPrefs() {
  const modelHubRootUrl = Services.env.get("MOZ_MODELS_HUB");
  if (!modelHubRootUrl) {
    throw new Error(
      "MOZ_MODELS_HUB is not set; run with --hooks toolkit/components/ml/tests/tools/hooks_local_hub.py"
    );
  }

  return [
    ["browser.ml.enable", true],
    ["browser.ml.modelHubRootUrl", modelHubRootUrl],
    ["browser.ml.modelHubUrlTemplate", "{model}/{revision}"],
  ];
}

// The measurement currently capturing engine internals, if any. Only
// measureScenario sets this, for the duration of one scenario.
let activeCapture = null;

function captureFor(pipelineOptions) {
  if (activeCapture && pipelineOptions?.featureId === activeCapture.featureId) {
    return activeCapture;
  }
  return null;
}

// Test-side wrappers recording engine creations and completed runs for the
// feature a measurement is capturing; every other caller passes through
// untouched. yield* forwards the stream to the feature unchanged and
// evaluates to the generator's final response; a stream its consumer
// abandons never reaches that response, so such a run records nothing.
const original = {
  getEngine: MLEngineParent.prototype.getEngine,
  run: MLEngine.prototype.run,
  runWithGenerator: MLEngine.prototype.runWithGenerator,
};

MLEngineParent.prototype.getEngine = async function (params) {
  const capture = captureFor(params.pipelineOptions);
  if (!capture) {
    return original.getEngine.call(this, params);
  }
  const start = ChromeUtils.now();
  const engine = await original.getEngine.call(this, params);
  capture.engineCreations.push(ChromeUtils.now() - start);
  return engine;
};

MLEngine.prototype.run = async function (request) {
  const result = await original.run.call(this, request);
  if (result?.metrics) {
    captureFor(this.pipelineOptions)?.runMetrics.push(result.metrics);
  }
  return result;
};

MLEngine.prototype.runWithGenerator = async function* (request) {
  const result = yield* original.runWithGenerator.call(this, request);
  if (result?.metrics) {
    captureFor(this.pipelineOptions)?.runMetrics.push(result.metrics);
  }
  return result;
};

/**
 * Shared mechanism for ML end-to-end performance tests: environment,
 * measurement, and reporting. Feature tests own their scenario and metric
 * names.
 */
export const MLPerfTestUtils = {
  /**
   * Measures one full user interaction and returns everything observed as a
   * plain record; callers decide what to report.
   *
   * With a featureId, that feature's MLEngine activity is captured for the
   * duration of the call: creation durations, and for every run that
   * completes, the metrics the backend reports with its final response. A
   * run the feature abandons mid-stream produces no final response and so
   * no run metrics.
   *
   * @param {() => Promise<Record<string, number>>} scenario - Runs one full
   *   user interaction and resolves with its spans, keyed by span name.
   * @param {object} [options]
   * @param {string} [options.featureId] - Capture engine internals for this
   *   feature.
   * @param {boolean} [options.sampleMemory=false] - Sample the inference
   *   process memory during the scenario. Sampling perturbs it.
   * @param {number} [options.memorySampleIntervalMs=100] - Sampling interval.
   * @returns {Promise<{
   *   spans: Record<string, number>,
   *   engineCreations: number[],
   *   runMetrics: object[],
   *   peakMemory: number | undefined,
   * }>} peakMemory is the sampled peak in MiB.
   */
  async measureScenario(
    scenario,
    { featureId, sampleMemory = false, memorySampleIntervalMs = 100 } = {}
  ) {
    if (activeCapture) {
      throw new Error("Another measurement is already in progress");
    }
    const capture = { featureId, engineCreations: [], runMetrics: [] };
    const sampler = sampleMemory
      ? this.startPeakInferenceMemorySampler(memorySampleIntervalMs)
      : null;

    activeCapture = featureId ? capture : null;
    let spans;
    let peakMemory;
    try {
      spans = await scenario();
    } finally {
      activeCapture = null;
      peakMemory = sampler ? await sampler.stop() : undefined;
    }

    return {
      spans,
      engineCreations: capture.engineCreations,
      runMetrics: capture.runMetrics,
      peakMemory,
    };
  },

  /**
   * Samples the inference process memory until stopped, with a final sample
   * taken by stop(). Each sample is a full requestProcInfo walk, so keep the
   * interval reasonable.
   *
   * @param {number} [intervalMs=100] - The sampling interval.
   * @returns {{stop: () => Promise<number>}} A sampler whose stop() resolves
   *   with the peak memory in MiB.
   */
  startPeakInferenceMemorySampler(intervalMs = 100) {
    let peakMemory = 0;
    let chain = Promise.resolve();

    const sample = () => {
      chain = chain
        .then(async () => {
          const procInfo = await getInferenceProcessInfo();
          if (procInfo.memory && procInfo.memory > peakMemory) {
            peakMemory = procInfo.memory;
          }
        })
        .catch(() => {});
    };

    sample();
    const intervalId = setInterval(sample, intervalMs);

    return {
      async stop() {
        clearInterval(intervalId);
        sample();
        await chain;
        return Math.round(peakMemory / ONE_MIB);
      },
    };
  },

  /**
   * Creates a journal collecting measurement series under
   * "<metricPrefix>-<name>" and reporting them in the format the mozperftest
   * runner scrapes.
   *
   * addRecord maps a measureScenario record to suffixed series: each span by
   * its name, "engine-creation" per engine created, and every numeric metric
   * of each completed run kebab-cased (e.g. timeToFirstToken becomes
   * "time-to-first-token"). The run vocabulary is the backend's; the test's
   * perfherder_metrics declarations select which series perfherder ingests,
   * since mozperftest drops any series whose name contains no declared
   * metric name, while every series stays in the perfMetrics log line.
   * peakMemory is deliberately not mapped; callers report it explicitly from
   * the records they trust.
   *
   * @param {object} config
   * @param {Function} config.info - The mochitest info logger.
   * @param {object} config.Assert - The mochitest Assert object.
   * @param {string} config.metricPrefix - Prefix for every series name.
   * @returns {{
   *   add: (name: string, value: number) => void,
   *   addRecord: (record: object, suffix: string) => void,
   *   report: () => void,
   * }}
   */
  createJournal({ info, Assert, metricPrefix }) {
    const series = new Map();

    return {
      add(name, value) {
        const fullName = `${metricPrefix}-${name}`;
        Assert.ok(
          Number.isFinite(value),
          `${fullName} is a finite measurement`
        );
        if (!series.has(fullName)) {
          series.set(fullName, []);
        }
        series.get(fullName).push(value);
      },

      addRecord(record, suffix) {
        for (const [name, value] of Object.entries(record.spans)) {
          this.add(`${name}${suffix}`, value);
        }
        for (const value of record.engineCreations) {
          this.add(`engine-creation${suffix}`, value);
        }
        for (const metrics of record.runMetrics) {
          for (const [field, value] of Object.entries(metrics)) {
            if (Number.isFinite(value)) {
              this.add(`${kebabCase(field)}${suffix}`, value);
            }
          }
        }
      },

      report() {
        const metrics = [];
        for (const [name, values] of series) {
          metrics.push({ name, values, value: median(values) });
        }
        info(`perfMetrics | ${JSON.stringify(metrics)}`);
      },
    };
  },

  /**
   * Runs a feature scenario through the standard measurement policy and
   * reports the collected series. While it runs, ML is enabled and the
   * default model hub is redirected to the local hub server (MOZ_MODELS_HUB,
   * served by the hooks_local_hub.py hook); the prefs are restored and the ML
   * engine process torn down before it resolves.
   *
   * Each measurement's series carry a suffix for the engine lifecycle state
   * it ran from:
   *
   * - "-first-use": one session-fresh run, capturing the one-time cost of
   *   landing the model. Turn off measureFirstUse for scenarios that cannot
   *   run session-fresh, e.g. a query flow whose model was already landed by
   *   an ingestion flow measured before it.
   * - "-cold": runs with the engine process torn down before each; model
   *   reads may still be served from the OS page cache.
   * - "-warm": back-to-back runs after an unreported priming run, meaningful
   *   only for features whose engines stay resident (timeoutMS).
   *
   * "<metricPrefix>-peak-memory" comes from further cold runs that report
   * nothing else, so sampling never perturbs another series. Scenarios whose
   * engines live for less than a few sampling intervals should request a
   * faster memorySampleIntervalMs.
   *
   * @param {object} config
   * @param {Function} config.info - The mochitest info logger.
   * @param {object} config.Assert - The mochitest Assert object.
   * @param {string} config.metricPrefix - Prefix for every reported series.
   * @param {string} [config.featureId] - Capture engine internals for this
   *   feature (see measureScenario and createJournal.addRecord).
   * @param {() => Promise<Record<string, number>>} config.scenario - Runs one
   *   full user interaction and resolves with its spans, keyed by span name.
   * @param {boolean} [config.measureFirstUse=true] - Measure the
   *   session-fresh run.
   * @param {number} [config.coldIterations=5] - Cold-engine latency runs.
   * @param {number} [config.warmIterations=0] - Warm-engine latency runs.
   * @param {number} [config.memoryIterations=3] - Memory-sampled runs.
   * @param {number} [config.memorySampleIntervalMs=100] - Memory sampling
   *   interval.
   * @returns {Promise<void>}
   */
  async runPerfScenario({
    info,
    Assert,
    metricPrefix,
    featureId,
    scenario,
    measureFirstUse = true,
    coldIterations = 5,
    warmIterations = 0,
    memoryIterations = 3,
    memorySampleIntervalMs = 100,
  }) {
    const journal = this.createJournal({ info, Assert, metricPrefix });
    const restorePrefs = applyPrefs(modelHubPrefs());
    info(
      `MLPerf environment: modelHubRootUrl=${Services.prefs.getStringPref(
        "browser.ml.modelHubRootUrl"
      )}`
    );
    const measure = options =>
      this.measureScenario(scenario, { featureId, ...options });

    try {
      if (measureFirstUse) {
        await destroyEngines();
        const firstUse = await measure();
        if (featureId) {
          Assert.greater(
            firstUse.engineCreations.length,
            0,
            `The first use created an engine for "${featureId}"`
          );
        }
        journal.addRecord(firstUse, "-first-use");
      }

      for (let i = 0; i < coldIterations; i++) {
        await destroyEngines();
        journal.addRecord(await measure(), "-cold");
      }

      if (warmIterations) {
        await destroyEngines();
        await measure();
        for (let i = 0; i < warmIterations; i++) {
          journal.addRecord(await measure(), "-warm");
        }
      }

      for (let i = 0; i < memoryIterations; i++) {
        await destroyEngines();
        const { peakMemory } = await measure({
          sampleMemory: true,
          memorySampleIntervalMs,
        });
        Assert.greater(
          peakMemory,
          0,
          "The memory sampler saw the inference process"
        );
        journal.add("peak-memory", peakMemory);
      }
    } finally {
      await destroyEngines();
      restorePrefs();
    }

    journal.report();
  },
};
