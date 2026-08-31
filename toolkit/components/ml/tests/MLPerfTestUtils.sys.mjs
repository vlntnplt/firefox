/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { getInferenceProcessInfo } from "chrome://global/content/ml/Utils.sys.mjs";
import { EngineProcess } from "chrome://global/content/ml/EngineProcess.sys.mjs";
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

/**
 * Shared mechanism for ML end-to-end performance tests: environment,
 * measurement policy, and reporting. Feature tests own their scenario and
 * metric names.
 */
export const MLPerfTestUtils = {
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
   * Creates a journal collecting measurement series and reporting them in the
   * format the mozperftest runner scrapes.
   *
   * @param {object} harness - The calling test's harness globals.
   * @param {Function} harness.info - The mochitest info logger.
   * @param {object} harness.Assert - The mochitest Assert object.
   * @returns {{add: (name: string, value: number) => void, report: () => void}}
   */
  createJournal({ info, Assert }) {
    const series = new Map();

    return {
      add(name, value) {
        Assert.ok(Number.isFinite(value), `${name} is a finite measurement`);
        if (!series.has(name)) {
          series.set(name, []);
        }
        series.get(name).push(value);
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
   * Series are named "<metricPrefix>-<span name>" plus a suffix for the
   * engine lifecycle state the scenario ran from; perfherder declarations
   * match emitted names as substrings, so the suffixed series inherit the
   * bare name's declaration.
   *
   * - "-first-use": one session-fresh run, capturing the one-time cost of
   *   landing the model.
   * - "-cold": runs with the engine process torn down before each; model
   *   reads may still be served from the OS page cache.
   * - "-warm": back-to-back runs after a priming run, meaningful only for
   *   features whose engines stay resident (timeoutMS).
   *
   * "<metricPrefix>-peak-memory" is the inference process peak over further
   * cold runs, sampled around the scenario so it never perturbs a latency
   * series. Scenarios whose engines live for less than a few sampling
   * intervals should request a faster memorySampleIntervalMs.
   *
   * @param {object} config
   * @param {Function} config.info - The mochitest info logger.
   * @param {object} config.Assert - The mochitest Assert object.
   * @param {string} config.metricPrefix - Prefix for every reported series.
   * @param {() => Promise<Record<string, number>>} config.scenario - Runs one
   *   full user interaction and resolves with its spans, keyed by span name.
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
    scenario,
    coldIterations = 5,
    warmIterations = 0,
    memoryIterations = 3,
    memorySampleIntervalMs = 100,
  }) {
    const journal = this.createJournal({ info, Assert });
    const restorePrefs = applyPrefs(modelHubPrefs());

    const addSpans = (spans, suffix) => {
      for (const [name, value] of Object.entries(spans)) {
        journal.add(`${metricPrefix}-${name}${suffix}`, value);
      }
    };

    try {
      addSpans(await scenario(), "-first-use");

      for (let i = 0; i < coldIterations; i++) {
        await destroyEngines();
        addSpans(await scenario(), "-cold");
      }

      if (warmIterations) {
        await destroyEngines();
        await scenario();
        for (let i = 0; i < warmIterations; i++) {
          addSpans(await scenario(), "-warm");
        }
      }

      for (let i = 0; i < memoryIterations; i++) {
        await destroyEngines();
        const sampler = this.startPeakInferenceMemorySampler(
          memorySampleIntervalMs
        );
        try {
          await scenario();
        } finally {
          journal.add(`${metricPrefix}-peak-memory`, await sampler.stop());
        }
      }
    } finally {
      await destroyEngines();
      restorePrefs();
    }

    journal.report();
  },
};
