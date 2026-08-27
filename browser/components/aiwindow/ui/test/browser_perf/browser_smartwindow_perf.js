/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/**
 * End-to-end responsiveness of Smart Window with mocked MLPA.
 *
 * All flows include any on-device ML calls that could occur during a Smart Window
 * user interaction. To support this and avoid fetching models and Remote Settings
 * in every subtest, this reuses the taskcluster fetches from the toolkit/component/ml
 * perftest suite.
 *
 * Each metric is an interval read from the "SmartWindow" profiler markers the
 * product emits, with the LLM calls stubbed to answer instantly. These metrics
 * track the overhead introduced in the "hot" paths of Smart Window:
 *
 *  ttft-overhead: from message submitted to first token, excluding MLPA calls
 *  tail-overhead: from first token to turn completely processed
 *
 * Each metric is measured in various scenarios, as some algorithms like the memory
 * indexing have execution times that depend on the user profile state. The scenarios
 * exercise various settings for:
 *
 *  Window state: cold | warm
 *     A cold window has no ML engines running and nothing cached (like the memory embeddings)
 *     A warm window has warm engines and some cache is possible
 *  Profile:
 *     visits: number of entries in the history
 *     memories: number of memories in store
 *
 */

const perfMetadata = {
  owner: "GenAI Team",
  name: "browser_smartwindow_perf.js",
  description:
    "User-perceived responsiveness of Smart Window across window states and profile sizes",
  options: {
    default: {
      perfherder: true,
      perfherder_metrics: [
        { name: "ttft-overhead", unit: "ms", shouldAlert: true },
        { name: "tail-overhead", unit: "ms", shouldAlert: true },
      ],
      verbose: true,
      manifest: "perftest.toml",
      manifest_flavor: "browser-chrome",
      try_platform: ["linux", "mac", "win"],
    },
  },
};

requestLongerTimeout(30);

const { EngineProcess } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);
const { MemoryStore } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/services/MemoryStore.sys.mjs"
);
const { ProfilerTestUtils } = ChromeUtils.importESModule(
  "resource://testing-common/ProfilerTestUtils.sys.mjs"
);

const ITERATIONS = 5;
const CHUNKS = ["The ", "quick ", "brown ", "fox."];
const PROMPT = "what should I look at next?";
const LONG_PROMPT = "Summarize the following passage. ".repeat(300);
const DEEP_CONVERSATION_TURNS = 5;

const TTFT_MARKER = "Time to first token (TTFT)";
const TURNAROUND_MARKER = "Total turnaround time";
const SERVER_MARKER = "ServerE2E";

const PROFILES = {
  "fresh-profile": { visits: 0, memories: 0 },
  "medium-profile": { visits: 2000, memories: 20 },
  "large-profile": { visits: 20000, memories: 200 },
};

// One row per measured situation. Grouped by profile so the corpus is seeded
// once rather than once per iteration.
const SCENARIOS = [
  { profile: "fresh-profile", windowState: "cold" },
  { profile: "fresh-profile", windowState: "warm" },
  { profile: "medium-profile", windowState: "cold" },
  { profile: "medium-profile", windowState: "warm" },
  { profile: "medium-profile", windowState: "warm", prompt: "long-prompt" },
  { profile: "medium-profile", windowState: "warm", prompt: "deep-chat" },
  { profile: "large-profile", windowState: "cold" },
  { profile: "large-profile", windowState: "warm" },
];

/**
 * Subtest name for a metric in a scenario, e.g. "ttft-overhead-cold-large-profile"
 * or "tail-overhead-warm-medium-profile-long-prompt".
 *
 * @param {string} metric - Metric family.
 * @param {object} scenario - Row from SCENARIOS.
 * @returns {string} The subtest name.
 */
function metricName(metric, scenario) {
  const parts = [metric, scenario.windowState, scenario.profile];
  if (scenario.prompt) {
    parts.push(scenario.prompt);
  }
  return parts.join("-");
}

const journal = {};

function record(name, value) {
  if (!journal[name]) {
    journal[name] = [];
  }
  journal[name].push(value);
}

/**
 * Fills history and the memory store to the size the profile describes. Called
 * once per profile.
 *
 * @param {string} name - Key into PROFILES.
 */
async function seedProfile(name) {
  const { visits, memories } = PROFILES[name];

  await PlacesUtils.history.clear();
  for (let start = 0; start < visits; start += 1000) {
    const pages = [];
    for (let i = start; i < Math.min(start + 1000, visits); i++) {
      pages.push({
        url: `https://example.com/${name}/page-${i}`,
        title: `${name} history entry ${i}`,
        visits: [{ transition: PlacesUtils.history.TRANSITIONS.LINK }],
      });
    }
    await PlacesUtils.history.insertMany(pages);
  }

  for (const memory of await MemoryStore.getMemories({
    includeSoftDeleted: true,
  })) {
    await MemoryStore.hardDeleteMemory(memory.id);
  }
  for (let i = 0; i < memories; i++) {
    await MemoryStore.addMemory({
      id: `perf-mem-${i}`,
      memory_summary: `Reads about ${name} topic number ${i} regularly`,
    });
  }
}

/**
 * Puts the browser in the state the scenario's window axis describes. A cold
 * window has no inference engines running and nothing embedded yet.
 *
 * @param {string} windowState - "cold" or "warm".
 */
async function applyWindowState(windowState) {
  if (windowState !== "cold") {
    return;
  }
  await EngineProcess.destroyMLEngine();
  await TestUtils.waitForCondition(
    () => EngineProcess.areAllEnginesTerminated(),
    "Wait for the inference engines to terminate"
  );
  MemoryStore._clearEmbeddingsCache();
}

/**
 * Types a prompt, selects the chat action, and submits it. Completion is
 * observed by the caller through the turnaround marker count.
 *
 * @param {MozBrowser} browser - The browser hosting ai-window.
 * @param {string} text - The prompt to submit.
 */
async function submitPrompt(browser, text) {
  await SpecialPowers.spawn(browser, [text], async prompt => {
    const aiWindow = content.document.querySelector("ai-window");
    const bar = await ContentTaskUtils.waitForCondition(() => {
      const el = aiWindow.shadowRoot.querySelector("#ai-window-smartbar");
      return el && el.inputField && el;
    }, "Wait for the smartbar");
    bar.value = prompt.slice(0, -1);
    bar.inputField.focus();
    EventUtils.sendString(prompt.slice(-1), content);
    await bar.lastQueryContextPromise;
  });

  await selectExplicitSmartbarAction(browser, "chat");
  await waitForSmartbarAction(browser, "chat");

  // The action dropdown keeps focus, so the input is refocused before Enter.
  await SpecialPowers.spawn(browser, [], async () => {
    const aiWindow = content.document.querySelector("ai-window");
    const bar = aiWindow.shadowRoot.querySelector("#ai-window-smartbar");
    bar.inputField.focus();
    await ContentTaskUtils.waitForCondition(
      () => bar.matches(":focus-within"),
      "Wait for the smartbar to accept focus"
    );
  });

  await SpecialPowers.spawn(browser, [], () => {
    EventUtils.synthesizeKey("KEY_Enter", {}, content);
  });
}

/**
 * Collects the "SmartWindow" markers of the parent process, where ai-window
 * and the chat pipeline run. Reading only the local process cannot block on
 * other processes, unlike a full profile capture.
 *
 * @returns {Array<{label: string, duration: number}>}
 */
function smartWindowMarkers() {
  const into = [];
  for (const thread of Services.profiler.getProfileData().threads) {
    const { schema, data } = thread.markers;
    for (const row of data) {
      if (thread.stringTable[row[schema.name]] !== "SmartWindow") {
        continue;
      }
      const payload = row[schema.data];
      // The text payload's name is interned in the thread's string table.
      let label = payload && payload.name;
      if (typeof label === "number") {
        label = thread.stringTable[label];
      }
      into.push({
        label: label || "",
        duration: row[schema.endTime] - row[schema.startTime],
      });
    }
  }
  return into;
}

function markerDurations(markers, label) {
  return markers
    .filter(marker => marker.label === label)
    .map(marker => marker.duration);
}

/**
 * Waits until the given number of turns have fully processed, using the
 * turnaround markers as the completion signal.
 *
 * @param {number} turns - Total turns expected since the profiler started.
 */
async function waitForCompletedTurns(turns) {
  try {
    await TestUtils.waitForCondition(
      () =>
        markerDurations(smartWindowMarkers(), TURNAROUND_MARKER).length >=
        turns,
      `Wait for turn ${turns} to completely process`,
      100,
      600
    );
  } catch (e) {
    const labels = smartWindowMarkers().map(marker => marker.label);
    info(`Markers observed while waiting: ${JSON.stringify(labels)}`);
    throw e;
  }
}

/**
 * Runs one iteration of a scenario in its own window.
 *
 * @param {object} scenario - Row from SCENARIOS.
 * @returns {Promise<object>} The measured intervals.
 */
async function runScenario(scenario) {
  await applyWindowState(scenario.windowState);

  const win = await openAIWindow();
  try {
    const browser = win.gBrowser.selectedBrowser;
    await getAichatBrowser(browser);
    const prompt = scenario.prompt === "long-prompt" ? LONG_PROMPT : PROMPT;
    let markers;
    let turns = 0;

    await ProfilerTestUtils.startProfiler({
      features: ["nostacksampling"],
      threads: ["GeckoMain"],
    });
    await withServer({ streamChunks: CHUNKS }, async () => {
      // A warm window is one that has already answered something, so its
      // engines are up and its memory embeddings are cached.
      if (scenario.windowState === "warm") {
        await submitPrompt(browser, "hello there, what is this?");
        await waitForCompletedTurns(++turns);
      }
      if (scenario.prompt === "deep-chat") {
        for (let i = 0; i < DEEP_CONVERSATION_TURNS; i++) {
          await submitPrompt(browser, `tell me more about topic ${i}`);
          await waitForCompletedTurns(++turns);
        }
      }

      await submitPrompt(browser, prompt);
      await waitForCompletedTurns(++turns);
      markers = smartWindowMarkers();
    });

    Assert.ok(
      !EngineProcess.areAllEnginesTerminated(),
      "The on-device ML engines ran during the turn"
    );
    if (PROFILES[scenario.profile].memories > 0) {
      Assert.ok(
        MemoryStore.memoryEmbeddingsCache,
        "The memories embedder produced embeddings"
      );
    }
    Assert.ok(
      markers.some(marker => marker.label === SERVER_MARKER),
      "The turns went through the mocked MLPA server"
    );

    const ttfts = markerDurations(markers, TTFT_MARKER);
    const turnarounds = markerDurations(markers, TURNAROUND_MARKER);
    Assert.equal(ttfts.length, turns, "Every turn emitted a TTFT marker");
    Assert.equal(
      turnarounds.length,
      turns,
      "Every turn emitted a turnaround marker"
    );

    // The measured turn is the last one; earlier turns only warm the window.
    const ttft = ttfts[turns - 1];
    const turnaround = turnarounds[turns - 1];
    return {
      "ttft-overhead": Math.round(ttft),
      "tail-overhead": Math.round(turnaround - ttft),
    };
  } finally {
    if (Services.profiler.IsActive()) {
      await Services.profiler.StopProfiler();
    }
    await BrowserTestUtils.closeWindow(win);
  }
}

add_setup(async function () {
  const prefs = [
    ["browser.smartwindow.firstrun.modelChoice", "0"],
    // The memories flows are only reachable with ToS consent.
    ["browser.smartwindow.tos.consentTime", 123],
  ];
  const modelHubRootUrl = Services.env.get("MOZ_MODELS_HUB");
  if (modelHubRootUrl) {
    prefs.push(["browser.ml.modelHubRootUrl", modelHubRootUrl]);
  }
  await SpecialPowers.pushPrefEnv({ set: prefs });
  registerCleanupFunction(async () => {
    await PlacesUtils.history.clear();
    await SpecialPowers.popPrefEnv();
  });
});

for (const profile of Object.keys(PROFILES)) {
  const rows = SCENARIOS.filter(scenario => scenario.profile === profile);

  add_task({ name: `smartwindow_${profile}` }, async function () {
    await seedProfile(profile);

    for (const scenario of rows) {
      for (let i = 0; i < ITERATIONS; i++) {
        const measured = await runScenario(scenario);
        for (const [metric, value] of Object.entries(measured)) {
          record(metricName(metric, scenario), value);
        }
      }
    }
  });
}

add_task(function report() {
  const metrics = Object.entries(journal).map(([name, values]) => ({
    name,
    values,
    unit: "ms",
    shouldAlert: true,
  }));

  Assert.equal(
    metrics.length,
    SCENARIOS.length * 2,
    "Every scenario reported every metric"
  );
  for (const { name, values } of metrics) {
    Assert.equal(
      values.length,
      ITERATIONS,
      `${name} was measured on every iteration`
    );
  }

  info(`perfMetrics | ${JSON.stringify(metrics)}`);
});
