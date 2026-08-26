/* Any copyright is dedicated to the Public Domain.
http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/**
 * Client-side overhead of a Smart Window session's first chat turn, around a
 * mocked provider (MLPA) call:
 *
 *   submit -> [ttft-overhead-pre]  -> provider request goes out
 *          -> [mocked provider]    -> first response chunk handed back
 *          -> [ttft-overhead-post] -> first token reaches the app
 *          -> [tail-overhead]      -> completion done client-side
 *
 * TTFT overhead = pre + post; completion overhead = TTFT overhead + tail.
 * The mock sits at the OpenAI-protocol HTTP boundary, so engine build,
 * serialization, necko, and SSE parsing are all measured; auxiliary LLM
 * features get inert engine stubs. Prompts come from the in-tree RS dump.
 *
 * attribution-* metrics sum the product's SmartWindow phase markers inside
 * each scenario's boundary marker; attribution-unattributed is the pre
 * overhead no marker explains. Scenarios: send as soon as the chat is
 * usable, and send after IDLE_MS of think-time.
 */

const perfMetadata = {
  owner: "AI Platform",
  name: "browser_smartwindow_first_turn_perf.js",
  description:
    "Client-side TTFT overhead (pre/post provider call) of a Smart Window " +
    "session's first chat turn, with the provider mocked",
  options: {
    default: {
      perfherder: true,
      perfherder_metrics: [
        { name: "ttft-overhead-immediate", unit: "ms", shouldAlert: false },
        { name: "ttft-overhead-pre-immediate", unit: "ms", shouldAlert: false },
        {
          name: "ttft-overhead-post-immediate",
          unit: "ms",
          shouldAlert: false,
        },
        { name: "tail-overhead-immediate", unit: "ms", shouldAlert: false },
        { name: "ttft-overhead-after-idle", unit: "ms", shouldAlert: false },
        {
          name: "ttft-overhead-pre-after-idle",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "ttft-overhead-post-after-idle",
          unit: "ms",
          shouldAlert: false,
        },
        { name: "tail-overhead-after-idle", unit: "ms", shouldAlert: false },
        {
          name: "attribution-engine-build-immediate",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-system-prompt-immediate",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-realtime-context-immediate",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-memories-context-immediate",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-unattributed-immediate",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-engine-build-after-idle",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-system-prompt-after-idle",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-realtime-context-after-idle",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-memories-context-after-idle",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "attribution-unattributed-after-idle",
          unit: "ms",
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

const { MemoryStore } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/services/MemoryStore.sys.mjs"
);
const { MESSAGE_ROLE } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/ui/modules/ChatStore.sys.mjs"
);
const { MEMORY_SENSITIVITY_CATEGORY_NOT_SENSITIVE } =
  ChromeUtils.importESModule(
    "moz-src:///browser/components/aiwindow/models/memories/MemoriesConstants.sys.mjs"
  );

// A post-onboarding profile: enough memories that first-turn retrieval does
// real work.
const SEED_MEMORY_COUNT = 20;
const PROMPT = "What do you know about my coffee habits?";
// User think-time between the chat becoming usable and the first send.
const IDLE_MS = 2000;

const journal = {
  "ttft-overhead-immediate": [],
  "ttft-overhead-pre-immediate": [],
  "ttft-overhead-post-immediate": [],
  "tail-overhead-immediate": [],
  "ttft-overhead-after-idle": [],
  "ttft-overhead-pre-after-idle": [],
  "ttft-overhead-post-after-idle": [],
  "tail-overhead-after-idle": [],
};

// SmartWindow marker texts summed into attribution-* metrics. Markers nested
// inside these phases (e.g. MemoryStore:embed_memories) are excluded to
// avoid double counting.
const PHASE_METRICS = {
  "PromptLoader:build_engine": "engine-build",
  "ChatConversation:load_system_prompt": "system-prompt",
  "ChatConversation:inject_realtime_context": "realtime-context",
  "ChatConversation:inject_memories_context": "memories-context",
};
const BOUNDARY_MARKER = "SWPerfTestWindow";

// Chat-request arrival at the mock server; it responds immediately, so this
// timestamp both ends the pre window and starts the post window.
const chatRequestTimes = [];

add_setup(async function () {
  const modelHubRootUrl = Services.env.get("MOZ_MODELS_HUB");
  if (!modelHubRootUrl) {
    throw new Error(
      "MOZ_MODELS_HUB is not set; run via ./mach perftest with --hooks " +
        "toolkit/components/ml/tests/tools/hooks_local_hub.py and " +
        "MOZ_ML_LOCAL_DIR pointing at a directory containing onnx-models/"
    );
  }
  await SpecialPowers.pushPrefEnv({
    set: [
      ["browser.ml.modelHubRootUrl", modelHubRootUrl],
      // The flows under test are only reachable with ToS consent.
      ["browser.smartwindow.tos.consentTime", 123],
    ],
  });

  // Undo head.js's prompt stubs so the real PromptLoader path runs on
  // production-sized records: serve the in-tree ai-window-prompts dump
  // through the RS client stub.
  const { SharedUtils } = ChromeUtils.importESModule(
    "resource://services-settings/SharedUtils.sys.mjs"
  );
  const dump = await SharedUtils.loadJSONDump("main", "ai-window-prompts");
  Assert.ok(dump && dump.data.length, "loaded the in-tree RS dump");
  _setRemoteClientForTesting({ get: async () => dump.data });
  _setLoadPromptForTesting(null);

  await seedMemories();

  // Prime the model download, then return to a cold engine so no scenario
  // measures networking.
  info("Priming embedding model download");
  await MemoriesManager.getRelevantMemories("priming query");
  await shutdownEmbeddings();

  const netBoundaries = await stubEngineNetworkBoundaries({
    serverOptions: {
      streamChunks: ["First", " token", " stream."],
      onRequest(body) {
        if (body.stream) {
          chatRequestTimes.push(ChromeUtils.now());
        }
      },
    },
  });

  // Marker-only profiling for phase attribution.
  Services.profiler.StartProfiler(
    65536,
    10,
    ["nostacksampling"],
    ["GeckoMain"]
  );
  Assert.ok(Services.profiler.IsActive(), "marker-only profiler is running");

  registerCleanupFunction(async () => {
    if (Services.profiler.IsActive()) {
      Services.profiler.StopProfiler();
    }
    await netBoundaries.restore();
    await shutdownEmbeddings();
    await deleteAllMemories();
  });
});

add_task(async function test_first_turn_send_immediately() {
  await runFirstTurnScenario("immediate", { idleMs: 0 });
});

add_task(async function test_first_turn_send_after_idle() {
  await runFirstTurnScenario("after-idle", { idleMs: IDLE_MS });
});

add_task(async function report() {
  const profile = await Services.profiler.getProfileDataAsync();
  Services.profiler.StopProfiler();
  const markers = collectIntervalMarkers(profile);

  for (const label of ["immediate", "after-idle"]) {
    const window = markers.find(
      m => m.name === BOUNDARY_MARKER && m.text === label
    );
    Assert.ok(window, `found the ${label} scenario's boundary marker`);

    const sums = {};
    for (const marker of markers) {
      const metric = PHASE_METRICS[marker.text];
      if (
        metric &&
        marker.name === "SmartWindow" &&
        marker.start >= window.start &&
        marker.end <= window.end
      ) {
        sums[metric] = (sums[metric] || 0) + (marker.end - marker.start);
      }
    }

    let attributed = 0;
    for (const metric of new Set(Object.values(PHASE_METRICS))) {
      const duration = sums[metric] || 0;
      attributed += duration;
      journal[`attribution-${metric}-${label}`] = [duration];
    }
    const pre = journal[`ttft-overhead-pre-${label}`][0];
    journal[`attribution-unattributed-${label}`] = [pre - attributed];
  }

  const reported = Object.entries(journal).map(([name, values]) => ({
    name,
    values,
    value: values.length ? values[0] : 0,
  }));
  info(`perfMetrics | ${JSON.stringify(reported)}`);
});

/**
 * Flattens all interval markers out of a gecko profile, across the parent
 * process and its subprocesses.
 *
 * @param {object} profile  Output of Services.profiler.getProfileDataAsync()
 * @returns {Array<{name: string, text: ?string, start: number, end: number}>}
 */
function collectIntervalMarkers(profile) {
  const found = [];
  const walkProcess = process => {
    for (const thread of process.threads || []) {
      const { markers, stringTable } = thread;
      const schema = markers.schema;
      for (const row of markers.data) {
        const start = row[schema.startTime];
        const end = row[schema.endTime];
        if (typeof start !== "number" || typeof end !== "number") {
          continue;
        }
        const payload = row[schema.data];
        // Text payloads intern their string in the thread's stringTable.
        let text = payload ? payload.name : null;
        if (typeof text === "number") {
          text = stringTable[text];
        }
        found.push({
          name: stringTable[row[schema.name]],
          text,
          start,
          end,
        });
      }
    }
    (process.processes || []).forEach(walkProcess);
  };
  walkProcess(profile);
  return found;
}

/**
 * Opens a fresh Smart Window on a cold embeddings engine, optionally idles,
 * sends one message, and records the pre/post provider-call overhead.
 *
 * @param {string} label   Scenario suffix for journal metric names
 * @param {object} options
 * @param {number} options.idleMs  Think-time before the send, in ms
 */
async function runFirstTurnScenario(label, { idleMs }) {
  await shutdownEmbeddings();

  const { win } = await openAIWindowWithSidebar();
  const sidebarBrowser = win.document.getElementById("ai-window-browser");

  if (idleMs) {
    // eslint-disable-next-line mozilla/no-arbitrary-setTimeout
    await new Promise(resolve => win.setTimeout(resolve, idleMs));
  }

  const conversation = AIWindow.getActiveConversation(win);
  const firstToken = new Promise(resolve => {
    conversation.on("chat-conversation:message-update", (_e, message) => {
      if (
        message &&
        message.role === MESSAGE_ROLE.ASSISTANT &&
        message.content &&
        message.content.body
      ) {
        resolve(ChromeUtils.now());
      }
    });
  });
  const messageComplete = new Promise(resolve => {
    conversation.on("chat-conversation:message-complete", (_e, message) => {
      if (message && message.role === MESSAGE_ROLE.ASSISTANT) {
        resolve(ChromeUtils.now());
      }
    });
  });

  await typeInSmartbar(sidebarBrowser, PROMPT);
  chatRequestTimes.length = 0;
  const tSubmit = ChromeUtils.now();
  await submitSmartbar(sidebarBrowser);

  const tFirstToken = await firstToken;
  const tComplete = await messageComplete;

  Assert.equal(
    chatRequestTimes.length,
    1,
    "exactly one chat request reached the mock server"
  );
  const tRequest = chatRequestTimes[0];
  const pre = tRequest - tSubmit;
  const post = tFirstToken - tRequest;
  journal[`ttft-overhead-${label}`].push(pre + post);
  journal[`ttft-overhead-pre-${label}`].push(pre);
  journal[`ttft-overhead-post-${label}`].push(post);
  journal[`tail-overhead-${label}`].push(tComplete - tFirstToken);

  // Brackets this scenario's markers in profile time; no clock-domain
  // conversion needed.
  ChromeUtils.addProfilerMarker(BOUNDARY_MARKER, { startTime: tSubmit }, label);

  await BrowserTestUtils.closeWindow(win);
}

async function seedMemories() {
  for (let i = 0; i < SEED_MEMORY_COUNT; i++) {
    await MemoryStore.addMemory({
      memory_summary: `Interest number ${i}: enjoys topic-${i} content`,
      reasoning: `Regularly visits sites about topic-${i}`,
      tags: [`category:Topic ${i}`, "intent:Entertain / Relax"],
      sources: ["history"],
      sensitivity_category: MEMORY_SENSITIVITY_CATEGORY_NOT_SENSITIVE,
      recent_accessed_counts: { 0: 0 },
    });
  }
}

async function deleteAllMemories() {
  const memories = await MemoryStore.getMemories({ includeSoftDeleted: true });
  for (const memory of memories) {
    await MemoryStore.hardDeleteMemory(memory.id);
  }
}

/**
 * Terminates the embeddings engine and drops the corpus cache so the next
 * embedding call starts from a cold engine (the model stays on disk).
 */
async function shutdownEmbeddings() {
  try {
    if (MemoryStore.embeddingsGenerator) {
      await MemoryStore.embeddingsGenerator.shutdown();
    }
  } catch (e) {
    info(`embeddings shutdown: ${e}`);
  }
  MemoryStore.embeddingsGenerator = null;
  MemoryStore._clearEmbeddingsCache();
}
