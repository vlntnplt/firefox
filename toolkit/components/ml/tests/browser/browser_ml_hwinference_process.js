/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/* global TextGenerator */

requestLongerTimeout(10);

const { ProfilerTestUtils } = ChromeUtils.importESModule(
  "resource://testing-common/ProfilerTestUtils.sys.mjs"
);

// top-k:1 only filters; without a final `dist` sampler llama.cpp crashes.
const TINYSTORIES_GREEDY_SAMPLERS = [
  { type: "top-k", topK: 1 },
  { type: "dist" },
];

const TINYSTORIES_STORYTELLER_PROMPT = [
  { role: "system", content: "You are a friendly storyteller." },
  { role: "user", content: "Once upon a time there was a small mouse who" },
];

const HWI_OPTIONS = engineId => ({
  backend: "llama.cpp",
  engineId,
  taskName: "text-generation",
  modelId: "Mozilla/test-llama",
  modelFile: "TinyStories-656K.Q8_0.gguf",
  modelRevision: "main",
  numContext: 256,
});

async function drainGenerator(engine, { nPredict = 32 } = {}) {
  let text = "";
  for await (const chunk of engine.runWithGenerator({
    prompt: TINYSTORIES_STORYTELLER_PROMPT,
    samplers: TINYSTORIES_GREEDY_SAMPLERS,
    nPredict,
  })) {
    text += chunk.text;
  }
  return { text };
}

async function hwInferencePid() {
  const procInfo = await ChromeUtils.requestProcInfo();
  for (const child of procInfo.children) {
    if (
      child.type.startsWith("utility") &&
      child.utilityActors.some(actor => actor.actorName === "hwInference")
    ) {
      return child.pid;
    }
  }
  return 0;
}

add_task(async function test_hwinference_crash_and_respawn() {
  const { cleanup } = await setup({
    prefs: [["browser.ml.llama.hwInference", true]],
  });
  try {
    // A generation long enough to still be running when the kill lands:
    // one-token chunks, no stop on end of text, and a context to fill.
    const engine = await createEngine({
      ...HWI_OPTIONS("hwi-crash"),
      numContext: 4096,
    });

    const pid = await hwInferencePid();
    Assert.greater(pid, 0, "The HWInference utility process is running");

    const processTools = Cc["@mozilla.org/processtools-service;1"].getService(
      Ci.nsIProcessToolsService
    );
    const gone = TestUtils.topicObserved(
      "ipc:utility-shutdown",
      (subject, data) => parseInt(data, 10) === pid
    );
    await Assert.rejects(
      (async () => {
        for await (const chunk of engine.runWithGenerator({
          prompt: TINYSTORIES_STORYTELLER_PROMPT,
          samplers: TINYSTORIES_GREEDY_SAMPLERS,
          nPredict: 4096,
          minOutputBufferSize: 1,
          stopOnEndOfGenerationTokens: false,
        })) {
          info(`chunk before kill: ${chunk.text}`);
          processTools.kill(pid);
        }
      })(),
      /went away|abort/i,
      "A generation interrupted by a process crash rejects with the " +
        "generator-teardown error"
    );
    await gone;
    noteIntentionalUtilityCrash(pid);
    await engine.terminate();

    const respawned = await createEngine(HWI_OPTIONS("hwi-crash-respawn"));
    const { text } = await drainGenerator(respawned, { nPredict: 8 });
    info(`Respawned engine generated: ${text}`);
    Assert.greater(
      text.length,
      0,
      "A fresh engine after the crash generates again"
    );
    const newPid = await hwInferencePid();
    Assert.greater(newPid, 0, "A HWInference utility process is running again");
    Assert.notEqual(newPid, pid, "The utility process was respawned");
    await respawned.terminate();
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

function getPayloads(profile, type) {
  return ProfilerTestUtils.getPayloadsOfTypeFromAllThreads(profile, type);
}

async function createTinyStoriesGenerator() {
  const modelPath = getTestFilePath(
    "data/Mozilla/test-llama/main/TinyStories-656K.Q8_0.gguf"
  );
  const modelFile = await File.createFromFileName(modelPath);
  return TextGenerator.create(modelFile, { contextSize: 512 });
}

add_task(async function test_process_lifecycle_markers() {
  // The suite pins the idle grace to 0; reuse needs a real idle window. A
  // deferred drop cannot be recalled by lowering the pref afterwards, so it
  // is short enough to wait out.
  const graceMs = 2000;
  await SpecialPowers.pushPrefEnv({
    set: [["browser.ml.hwinference.browser_idle_shutdown_grace_ms", graceMs]],
  });

  await ProfilerTestUtils.startProfiler({
    features: ["stackwalk", "js"],
    threads: ["GeckoMain", "TextGenerator"],
  });

  const cold = await createTinyStoriesGenerator();
  const pid = await hwInferencePid();
  Assert.greater(pid, 0, "The HWInference utility process is running");
  const gone = TestUtils.topicObserved(
    "ipc:utility-shutdown",
    (subject, data) => parseInt(data, 10) === pid
  );
  cold.terminate();
  const warm = await createTinyStoriesGenerator();
  warm.terminate();
  await gone;

  const profile = await ProfilerTestUtils.stopNowAndGetProfile();

  const spawn = getPayloads(profile, "MLProcessSpawn");
  const releases = getPayloads(profile, "MLProcessRelease");
  const creates = getPayloads(profile, "MLGeneratorCreate");

  Assert.equal(spawn.length, 1, "The process was launched exactly once");
  Assert.equal(
    getPayloads(profile, "MLFailed").length,
    0,
    "A successful launch reports no failed phase"
  );
  Assert.equal(creates.length, 2, "Both creates are marked");

  const acquires = getPayloads(profile, "MLProcessAcquire");
  Assert.equal(acquires.length, 2, "Both process acquisitions are marked");
  Assert.ok(
    !acquires[0].processReused,
    "The first acquisition paid for a spawn"
  );
  Assert.ok(acquires[1].processReused, "The second one reused the process");
  Assert.equal(
    creates.filter(payload => payload.spawnMs !== undefined).length,
    0,
    "Generator create does not fold in the process acquisition"
  );

  Assert.equal(releases.length, 2, "Both keep-alive releases are marked");
  Assert.ok(
    releases.every(payload => payload.graceMs === graceMs),
    "Each release carries the grace it waited"
  );
  Assert.ok(
    !releases[0].retired,
    "The first release found the process still held by the second generator"
  );
  Assert.ok(releases[1].retired, "The last release retired the process");

  await SpecialPowers.popPrefEnv();
});
