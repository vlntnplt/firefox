/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

requestLongerTimeout(10);

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
    const engine = await createEngine(HWI_OPTIONS("hwi-crash"));

    const pid = await hwInferencePid();
    Assert.greater(pid, 0, "The HWInference utility process is running");

    const processTools = Cc["@mozilla.org/processtools-service;1"].getService(
      Ci.nsIProcessToolsService
    );
    await Assert.rejects(
      (async () => {
        for await (const chunk of engine.runWithGenerator({
          prompt: TINYSTORIES_STORYTELLER_PROMPT,
          samplers: TINYSTORIES_GREEDY_SAMPLERS,
          nPredict: 512,
        })) {
          info(`chunk before kill: ${chunk.text}`);
          processTools.kill(pid);
        }
      })(),
      /went away|abort/i,
      "A generation interrupted by a process crash rejects with the " +
        "generator-teardown error"
    );
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
