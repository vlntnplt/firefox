/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// The crash path of the HWInference routing: kill the utility process
// mid-generation, the run rejects, and the next engine respawns the
// process.

requestLongerTimeout(120);

// Fixtures (TinyStories options, greedy samplers, storyteller prompt,
// stream drain) live in head.js.
const HWI_OPTIONS = engineId => tinyStoriesOptions({ engineId });

async function hwInferencePid() {
  const procInfo = await ChromeUtils.requestProcInfo();
  for (const child of procInfo.children) {
    if (
      child.type.startsWith("utility") &&
      child.utilityActors?.some(actor =>
        actor.actorName?.toLowerCase().includes("hwinference")
      )
    ) {
      return child.pid;
    }
  }
  return 0;
}

add_task(async function test_hwinference_crash_and_respawn() {
  const { cleanup } = await setup();
  try {
    const engine = await createEngine(HWI_OPTIONS("hwi-crash"));

    const pid = await hwInferencePid();
    Assert.greater(pid, 0, "The HWInference utility process is running");

    const processTools = Cc["@mozilla.org/processtools-service;1"].getService(
      Ci.nsIProcessToolsService
    );
    // Kill on the first streamed chunk, so the generation is provably
    // mid-flight when the process dies.
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
