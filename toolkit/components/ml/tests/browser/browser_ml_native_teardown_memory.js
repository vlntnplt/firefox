/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/// <reference path="head.js" />

/**
 * Creating, running and terminating an onnx-native engine must give the
 * inference process's heap back. Every cycle leaves the process with no
 * engine and no worker, so heap-allocated measured after teardown should be
 * flat across cycles; growth is memory the native session held past
 * ReleaseSession.
 */

const OPTIONS = {
  taskName: "text-classification",
  modelId: "acme/bert",
  dtype: "q8",
  backend: "onnx-native",
  engineId: "ml-native-teardown-memory",
  modelHubUrlTemplate: "{model}/resolve/{revision}",
  timeoutMS: -1,
};

const CYCLES = 6;
const RUNS_PER_CYCLE = 4;
const INPUT = "the quick brown fox jumps over the lazy dog ".repeat(40);
const ALLOWED_GROWTH_BYTES = 2 * 1024 * 1024;

/**
 * Collects the inference process's heap-allocated amount and the number of
 * live worker reports from a memory report pass.
 *
 * @returns {Promise<{process: ?string, heapAllocated: ?number, workers: number}>}
 */
function collectInferenceMemory() {
  const mgr = Cc["@mozilla.org/memory-reporter-manager;1"].getService(
    Ci.nsIMemoryReporterManager
  );
  return new Promise(resolve => {
    const result = {
      process: null,
      heapAllocated: null,
      jsHeap: 0,
      workers: 0,
    };
    mgr.getReportsExtended(
      (process, path, _kind, _units, amount) => {
        if (!process.startsWith("inference")) {
          return;
        }
        result.process = process;
        if (path == "heap-allocated") {
          result.heapAllocated = amount;
        }
        if (path.startsWith("explicit/js-non-window/")) {
          result.jsHeap += amount;
        }
        if (path.startsWith("explicit/workers/")) {
          result.workers++;
        }
      },
      null,
      () => resolve(result),
      null,
      /* anonymize */ false,
      /* minimizeMemoryUsage */ true,
      /* DMDDumpIdent */ ""
    );
  });
}

async function inferenceMemoryAfterTeardown() {
  let result;
  await TestUtils.waitForCondition(async () => {
    result = await collectInferenceMemory();
    return result.heapAllocated !== null && result.workers == 0;
  }, "Waiting for the inference process to have no worker left.");
  return result;
}

add_task(async function test_onnx_native_teardown_releases_heap() {
  const { cleanup } = await setup();
  if (Services.profiler.IsActive()) {
    info("Stopping the profiler: it retains JIT frame info per dead worker.");
    Services.profiler.StopProfiler();
  }

  const heapAfterTeardown = [];
  let process = null;
  for (let cycle = 0; cycle < CYCLES; cycle++) {
    const engine = await createEngine(OPTIONS);
    Assert.equal(engine.pipelineOptions.backend, "onnx-native");
    for (let run = 0; run < RUNS_PER_CYCLE; run++) {
      await engine.run({ args: [INPUT] });
    }
    await engine.terminate();

    const memory = await inferenceMemoryAfterTeardown();
    process ??= memory.process;
    Assert.equal(
      memory.process,
      process,
      "The same inference process serves every cycle."
    );
    heapAfterTeardown.push(memory.heapAllocated);
    info(
      `cycle ${cycle}: heap-allocated after teardown = ${(
        memory.heapAllocated /
        1024 /
        1024
      ).toFixed(2)} MiB, js-non-window = ${(
        memory.jsHeap /
        1024 /
        1024
      ).toFixed(2)} MiB`
    );
  }

  const growth = heapAfterTeardown.at(-1) - heapAfterTeardown[0];
  Assert.less(
    growth,
    ALLOWED_GROWTH_BYTES,
    `heap-allocated after teardown grew by ${(growth / 1024 / 1024).toFixed(
      2
    )} MiB over ${CYCLES - 1} create/run/terminate cycles.`
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
});
