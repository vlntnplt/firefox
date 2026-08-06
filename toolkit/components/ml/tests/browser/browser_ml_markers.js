/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/* global TextGenerator */

requestLongerTimeout(10);

const { ProfilerTestUtils } = ChromeUtils.importESModule(
  "resource://testing-common/ProfilerTestUtils.sys.mjs"
);

const TINYSTORIES_GREEDY_SAMPLERS = [
  { type: "top-k", topK: 1 },
  { type: "dist" },
];

const EXPECTED_MARKERS = [
  "MLProcessAcquire",
  "MLGeneratorCreate",
  "MLGeneratorRun",
  "MLChunkSend",
  "MLBackendInit",
  "MLModelPrefill",
  "MLModelDecode",
];

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

add_task(async function test_ml_pipeline_markers() {
  await ProfilerTestUtils.startProfiler({
    features: ["stackwalk", "js"],
    threads: ["GeckoMain", "TextGenerator", "llama.cpp"],
  });

  const generator = await createTinyStoriesGenerator();

  try {
    await generator.generate(
      {
        messages: [{ role: "user", content: "Once upon a time" }],
        maxTokens: 16,
        bufferLength: 4,
        samplers: TINYSTORIES_GREEDY_SAMPLERS,
      },
      () => {}
    );
  } finally {
    generator.terminate();
  }

  const profile = await ProfilerTestUtils.stopNowAndGetProfile();

  for (const type of EXPECTED_MARKERS) {
    Assert.greater(getPayloads(profile, type).length, 0, `${type} was emitted`);

    Assert.ok(
      ProfilerTestUtils.getSchema(profile, type),
      `${type} has a marker schema`
    );
  }

  const threadsFor = type => {
    const found = new Set();
    (function walk(proc) {
      for (const thread of proc.threads) {
        const { markers } = thread;
        for (const t of markers.data) {
          if (t[markers.schema.data]?.type === type) {
            found.add(thread.name);
          }
        }
      }
      proc.processes.forEach(walk);
    })(profile);
    return found;
  };
  Assert.ok(
    threadsFor("MLGeneratorRun").has("GeckoMain"),
    "The parent half of a generation is on GeckoMain"
  );
  Assert.ok(
    [...threadsFor("MLModelPrefill")].some(name =>
      name.startsWith("TextGenerator")
    ),
    "The model half is on the generator thread in HWInference"
  );

  const generate = getPayloads(profile, "MLGeneratorRun")[0];
  Assert.greater(generate.computeMs, 0, "The model compute is reported");
  Assert.greater(
    generate.prefillTokensPerSecond,
    0,
    "The prefill throughput is reported alongside the decode one, so the " +
      "label cannot imply the whole interval ran at decode speed"
  );
  Assert.greaterOrEqual(
    generate.overheadMs,
    0,
    "The overhead around the model is reported"
  );
  Assert.greaterOrEqual(
    generate.deliverMs,
    0,
    "What the caller's callback cost is reported"
  );
  Assert.greater(
    generate.chunkTokens,
    0,
    "The chunk size the ttfc number refers to is reported"
  );

  const prefill = getPayloads(profile, "MLModelPrefill")[0];
  Assert.greater(prefill.promptTokens, 0, "The prompt tokens are counted");
  const decode = getPayloads(profile, "MLModelDecode")[0];
  Assert.ok(
    decode,
    "The prompt phase boundary was observed, so a decode phase is reported"
  );

  Assert.equal(
    getPayloads(profile, "MLFailed").length,
    0,
    "A clean run reports no failed phase"
  );

  Assert.greater(
    getPayloads(profile, "MLChunkSend").length,
    0,
    "The child marked the chunks it streamed out"
  );
});

add_task(async function test_process_lifecycle_markers() {
  // The suite pins processTimeout to 0; reuse needs a real idle window.
  await SpecialPowers.pushPrefEnv({
    set: [["browser.ml.hwInference.processTimeout", 120000]],
  });

  await ProfilerTestUtils.startProfiler({
    features: ["stackwalk", "js"],
    threads: ["GeckoMain", "TextGenerator"],
  });

  const cold = await createTinyStoriesGenerator();
  cold.terminate();
  const warm = await createTinyStoriesGenerator();
  warm.terminate();

  const profile = await ProfilerTestUtils.stopNowAndGetProfile();

  const spawn = getPayloads(profile, "MLProcessSpawn");
  const reuse = getPayloads(profile, "MLProcessReuse");
  const idle = getPayloads(profile, "MLProcessIdle");
  const creates = getPayloads(profile, "MLGeneratorCreate");

  Assert.equal(spawn.length, 1, "The process was launched exactly once");
  Assert.equal(
    getPayloads(profile, "MLFailed").length,
    0,
    "A successful launch reports no failed phase"
  );
  Assert.greater(reuse.length, 0, "Reusing the warm process is marked");
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

  Assert.greater(idle.length, 0, "The idle window is marked");
  Assert.greater(
    reuse.length,
    0,
    "A reuse event marks the end of an idle window"
  );

  await SpecialPowers.popPrefEnv();

  // Cycle a generator under the restored timeout so the process retires now.
  const last = await createTinyStoriesGenerator();
  last.terminate();
});

add_task(async function test_teardown_mid_generation_is_not_a_failure() {
  await ProfilerTestUtils.startProfiler({
    features: ["stackwalk", "js"],
    threads: ["GeckoMain", "TextGenerator"],
  });

  const generator = await createTinyStoriesGenerator();
  // Terminate only once a chunk arrived, so it lands mid-generation.
  let sawChunk;
  const firstChunk = new Promise(resolve => {
    sawChunk = resolve;
  });
  const generation = generator.generate(
    {
      messages: [{ role: "user", content: "Once upon a time" }],
      maxTokens: 4096,
      bufferLength: 1,
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
    },
    () => sawChunk()
  );
  await firstChunk;
  generator.terminate();
  await Assert.rejects(
    generation,
    /terminated/,
    "The abandoned generation rejects"
  );

  const profile = await ProfilerTestUtils.stopNowAndGetProfile();

  Assert.equal(
    getPayloads(profile, "MLFailed").length,
    0,
    "A teardown the caller asked for is not marked as a failed phase"
  );
  const stages = getPayloads(profile, "MLCancel").map(p => p.stage);
  Assert.ok(stages.includes("teardown"), "The teardown is marked as a cancel");
});

add_task(async function test_engine_tier_markers() {
  const { cleanup } = await setup({
    prefs: [["browser.ml.llama.hwInference", true]],
  });

  await ProfilerTestUtils.startProfiler({
    features: ["stackwalk", "js"],
    threads: ["GeckoMain", "TextGenerator"],
  });

  try {
    const engine = await createEngine({
      backend: "llama.cpp",
      engineId: "marker-engine",
      taskName: "text-generation",
      modelId: "Mozilla/test-llama",
      modelFile: "TinyStories-656K.Q8_0.gguf",
      modelRevision: "main",
      numContext: 256,
      featureId: "link-preview",
    });
    for await (const chunk of engine.runWithGenerator({
      prompt: [{ role: "user", content: "Once upon a time" }],
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
      nPredict: 16,
    })) {
      void chunk;
    }
    await engine.terminate();

    const profile = await ProfilerTestUtils.stopNowAndGetProfile();

    const create = getPayloads(profile, "MLEngineCreate")[0];
    Assert.ok(create, "AIR: engine create was emitted");
    Assert.equal(
      create.featureId,
      "link-preview",
      "The workload names the feature that asked for it"
    );
    Assert.greaterOrEqual(create.fetchMs, 0, "The model fetch is broken out");
    Assert.greaterOrEqual(
      create.overheadMs,
      0,
      "The JS scaffolding is broken out"
    );

    const fetch = getPayloads(profile, "MLModelFetch")[0];
    Assert.ok(fetch, "AIR: model fetch was emitted");
    Assert.greater(fetch.bytes, 0, "The model size is reported");

    const run = getPayloads(profile, "MLEngineRun")[0];
    Assert.ok(run, "AIR: engine run was emitted");
    Assert.ok(run.streaming, "The run is marked as streamed");
    Assert.greater(run.outputTokens, 0, "The run generated tokens");
    Assert.greater(run.computeMs, 0, "The model compute is reported");
    Assert.greaterOrEqual(
      run.overheadMs,
      0,
      "Everything outside the model is reported as overhead"
    );
  } finally {
    await cleanup();
  }
});

add_task(async function test_abandoned_stream_is_marked_cancelled() {
  const { cleanup } = await setup({
    prefs: [["browser.ml.llama.hwInference", true]],
  });

  await ProfilerTestUtils.startProfiler({
    features: ["stackwalk", "js"],
    threads: ["GeckoMain", "TextGenerator"],
  });

  try {
    const engine = await createEngine({
      backend: "llama.cpp",
      engineId: "cancelled-engine",
      taskName: "text-generation",
      modelId: "Mozilla/test-llama",
      modelFile: "TinyStories-656K.Q8_0.gguf",
      modelRevision: "main",
      numContext: 256,
      featureId: "link-preview",
    });
    for await (const chunk of engine.runWithGenerator({
      prompt: [{ role: "user", content: "Once upon a time" }],
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
      nPredict: 4096,
    })) {
      void chunk;
      break;
    }
    await engine.terminate();

    const profile = await ProfilerTestUtils.stopNowAndGetProfile();

    const run = getPayloads(profile, "MLEngineRun")[0];
    Assert.ok(run, "An abandoned run is still marked");
    Assert.equal(
      run.reason,
      "cancelled",
      "The run the consumer walked away from is a cancel, not a failure"
    );
    Assert.greater(
      run.outputTokens,
      0,
      "A cancelled run reports the tokens it did produce, so it cannot " +
        "read as a run that generated nothing"
    );
    const stages = getPayloads(profile, "MLCancel").map(p => p.stage);
    Assert.ok(
      stages.includes("requested"),
      "Abandoning the loop asked the generator to stop"
    );
    const generatorRun = getPayloads(profile, "MLGeneratorRun")[0];
    Assert.ok(
      generatorRun,
      "The cancelled generation is still marked as a generation"
    );
    Assert.equal(
      generatorRun.reason,
      "cancelled",
      "A cancel is a finish reason on the run, not a run of its own"
    );
    Assert.greater(
      generatorRun.generatedTokens,
      0,
      "The cancelled generation reports the tokens it produced"
    );
    Assert.ok(
      !stages.includes("teardown"),
      "Waiting for the cancelled run to settle means the terminate no " +
        "longer races its reply"
    );
  } finally {
    await cleanup();
  }
});
