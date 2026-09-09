/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/* global TextGenerator */

requestLongerTimeout(10);

// The concurrency contracts of TextGenerator, all enforced in the parent: one
// generate() per generator at a time, clear() only between generations, and
// generators independent of each other. They share one HW_INFERENCE_BROWSER
// process, where each runs on a thread of its own, so two generations run in
// parallel and contend for the CPU: each backend has its own threadpool, so
// callers sizing them for the whole machine oversubscribe it.

// top-k:1 only filters; without a final `dist` sampler llama.cpp crashes.
const TINYSTORIES_GREEDY_SAMPLERS = [
  { type: "top-k", topK: 1 },
  { type: "dist" },
];

const TINYSTORIES_STORYTELLER_PROMPT = [
  { role: "system", content: "You are a friendly storyteller." },
  { role: "user", content: "Once upon a time there was a small mouse who" },
];

// TinyStories decodes thousands of tokens per second, so a generation that
// must still be running when a second request lands, or that another one must
// finish inside, has to be long: one-token chunks, no stop on end of text, and
// a context to fill.
const CONTEXT_SIZE = 4096;

const LONG_REQUEST = {
  messages: TINYSTORIES_STORYTELLER_PROMPT,
  maxTokens: CONTEXT_SIZE - 96,
  bufferLength: 1,
  samplers: TINYSTORIES_GREEDY_SAMPLERS,
  stopOnEndOfGenerationTokens: false,
};

const SHORT_REQUEST = {
  messages: TINYSTORIES_STORYTELLER_PROMPT,
  maxTokens: 16,
  bufferLength: 1,
  samplers: TINYSTORIES_GREEDY_SAMPLERS,
};

const HWI_OPTIONS = engineId => ({
  backend: "llama.cpp",
  engineId,
  taskName: "text-generation",
  modelId: "Mozilla/test-llama",
  modelFile: "TinyStories-656K.Q8_0.gguf",
  modelRevision: "main",
  numContext: 256,
});

async function createTinyStoriesGenerator() {
  const modelPath = getTestFilePath(
    "data/Mozilla/test-llama/main/TinyStories-656K.Q8_0.gguf"
  );
  const modelFile = await File.createFromFileName(modelPath);
  return TextGenerator.create(modelFile, { contextSize: CONTEXT_SIZE });
}

async function hwInferencePids() {
  const procInfo = await ChromeUtils.requestProcInfo();
  return procInfo.children
    .filter(
      child =>
        child.type.startsWith("utility") &&
        child.utilityActors.some(actor => actor.actorName === "hwInference")
    )
    .map(child => child.pid);
}

async function waitForProcessExit() {
  await TestUtils.waitForCondition(
    async () => !(await hwInferencePids()).length,
    "The HWInference process exits with its last generator"
  );
}

/**
 * Starts a generation and exposes its progress: `firstDelta` resolves once
 * the generation is known to be running, `done` is the generate() promise,
 * and `settled` flips as soon as it settles.
 *
 * @param {TextGenerator} generator
 * @param {object} request
 * @param {?Array} log - When given, every delta is appended as { run, text }.
 */
function startGeneration(generator, request, log = null) {
  const run = { streamed: "", settled: false };
  let onFirstDelta;
  run.firstDelta = new Promise(resolve => {
    onFirstDelta = resolve;
  });
  run.done = generator.generate(request, text => {
    run.streamed += text;
    log?.push({ run, text });
    onFirstDelta();
  });
  const settle = () => {
    run.settled = true;
  };
  run.done.then(settle, settle);
  return run;
}

function events(name) {
  return Glean.firefoxAiRuntime[name].testGetValue() ?? [];
}

add_task(async function test_generate_overlap_rejects() {
  const generator = await createTinyStoriesGenerator();
  try {
    const first = generator.generate(LONG_REQUEST);
    await Assert.rejects(
      generator.generate(SHORT_REQUEST),
      err => err.name === "InvalidStateError" && /in flight/.test(err.message),
      "A second generate() while one is pending rejects with InvalidStateError"
    );

    const result = await first;
    Assert.greater(
      result.content.length,
      0,
      "The first generation still resolves normally after the rejected overlap"
    );
    Assert.equal(
      result.reason,
      "length",
      "The first generation ran to its budget"
    );
  } finally {
    generator.terminate();
  }
});

add_task(async function test_clear_during_generate_rejects() {
  const generator = await createTinyStoriesGenerator();
  try {
    const run = startGeneration(generator, LONG_REQUEST);
    await run.firstDelta;
    Assert.throws(
      () => generator.clear(),
      err => err.name === "InvalidStateError",
      "clear() while a generate() is pending throws InvalidStateError"
    );

    const result = await run.done;
    Assert.notEqual(
      result.reason,
      "cancelled",
      "The rejected clear() did not disturb the generation"
    );
    Assert.equal(result.content, run.streamed, "The streamed deltas match");

    generator.clear();
    const afterClear = await generator.generate(SHORT_REQUEST);
    Assert.equal(
      afterClear.usage.promptTokens,
      result.usage.promptTokens,
      "clear() between generations empties the history: the next prompt is " +
        "the first one's size again"
    );
  } finally {
    generator.terminate();
  }
});

add_task(async function test_cancel_is_scoped_to_its_generation() {
  const generator = await createTinyStoriesGenerator();
  try {
    const run = startGeneration(generator, LONG_REQUEST);
    await run.firstDelta;
    generator.cancel();
    const cancelled = await run.done;
    Assert.equal(cancelled.reason, "cancelled", "The running generation ended");

    const next = await generator.generate(SHORT_REQUEST);
    Assert.notEqual(
      next.reason,
      "cancelled",
      "The earlier cancel() does not apply to the next generate()"
    );
    Assert.greater(next.usage.generatedTokens, 0, "The next generation ran");
  } finally {
    generator.terminate();
  }
});

add_task(async function test_concurrent_creates_share_one_process() {
  await waitForProcessExit();
  const [a, b] = await Promise.all([
    createTinyStoriesGenerator(),
    createTinyStoriesGenerator(),
  ]);
  try {
    const pids = await hwInferencePids();
    Assert.equal(pids.length, 1, "Racing creates share one process");

    a.terminate();
    const result = await b.generate(SHORT_REQUEST);
    Assert.greater(result.content.length, 0, "The survivor still generates");
    Assert.deepEqual(
      await hwInferencePids(),
      pids,
      "The surviving generator keeps the process"
    );
  } finally {
    a.terminate();
    b.terminate();
  }
  await waitForProcessExit();
});

add_task(async function test_sibling_terminated_mid_generation() {
  const doomed = await createTinyStoriesGenerator();
  const survivor = await createTinyStoriesGenerator();
  try {
    const run = startGeneration(survivor, LONG_REQUEST);
    await run.firstDelta;
    Assert.ok(!run.settled, "The survivor is still generating");

    doomed.terminate();

    const result = await run.done;
    Assert.notEqual(
      result.reason,
      "cancelled",
      "Terminating a sibling does not cancel this generation"
    );
    Assert.equal(result.content, run.streamed, "The streamed deltas match");
  } finally {
    doomed.terminate();
    survivor.terminate();
  }
});

add_task(async function test_generations_run_in_parallel() {
  const a = await createTinyStoriesGenerator();
  const b = await createTinyStoriesGenerator();
  try {
    const runA = startGeneration(a, LONG_REQUEST);
    await runA.firstDelta;
    Assert.ok(!runA.settled, "a is still generating");

    const resultB = await b.generate(SHORT_REQUEST);
    Assert.ok(
      !runA.settled,
      "A short generation on b completes while a is still streaming"
    );
    Assert.greater(resultB.usage.generatedTokens, 0, "b generated");

    const resultA = await runA.done;
    Assert.equal(resultA.reason, "length", "a ran to its budget");
    Assert.equal(resultA.content, runA.streamed, "a streamed its content");
  } finally {
    a.terminate();
    b.terminate();
  }
});

add_task(async function test_create_does_not_wait_for_running_generation() {
  const a = await createTinyStoriesGenerator();
  let b = null;
  try {
    const runA = startGeneration(a, LONG_REQUEST);
    await runA.firstDelta;
    Assert.ok(!runA.settled, "a is still generating");

    b = await createTinyStoriesGenerator();
    Assert.ok(
      !runA.settled,
      "A second generator loads while the first one is still streaming"
    );
    await runA.done;
  } finally {
    a.terminate();
    b?.terminate();
  }
});

add_task(async function test_engine_overlap_rejects() {
  const { cleanup } = await setup({
    prefs: [["browser.ml.llama.hwInference", true]],
  });
  try {
    const engine = await createEngine(HWI_OPTIONS("hwi-concurrency-overlap"));
    const request = {
      prompt: TINYSTORIES_STORYTELLER_PROMPT,
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
      nPredict: 16,
    };
    const failuresBefore = events("runInferenceFailure").length;
    const successesBefore = events("runInferenceSuccessFlow").length;

    const first = engine.run(request);
    await Assert.rejects(
      engine.run(request),
      /already in progress/,
      "A second run() while one is pending rejects"
    );
    await first;

    const failures = events("runInferenceFailure");
    Assert.equal(failures.length, failuresBefore + 1, "One run failure");
    Assert.equal(
      failures.at(-1).extra.error,
      "A generation is already in progress",
      "The run failure says the caller overlapped its runs"
    );
    Assert.equal(
      events("runInferenceSuccessFlow").length,
      successesBefore + 1,
      "The first run succeeded"
    );
    await engine.terminate();
  } finally {
    await cleanup();
  }
});
