/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// What firefox.ai.runtime.engine_run records, checked against values the
// test controls: how long a run sleeps, what a scripted server streams and
// when, and how many tokens a generation is asked for.

requestLongerTimeout(4);

const ECHO_OPTIONS = {
  taskName: "moz-echo",
  timeoutMS: -1,
  modelId: "Mozilla/test",
  featureId: "test-feature",
  backend: "test-backend",
};

/**
 * @param {object} remoteClients - What setup() returns.
 * @returns {Promise<object>} An echo engine whose runtime is downloaded.
 */
async function createEchoEngine(remoteClients) {
  const engine = await createEngine(ECHO_OPTIONS);
  const warmUp = engine.run({ data: "warm up", sleepTime: 0 });
  await remoteClients["ml-onnx-runtime"].resolvePendingDownloads(1);
  await warmUp;
  return engine;
}

/**
 * @param {number} port - The port of a scripted completions server.
 * @param {object} [options] - More engine options.
 * @returns {Promise<object>} An engine on the openai backend.
 */
function createCloudEngine(port, options = {}) {
  return createEngine({
    taskName: "text-generation",
    featureId: "about-inference",
    backend: "openai",
    modelId: "scripted-model",
    apiKey: "test-key",
    baseURL: `http://localhost:${port}/v1`,
    maxRetries: 0,
    ...options,
  });
}

const CLOUD_REQUEST = { args: [{ role: "user", content: "Say hello" }] };
const CLOUD_STREAM_REQUEST = {
  ...CLOUD_REQUEST,
  streamOptions: { enabled: true },
};

/**
 * @param {AsyncGenerator} generator
 * @returns {Promise<{ text: string, chunks: number, promptTokens: number }>}
 *   What the stream generated, and how many prompt tokens it echoed back.
 */
async function drain(generator) {
  let text = "";
  let chunks = 0;
  let promptTokens = 0;
  for await (const chunk of generator) {
    if (chunk.isPrompt) {
      promptTokens += chunk.tokens?.length ?? 0;
    } else if (chunk.text) {
      text += chunk.text;
      chunks++;
    }
  }
  return { text, chunks, promptTokens };
}

/**
 * @returns {Record<string, string>[]} The extras of the engine_creation
 *   events recorded so far.
 */
function engineCreationEvents() {
  return (Glean.firefoxAiRuntime.engineCreation.testGetValue() ?? []).map(
    event => event.extra
  );
}

add_task(async function test_reused_engine_records_the_flow_of_each_request() {
  const { cleanup, remoteClients } = await setup();
  const creationsBefore = engineCreationEvents().length;
  const runsBefore = engineRunEvents().length;

  const first = await createEngine({ ...ECHO_OPTIONS, flowId: "flow-a" });
  const warmUp = first.run({ data: "first operation" });
  await remoteClients["ml-onnx-runtime"].resolvePendingDownloads(1);
  await warmUp;

  const second = await createEngine({ ...ECHO_OPTIONS, flowId: "flow-b" });
  Assert.equal(second, first, "The second operation is handed the live engine");
  await second.run({ data: "second operation" }, { flowId: "flow-b" });
  await drain(
    second.runWithGenerator({ data: "second operation" }, { flowId: "flow-b" })
  );
  await second.run({ data: "an operation that names no flow" });

  Assert.deepEqual(
    engineCreationEvents()
      .slice(creationsBefore)
      .map(creation => creation.flow_id),
    ["flow-a"],
    "Only the flow that created the engine has an engine_creation"
  );
  Assert.deepEqual(
    engineRunEvents()
      .slice(runsBefore)
      .map(run => run.flow_id),
    ["flow-a", "flow-b", "flow-b", "flow-a"],
    "A run is recorded under the flow it names, or else its engine's"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_run_records_the_model_remote_settings_picked() {
  const { cleanup, remoteClients } = await setup();
  const { modelId, ...options } = ECHO_OPTIONS;
  const engine = await createEngine({ ...options, engineId: "rs-model" });
  const warmUp = engine.run({ data: "warm up", sleepTime: 0 });
  await remoteClients["ml-onnx-runtime"].resolvePendingDownloads(1);
  const { output } = await warmUp;
  Assert.notEqual(output.modelId, modelId, "Remote Settings picked the model");

  const { run } = await recordEngineRunOf(() => engine.run({ data: "again" }));
  Assert.equal(
    run.model_id,
    output.modelId,
    "The run names the model the engine loaded"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_successful_run() {
  const { cleanup, remoteClients } = await setup();
  const engine = await createEchoEngine(remoteClients);

  const { run, result } = await recordEngineRunOf(() =>
    engine.run({ data: "echo me", sleepTime: 300 })
  );

  Assert.equal(result.output.echo, "echo me", "The run ran");
  Assert.equal(run.outcome, "success");
  Assert.equal(run.engine_id, "default-engine");
  Assert.equal(run.model_id, "Mozilla/test");
  Assert.equal(run.backend, "test-backend");
  Assert.equal(run.host_process, "inference");
  Assert.greaterOrEqual(
    Number(run.wall_milliseconds),
    300,
    "The wall time covers the time the run slept"
  );
  Assert.notEqual(run.cpu_milliseconds, undefined, "CPU time is measured");
  Assert.less(
    Number(run.cpu_milliseconds),
    Number(run.wall_milliseconds),
    "A run that sleeps uses less CPU time than wall time"
  );
  Assert.notEqual(run.memory_bytes, undefined, "Memory is measured");

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_failed_run() {
  const { cleanup, remoteClients } = await setup();
  const engine = await createEchoEngine(remoteClients);

  const { run, error } = await recordEngineRunOf(() => engine.run("throw"));

  Assert.ok(error, "The run threw");
  Assert.equal(run.outcome, "error");
  Assert.equal(run.error, "Error", "The error is recorded by name");
  Assert.ok(
    !Object.values(run).some(value => value.includes("intentionally")),
    "The error message is not recorded"
  );
  Assert.equal(run.engine_id, "default-engine");
  Assert.equal(run.backend, "test-backend");
  for (const key of ["cpu_milliseconds", "memory_bytes"]) {
    Assert.equal(run[key], undefined, `A run with no response has no ${key}`);
  }

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_run_aborted_by_its_caller() {
  const { cleanup, remoteClients } = await setup();
  const engine = await createEchoEngine(remoteClients);

  const controller = new AbortController();
  const { run, error, elapsed } = await recordEngineRunOf(() => {
    const pending = engine.run(
      { data: "slow", sleepTime: 3000 },
      { signal: controller.signal }
    );
    // eslint-disable-next-line mozilla/no-arbitrary-setTimeout
    setTimeout(() => controller.abort(), 100);
    return pending;
  });

  Assert.equal(error?.name, "AbortError", "The caller got the abort");
  Assert.equal(run.outcome, "aborted", "An abort is not a failure");
  Assert.less(elapsed, 3000, "The caller was released before the run ended");
  Assert.greaterOrEqual(
    Number(run.wall_milliseconds),
    100,
    "The wall time runs to the abort"
  );
  Assert.less(
    Number(run.wall_milliseconds),
    3000,
    "The wall time stops at the abort"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_run_ended_by_engine_teardown() {
  const { cleanup, remoteClients } = await setup();
  const engine = await createEchoEngine(remoteClients);

  const { run, error } = await recordEngineRunOf(() => {
    const pending = engine.run({ data: "slow", sleepTime: 3000 });
    // eslint-disable-next-line mozilla/no-arbitrary-setTimeout
    setTimeout(() => engine.terminate(), 100);
    return pending;
  });

  Assert.ok(error, "The run did not complete");
  Assert.equal(
    run.outcome,
    "aborted",
    "A run whose engine was torn down under it is aborted, not failed"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_run_on_an_engine_already_torn_down() {
  const { cleanup, remoteClients } = await setup();
  const engine = await createEchoEngine(remoteClients);
  await engine.terminate();

  const { run, error } = await recordEngineRunOf(() =>
    engine.run({ data: "anyone there?" })
  );
  Assert.ok(/Port does not exist/.test(error?.message), "The run is refused");
  Assert.equal(
    run.outcome,
    "aborted",
    "A run on an engine its holder tore down is recorded, as aborted"
  );

  const stream = await recordEngineRunOf(() =>
    drain(engine.runWithGenerator({ data: "anyone there?" }))
  );
  Assert.ok(
    /The port is null/.test(stream.error?.message),
    "The stream is refused"
  );
  Assert.equal(stream.run.outcome, "aborted", "and so is a stream on it");

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_run_records_the_purpose_it_names() {
  const { cleanup, remoteClients } = await setup();
  const engine = await createEchoEngine(remoteClients);

  const named = await recordEngineRunOf(() =>
    engine.run({ data: "named" }, { purpose: "link-preview" })
  );
  Assert.equal(named.run.outcome, "success");
  Assert.equal(named.run.purpose, "link-preview");

  const failed = await recordEngineRunOf(() =>
    engine.run("throw", { purpose: "link-preview" })
  );
  Assert.equal(failed.run.outcome, "error");
  Assert.equal(
    failed.run.purpose,
    "link-preview",
    "A failed run keeps its purpose"
  );

  const unnamed = await recordEngineRunOf(() =>
    engine.run({ data: "unnamed" })
  );
  Assert.equal(
    unnamed.run.purpose,
    ECHO_OPTIONS.featureId,
    "A run with no purpose, on an engine with none, falls back to the feature id"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_runs_sharing_an_engine_keep_their_own_purpose() {
  const { cleanup, remoteClients } = await setup();
  const engine = await createEchoEngine(remoteClients);
  const before = engineRunEvents().length;

  await Promise.all([
    engine.run(
      { data: "for semantic history", sleepTime: 200 },
      { purpose: "semantic-history:embed-query" }
    ),
    engine.run(
      { data: "for memories", sleepTime: 50 },
      { purpose: "memories:index-store" }
    ),
  ]);

  const runs = engineRunEvents().slice(before);
  Assert.equal(runs.length, 2, "Each run recorded its own event");
  const byPurpose = Object.fromEntries(runs.map(run => [run.purpose, run]));
  const history = byPurpose["semantic-history:embed-query"];
  const memories = byPurpose["memories:index-store"];
  Assert.ok(history && memories, "Each run is recorded under its own purpose");
  Assert.equal(
    memories.engine_id,
    history.engine_id,
    "Both ran on the same engine"
  );
  Assert.greaterOrEqual(
    Number(history.wall_milliseconds),
    200,
    "Each run has its own wall time"
  );
  Assert.greaterOrEqual(Number(memories.wall_milliseconds), 50);

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_shared_embedder_run_needs_a_purpose() {
  const { embeddingsGeneratorFactory } = ChromeUtils.importESModule(
    "chrome://global/content/ml/EmbeddingsGenerator.sys.mjs"
  );
  const embedder = embeddingsGeneratorFactory.forGeneral();
  const before = engineRunEvents().length;

  await Assert.rejects(
    embedder.embed("whose run is this"),
    /needs a purpose/,
    "The shared embedder refuses a run that names no purpose"
  );
  await Assert.rejects(
    embedder.embedMany(["whose", "runs"]),
    /needs a purpose/,
    "A batch needs one too"
  );
  Assert.equal(engineRunEvents().length, before, "No run, no event");
});

add_task(async function test_cloud_run_defaults_to_its_engine_purpose() {
  const { cleanup } = await setup();
  const { server, port } = startScriptedOpenAI({
    chunks: ["A title."],
    usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
  });
  const engine = await createCloudEngine(port, { purpose: "title-generation" });

  const created = Glean.firefoxAiRuntime.engineCreation.testGetValue().at(-1);
  Assert.equal(
    created.extra.purpose,
    "title-generation",
    "The engine's creation is recorded under its purpose"
  );

  const { run } = await recordEngineRunOf(() => engine.run(CLOUD_REQUEST));
  Assert.equal(
    run.purpose,
    "title-generation",
    "A run is recorded under the purpose its engine sends to MLPA"
  );

  const streamed = await recordEngineRunOf(() =>
    drain(engine.runWithGenerator(CLOUD_STREAM_REQUEST, { purpose: "chat" }))
  );
  Assert.equal(
    streamed.run.purpose,
    "chat",
    "A run that names its purpose is recorded under that one"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
  await stopMockOpenAI(server);
});

add_task(async function test_streamed_completion() {
  const { cleanup } = await setup();
  const chunks = ["One", " two", " three", " four"];
  const chunkDelayMs = 100;
  const { server, port } = startScriptedOpenAI({
    chunks,
    chunkDelayMs,
    usage: {
      prompt_tokens: 120,
      completion_tokens: 4,
      total_tokens: 124,
      prompt_tokens_details: { cached_tokens: 100 },
    },
  });
  const engine = await createCloudEngine(port);

  const { run, result } = await recordEngineRunOf(() =>
    drain(engine.runWithGenerator(CLOUD_STREAM_REQUEST))
  );

  Assert.equal(result.text, chunks.join(""), "The stream came through");
  Assert.equal(result.chunks, chunks.length);
  Assert.equal(run.outcome, "success");
  Assert.equal(run.finish_reason, "stop", "The service said the answer ended");
  Assert.equal(run.backend, "openai");
  Assert.equal(run.model_id, "scripted-model");
  Assert.equal(Number(run.input_tokens), 120, "Input tokens come from usage");
  Assert.equal(Number(run.cached_input_tokens), 100);
  Assert.equal(Number(run.output_tokens), 4, "Output tokens come from usage");
  Assert.greaterOrEqual(
    Number(run.wall_milliseconds),
    chunks.length * chunkDelayMs,
    "The wall time covers the whole stream"
  );
  Assert.greaterOrEqual(
    Number(run.time_to_first_chunk),
    chunkDelayMs,
    "The first chunk cannot arrive before the server sends it"
  );
  Assert.less(
    Number(run.time_to_first_chunk),
    2 * chunkDelayMs,
    "The time to first chunk stops at the first content chunk"
  );
  Assert.greaterOrEqual(
    Number(run.average_chunk_time),
    chunkDelayMs - 20,
    "Chunks are at least the server's pause apart"
  );
  Assert.less(
    Number(run.average_chunk_time),
    2 * chunkDelayMs,
    "The chunk cadence only counts chunks that carry content"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
  await stopMockOpenAI(server);
});

add_task(async function test_stream_cut_off_mid_answer() {
  const { cleanup } = await setup();
  const { server, port } = startScriptedOpenAI({
    chunks: ["One", " two", " three", " four"],
    chunkDelayMs: 20,
    cutOffAfter: 2,
  });
  const engine = await createCloudEngine(port);

  const { run, result, error } = await recordEngineRunOf(() =>
    drain(engine.runWithGenerator(CLOUD_STREAM_REQUEST))
  );

  Assert.equal(error, null, "The consumer is not told the answer is short");
  Assert.equal(result.text, "One two", "Half the answer arrived");
  Assert.equal(run.outcome, "success", "The run is recorded as it returned");
  Assert.equal(
    run.finish_reason,
    undefined,
    "No finish reason tells the cut-off answer from a complete one"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
  await stopMockOpenAI(server);
});

add_task(async function test_completion_without_stream() {
  const { cleanup } = await setup();
  const { server, port } = startScriptedOpenAI({
    chunks: ["A whole answer."],
    usage: { prompt_tokens: 50, completion_tokens: 7, total_tokens: 57 },
  });
  const engine = await createCloudEngine(port);

  const { run, result } = await recordEngineRunOf(() =>
    engine.run(CLOUD_REQUEST)
  );

  Assert.equal(result.finalOutput, "A whole answer.");
  Assert.equal(run.outcome, "success");
  Assert.equal(run.finish_reason, "stop", "The service said the answer ended");
  Assert.equal(Number(run.input_tokens), 50);
  Assert.equal(Number(run.output_tokens), 7);
  Assert.equal(
    run.cached_input_tokens,
    undefined,
    "Nothing was read from the prompt cache"
  );
  Assert.equal(
    run.time_to_first_chunk,
    undefined,
    "A run that does not stream has no first chunk"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
  await stopMockOpenAI(server);
});

add_task(async function test_completion_without_usage() {
  const { cleanup } = await setup();
  const { server, port } = startScriptedOpenAI({ chunks: ["Hi"] });
  const engine = await createCloudEngine(port);

  const { run } = await recordEngineRunOf(() =>
    drain(engine.runWithGenerator(CLOUD_STREAM_REQUEST))
  );

  Assert.equal(run.outcome, "success");
  for (const key of ["input_tokens", "cached_input_tokens", "output_tokens"]) {
    Assert.equal(run[key], undefined, `${key} is not made up`);
  }

  await EngineProcess.destroyMLEngine();
  await cleanup();
  await stopMockOpenAI(server);
});

add_task(async function test_completion_refused_by_the_server() {
  const { cleanup } = await setup();

  for (const status of [500, 401]) {
    const { server, port } = startScriptedOpenAI({ status });
    const engine = await createCloudEngine(port);

    for (const makeRun of [
      () => engine.run(CLOUD_REQUEST),
      () => drain(engine.runWithGenerator(CLOUD_STREAM_REQUEST)),
    ]) {
      const { run, error } = await recordEngineRunOf(makeRun);
      Assert.equal(error?.status, status, "The run threw the refusal");
      Assert.equal(run.outcome, "error");
      Assert.equal(
        run.http_status,
        String(status),
        "The status the service answered with tells refusals apart"
      );
      Assert.equal(run.backend, "openai");
      Assert.equal(run.output_tokens, undefined, "Nothing was generated");
      Assert.equal(run.finish_reason, undefined, "and nothing finished");
    }

    await EngineProcess.destroyMLEngine();
    await stopMockOpenAI(server);
  }

  await cleanup();
});

add_task(async function test_completion_from_a_server_out_of_reach() {
  const { cleanup } = await setup();
  const { server, port } = startScriptedOpenAI();
  await stopMockOpenAI(server);
  const engine = await createCloudEngine(port);

  for (const makeRun of [
    () => engine.run(CLOUD_REQUEST),
    () => drain(engine.runWithGenerator(CLOUD_STREAM_REQUEST)),
  ]) {
    const { run, error } = await recordEngineRunOf(makeRun);
    Assert.ok(error, "The run threw");
    Assert.equal(run.outcome, "error");
    Assert.equal(
      run.http_status,
      undefined,
      "A run that got no response has no HTTP status"
    );
  }

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

add_task(async function test_completion_rate_limited_by_the_server() {
  const { cleanup } = await setup();
  const { server, port } = startScriptedOpenAI({ status: 429 });
  const engine = await createCloudEngine(port);

  for (const makeRun of [
    () => engine.run(CLOUD_REQUEST),
    () => drain(engine.runWithGenerator(CLOUD_STREAM_REQUEST)),
  ]) {
    const { run, error } = await recordEngineRunOf(makeRun);
    Assert.equal(error?.status, 429, "The run threw the rate limit");
    Assert.equal(run.http_status, "429", "The status is recorded");
    Assert.equal(
      run.outcome,
      "limited",
      "A rate-limited run is recorded apart from failed ones"
    );
    Assert.equal(run.error, undefined, "It names no error");
  }

  await EngineProcess.destroyMLEngine();
  await cleanup();
  await stopMockOpenAI(server);
});

add_task(async function test_stream_abandoned_by_its_consumer() {
  const { cleanup } = await setup();
  const { server, port } = startScriptedOpenAI({
    chunks: ["One", " two", " three", " four", " five"],
    chunkDelayMs: 200,
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  });
  const engine = await createCloudEngine(port);

  const { run } = await recordEngineRunOf(async () => {
    for await (const chunk of engine.runWithGenerator(CLOUD_STREAM_REQUEST)) {
      Assert.equal(chunk.text, "One", "The consumer stops at the first chunk");
      break;
    }
  });

  Assert.equal(run.outcome, "aborted", "Giving a stream up is not a failure");
  Assert.equal(
    run.output_tokens,
    undefined,
    "The usage of the whole answer is not credited to the part that was read"
  );

  await EngineProcess.destroyMLEngine();
  await cleanup();
  await stopMockOpenAI(server);
});

add_task(async function test_generation_token_counts() {
  const nPredict = 8;
  await runOnBothInferenceProcesses(async () => {
    const { cleanup } = await setup();
    try {
      const engine = await createEngine(TINYSTORIES_ENGINE_OPTIONS);
      const request = {
        prompt: TINYSTORIES_STORYTELLER_PROMPT,
        samplers: TINYSTORIES_GREEDY_SAMPLERS,
        nPredict,
      };

      const whole = await recordEngineRunOf(() => engine.run(request));
      const { metrics } = whole.result;
      Assert.equal(metrics.outputTokens, nPredict, "The run generated");
      Assert.greater(metrics.inputTokens, 0, "The prompt was tokenized");
      Assert.equal(whole.run.outcome, "success");
      Assert.equal(whole.run.backend, "llama.cpp");
      Assert.equal(Number(whole.run.output_tokens), nPredict);
      Assert.equal(Number(whole.run.input_tokens), metrics.inputTokens);
      Assert.equal(
        whole.run.time_to_first_chunk,
        undefined,
        "A run that does not stream has no first chunk"
      );

      const streamed = await recordEngineRunOf(() =>
        drain(engine.runWithGenerator(request))
      );
      Assert.greater(streamed.result.chunks, 0, "The stream generated text");
      Assert.equal(streamed.run.outcome, "success");
      Assert.equal(
        Number(streamed.run.output_tokens),
        nPredict,
        "A streamed run counts the tokens it generated, not its prompt"
      );
      Assert.equal(
        Number(streamed.run.input_tokens),
        metrics.inputTokens,
        "The same prompt has the same input token count streamed"
      );
      Assert.notEqual(
        streamed.run.time_to_first_chunk,
        undefined,
        "A streamed run times its first chunk"
      );

      // Only the worker echoes the prompt; either way it is not output.
      const echoed = await recordEngineRunOf(() =>
        drain(engine.runWithGenerator({ ...request, skipPrompt: false }))
      );
      info(`The stream echoed ${echoed.result.promptTokens} prompt tokens`);
      Assert.equal(
        Number(echoed.run.output_tokens),
        nPredict,
        "A prompt streamed back does not count as generated tokens"
      );
      Assert.equal(Number(echoed.run.input_tokens), metrics.inputTokens);
    } finally {
      await EngineProcess.destroyMLEngine();
      await cleanup();
    }
  });
});
