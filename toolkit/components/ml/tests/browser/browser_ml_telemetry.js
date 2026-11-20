/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

requestLongerTimeout(2);

const RAW_PIPELINE_OPTIONS = {
  taskName: "moz-echo",
  timeoutMS: -1,
  modelId: "Mozilla/test",
  featureId: "test-feature",
  backend: "test-backend",
};

const { sinon } = ChromeUtils.importESModule(
  "resource://testing-common/Sinon.sys.mjs"
);

const { MLTelemetry } = ChromeUtils.importESModule(
  "chrome://global/content/ml/MLTelemetry.sys.mjs"
);

// Don't leak this env var into later tests.
registerCleanupFunction(() => {
  Services.env.set("MOZ_ALLOW_EXTERNAL_ML_HUB", "");
});

/**
 * @param {"success" | "error"} outcome
 * @returns {number} How many engine_run events ended that way.
 */
function countRuns(outcome) {
  return (Glean.firefoxAiRuntime.engineRun.testGetValue() ?? []).filter(
    event => event.extra.outcome === outcome
  ).length;
}

/**
 * @param {"success" | "error"} outcome
 * @returns {number} How many engine_creation events ended that way.
 */
function countCreations(outcome) {
  return (Glean.firefoxAiRuntime.engineCreation.testGetValue() ?? []).filter(
    ({ extra }) => extra.outcome === outcome
  ).length;
}

const TELEMETRY_TEST_STUB_URL =
  "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/ml_telemetry_stub.worker.mjs";

/**
 * Check that we record the engine creation and the inference run
 */
add_task(async function test_default_telemetry() {
  const { cleanup, remoteClients } = await setup();
  const engineCreationSuccessCount = countCreations("success");
  const runInferenceSuccessCount = countRuns("success");
  const runInferenceFailureCount = countRuns("error");
  const engineCreationFailureCount = countCreations("error");

  info("Get the engine");
  const engineInstance = await createEngine(RAW_PIPELINE_OPTIONS);

  info("Run the inference");
  const inferencePromise = engineInstance.run({
    data: "This gets echoed.",
  });

  info("Wait for the pending downloads.");
  await remoteClients["ml-onnx-runtime"].resolvePendingDownloads(1);

  const res = await inferencePromise;
  Assert.equal(
    res.output.echo,
    "This gets echoed.",
    "The text get echoed exercising the whole flow."
  );

  {
    info("Test the engine_run event");
    const value = Glean.firefoxAiRuntime.engineRun.testGetValue();
    Assert.ok(
      value && !!value.length,
      "At least one engine_run event was recorded"
    );
    const lastEngineRunEvent = value.at(-1);
    const { extra } = lastEngineRunEvent;
    const checkNumber = key => {
      const value = extra[key];
      Assert.notEqual(value, null, `${key} should be present`);
      const number = Number(value); // Quantities are stored as strings.
      Assert.ok(!Number.isNaN(number), `${key} should be a number`);
      Assert.greater(number, 0, `${key} should be greater than 0`);
    };
    checkNumber("cpu_milliseconds");
    checkNumber("wall_milliseconds");
    checkNumber("memory_bytes");
    checkNumber("system_memory_mb");

    Assert.equal(extra.outcome, "success");
    Assert.equal(extra.error, undefined, "A successful run names no error");
    Assert.equal(extra.engine_id, "default-engine");
    Assert.equal(extra.model_id, "Mozilla/test");
    Assert.equal(extra.backend, "test-backend");
    Assert.equal(extra.host_process, "inference");
  }

  Assert.equal(res.output.dtype, "q8", "The config was enriched by RS");
  ok(
    !EngineProcess.areAllEnginesTerminated(),
    "The engine process is still active."
  );

  Assert.equal(countCreations("success"), engineCreationSuccessCount + 1);

  Assert.equal(countCreations("success"), engineCreationSuccessCount + 1);

  Assert.equal(countRuns("success"), runInferenceSuccessCount + 1);

  Assert.equal(countRuns("error"), runInferenceFailureCount);

  Assert.equal(countCreations("error"), engineCreationFailureCount);

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

/**
 * Check that we record the engine creation and the inference failure
 */
add_task(async function test_ml_engine_run_failure() {
  const { cleanup, remoteClients } = await setup();
  const engineCreationSuccessCount = countCreations("success");
  const runInferenceSuccessCount = countRuns("success");
  const runInferenceFailureCount = countRuns("error");
  const engineCreationFailureCount = countCreations("error");

  info("Get the engine");
  const engineInstance = await createEngine(RAW_PIPELINE_OPTIONS);

  info("Run the inference with a throwing example.");
  const inferencePromise = engineInstance.run("throw");

  info("Wait for the pending downloads.");
  await remoteClients["ml-onnx-runtime"].resolvePendingDownloads(1);

  let error;
  try {
    await inferencePromise;
  } catch (e) {
    error = e;
  }
  is(
    error?.message,
    'Error: Received the message "throw", so intentionally throwing an error.',
    "The error is correctly surfaced."
  );

  Assert.equal(countCreations("success"), engineCreationSuccessCount + 1);

  Assert.equal(countRuns("success"), runInferenceSuccessCount);

  Assert.equal(countRuns("error"), runInferenceFailureCount + 1);
  const { extra } = Glean.firefoxAiRuntime.engineRun.testGetValue().at(-1);
  Assert.equal(extra.outcome, "error");
  Assert.equal(extra.error, "Error", "The error is recorded by name");
  Assert.equal(extra.engine_id, "default-engine");

  Assert.equal(countCreations("error"), engineCreationFailureCount);

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

/**
 * Check that we record the engine creation failure
 */
add_task(async function test_engine_creation_failure() {
  const { cleanup } = await setup();
  const engineCreationSuccessCount = countCreations("success");
  const engineCreationFailureCount = countCreations("error");
  const runInferenceSuccessCount = countRuns("success");
  const runInferenceFailureCount = countRuns("error");

  try {
    await createEngine({ taskName: "moz-echo", featureId: "I DONT EXIST" });
  } catch (e) {}

  Assert.equal(countCreations("success"), engineCreationSuccessCount);

  Assert.equal(countCreations("success"), engineCreationSuccessCount);

  Assert.equal(countRuns("success"), runInferenceSuccessCount);

  Assert.equal(countRuns("error"), runInferenceFailureCount);

  Assert.equal(countCreations("error"), engineCreationFailureCount + 1);

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

/**
 * A request refused before an engine is created is a failed creation too, and
 * leaves the lock of a creation under way alone.
 */
add_task(async function test_engine_creation_refused_for_memory() {
  const { cleanup } = await setup({
    prefs: [
      ["browser.ml.checkForMemory", true],
      ["browser.ml.minimumPhysicalMemory", 99999],
    ],
  });
  const creationsBefore = (
    Glean.firefoxAiRuntime.engineCreation.testGetValue() ?? []
  ).length;
  const creationUnderWay = Promise.withResolvers().promise;
  MLEngineParent.engineLocks.set("default-engine", creationUnderWay);

  try {
    await Assert.rejects(
      createEngine(RAW_PIPELINE_OPTIONS),
      error => error.name === "NotEnoughMemoryError",
      "The engine is refused on a machine with too little memory"
    );

    Assert.deepEqual(
      (Glean.firefoxAiRuntime.engineCreation.testGetValue() ?? [])
        .slice(creationsBefore)
        .map(({ extra }) => [extra.engine_id, extra.outcome, extra.error]),
      [["default-engine", "error", "NotEnoughMemoryError"]],
      "The refused creation is recorded once, with the name of its error"
    );
    Assert.equal(
      Glean.firefoxAiRuntime.engineCreation.testGetValue().at(-1).extra
        .failed_step,
      undefined,
      "A creation refused before it starts fails at no step"
    );
    Assert.equal(
      MLEngineParent.engineLocks.get("default-engine"),
      creationUnderWay,
      "The refused request does not release the lock of the other creation"
    );
  } finally {
    MLEngineParent.engineLocks.delete("default-engine");
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

/**
 * A failed creation says what it was doing when it failed.
 */
add_task(async function test_engine_creation_failed_step() {
  const modelOptions = {
    taskName: "real-llama-text-generation",
    featureId: "link-preview",
    backend: "llama.cpp",
    modelId: "acme/bert",
    modelHubUrlTemplate: "{model}/resolve/{revision}",
    modelRevision: "v0.1",
    modelHubRootUrl:
      "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
    modelFile: "onnx/config.json",
  };
  const cases = [
    {
      step: "model-download",
      why: "The hub does not have the model",
      options: {
        ...modelOptions,
        engineId: "given-up",
        modelId: "acme-not-found/bert",
      },
    },
    {
      step: "model-cache",
      why: "The model arrives and cannot be stored",
      options: { ...modelOptions, engineId: "given-up" },
      stub: sandbox =>
        sandbox
          .stub(TestIndexedDBCache.prototype, "put")
          .rejects(new DOMException("The disk is full", "QuotaExceededError")),
    },
    {
      step: "runtime-load",
      why: "The runtime cannot be fetched",
      options: { ...modelOptions, engineId: "no-runtime", backend: "onnx" },
      stub: () =>
        MLEngineParent.getWasmArrayBuffer.rejects(new Error("No runtime")),
    },
    {
      step: "model-load",
      why: "The backend does not start with the model it downloaded",
      options: { ...modelOptions, engineId: "fails-to-load" },
    },
  ];

  Services.env.set("MOZ_ALLOW_EXTERNAL_ML_HUB", "true");
  for (const { step, why, options, stub } of cases) {
    const sandbox = sinon.createSandbox();
    sandbox
      .stub(MLEngineParent, "getWasmArrayBuffer")
      .returns(new ArrayBuffer(16));
    sandbox
      .stub(MLEngineParent, "getWorkerConfig")
      .returns({ url: TELEMETRY_TEST_STUB_URL, options: { type: "module" } });
    stub?.(sandbox);
    await IndexedDBCache.init({ reset: true });
    await EngineProcess.destroyMLEngine();
    const creationsBefore = (
      Glean.firefoxAiRuntime.engineCreation.testGetValue() ?? []
    ).length;

    try {
      await Assert.rejects(createEngine(options), /./, why);
      Assert.deepEqual(
        (Glean.firefoxAiRuntime.engineCreation.testGetValue() ?? [])
          .slice(creationsBefore)
          .map(({ extra }) => [extra.outcome, extra.failed_step]),
        [["error", step]],
        `${why}: the creation fails at ${step}`
      );
    } finally {
      sandbox.restore();
      await EngineProcess.destroyMLEngine();
      await IndexedDBCache.init({ reset: true });
    }
  }
});

/**
 * A creation its caller gives up is aborted, not failed.
 */
add_task(async function test_engine_creation_given_up_by_its_caller() {
  Services.env.set("MOZ_ALLOW_EXTERNAL_ML_HUB", "true");
  const wasmBufferStub = sinon
    .stub(MLEngineParent, "getWasmArrayBuffer")
    .returns(new ArrayBuffer(16));
  const workerConfigStub = sinon
    .stub(MLEngineParent, "getWorkerConfig")
    .returns({ url: TELEMETRY_TEST_STUB_URL, options: { type: "module" } });
  await IndexedDBCache.init({ reset: true });
  await EngineProcess.destroyMLEngine();
  const creationsBefore = (
    Glean.firefoxAiRuntime.engineCreation.testGetValue() ?? []
  ).length;
  const controller = new AbortController();
  controller.abort();

  try {
    await Assert.rejects(
      createEngine(
        {
          engineId: "given-up",
          taskName: "real-llama-text-generation",
          featureId: "link-preview",
          backend: "llama.cpp",
          modelId: "acme/bert",
          modelHubUrlTemplate: "{model}/resolve/{revision}",
          modelRevision: "v0.1",
          modelHubRootUrl:
            "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
          modelFile: "onnx/config.json",
        },
        null,
        controller.signal
      ),
      /abort/i,
      "The creation does not survive its caller's abort"
    );

    Assert.deepEqual(
      (Glean.firefoxAiRuntime.engineCreation.testGetValue() ?? [])
        .slice(creationsBefore)
        .map(({ extra }) => [extra.engine_id, extra.outcome, extra.error]),
      [["given-up", "aborted", undefined]],
      "The creation is recorded as aborted, with no error"
    );
  } finally {
    await EngineProcess.destroyMLEngine();
    await IndexedDBCache.init({ reset: true });
    wasmBufferStub.restore();
    workerConfigStub.restore();
  }
});

/**
 * Check that model download telemetry is working as expected
 */
add_task(async function test_model_download_telemetry_success() {
  let initialModelDownloadsCount =
    Glean.firefoxAiRuntime.modelDownload.testGetValue()?.length || 0;
  // Allow any url
  Services.env.set("MOZ_ALLOW_EXTERNAL_ML_HUB", "true");

  let wasmBufferStub = sinon
    .stub(MLEngineParent, "getWasmArrayBuffer")
    .returns(new ArrayBuffer(16));

  let promiseStub = sinon
    .stub(MLEngineParent, "getWorkerConfig")
    .callsFake(function () {
      return { url: TELEMETRY_TEST_STUB_URL, options: { type: "module" } };
    });

  await IndexedDBCache.init({ reset: true });
  await EngineProcess.destroyMLEngine();

  await createEngine({
    engineId: "main",
    taskName: "real-llama-text-generation",
    featureId: "link-preview",
    backend: "llama.cpp",
    modelId: "acme/bert",
    modelHubUrlTemplate: "{model}/resolve/{revision}",
    modelRevision: "v0.1",
    modelHubRootUrl:
      "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
    modelFile: "onnx/config.json",
  });

  let observed = Glean.firefoxAiRuntime.modelDownload.testGetValue();

  Assert.equal(
    observed?.length || 0,
    initialModelDownloadsCount + 1,
    "The download recorded one event"
  );
  const download = observed.at(-1).extra;
  Assert.equal(download.outcome, "success");
  Assert.equal(
    download.error,
    undefined,
    "A download that succeeded names no error"
  );
  Assert.equal(Number(download.files), 2, "Both files were downloaded");
  Assert.greater(Number(download.bytes), 0, "Their size is recorded");
  Assert.equal(Number(download.failed_files), 0);
  Assert.greaterOrEqual(Number(download.duration), 0);
  Assert.deepEqual(
    [download.engine_id, download.model_id, download.model_revision],
    ["main", "acme/bert", "v0.1"],
    "The download names the engine it is for, and the model and revision"
  );
  Assert.equal(
    download.purpose,
    "link-preview",
    "An engine created with no purpose downloads under its feature id"
  );

  await EngineProcess.destroyMLEngine();
  await IndexedDBCache.init({ reset: true });

  wasmBufferStub.restore();
  promiseStub.restore();
});

/**
 * Check that the duration of a model download ends with its last file, however
 * long the backend then takes to load the model.
 */
add_task(async function test_model_download_duration_excludes_initialization() {
  Services.env.set("MOZ_ALLOW_EXTERNAL_ML_HUB", "true");
  const wasmBufferStub = sinon
    .stub(MLEngineParent, "getWasmArrayBuffer")
    .returns(new ArrayBuffer(16));
  const workerStub = sinon
    .stub(MLEngineParent, "getWorkerConfig")
    .returns({ url: TELEMETRY_TEST_STUB_URL, options: { type: "module" } });
  await IndexedDBCache.init({ reset: true });
  await EngineProcess.destroyMLEngine();

  const before =
    Glean.firefoxAiRuntime.modelDownload.testGetValue()?.length || 0;
  const start = ChromeUtils.now();
  await createEngine({
    engineId: "slow-init",
    taskName: "real-llama-text-generation",
    featureId: "link-preview",
    backend: "llama.cpp",
    modelId: "acme/bert",
    modelHubUrlTemplate: "{model}/resolve/{revision}",
    modelRevision: "v0.1",
    modelHubRootUrl:
      "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
    modelFile: "onnx/config.json",
  });
  const creation = ChromeUtils.now() - start;
  Assert.greaterOrEqual(creation, 500, "The backend took its time to load");

  const events = Glean.firefoxAiRuntime.modelDownload
    .testGetValue()
    .slice(before)
    .map(event => event.extra);
  Assert.equal(events.length, 1, "The download was recorded");
  const [end] = events;
  Assert.greater(Number(end.files), 0, "It downloaded files");
  Assert.less(
    Number(end.duration),
    500,
    "The download duration leaves the time the backend took to load out"
  );
  Assert.lessOrEqual(
    Number(end.duration),
    Math.ceil(creation),
    "The download duration fits in the time creation took"
  );

  await EngineProcess.destroyMLEngine();
  await IndexedDBCache.init({ reset: true });
  wasmBufferStub.restore();
  workerStub.restore();
});

/**
 * Check that model download telemetry is working as expected
 */
add_task(async function test_model_download_telemetry_fail() {
  let initialModelDownloadsCount =
    Glean.firefoxAiRuntime.modelDownload.testGetValue()?.length || 0;
  // Allow any url
  Services.env.set("MOZ_ALLOW_EXTERNAL_ML_HUB", "true");

  let wasmBufferStub = sinon
    .stub(MLEngineParent, "getWasmArrayBuffer")
    .returns(new ArrayBuffer(16));

  let promiseStub = sinon
    .stub(MLEngineParent, "getWorkerConfig")
    .callsFake(function () {
      return { url: TELEMETRY_TEST_STUB_URL, options: { type: "module" } };
    });

  await IndexedDBCache.init({ reset: true });
  await EngineProcess.destroyMLEngine();
  await createEngine({
    engineId: "main",
    taskName: "real-llama-text-generation",
    featureId: "link-preview",
    backend: "llama.cpp",
    modelId: "acme-not-found/bert",
    modelHubUrlTemplate: "{model}/resolve/{revision}",
    modelRevision: "v0.1",
    modelHubRootUrl:
      "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
    modelFile: "onnx/config.json",
  }).catch(() => {});

  let observed = Glean.firefoxAiRuntime.modelDownload.testGetValue();

  Assert.equal(
    observed?.length || 0,
    initialModelDownloadsCount + 1,
    "The download recorded one event"
  );
  const download = observed.at(-1).extra;
  Assert.equal(download.outcome, "error");
  Assert.ok(download.error, "A failed download names its error");
  Assert.equal(Number(download.files), 0);
  Assert.equal(Number(download.failed_files), 2, "Both files failed");

  await EngineProcess.destroyMLEngine();
  await IndexedDBCache.init({ reset: true });

  wasmBufferStub.restore();
  promiseStub.restore();
});

/**
 * Check that model download telemetry is working as expected
 */
add_task(async function test_model_download_telemetry_mixed() {
  let initialModelDownloadsCount =
    Glean.firefoxAiRuntime.modelDownload.testGetValue()?.length || 0;
  // Allow any url
  Services.env.set("MOZ_ALLOW_EXTERNAL_ML_HUB", "true");

  let wasmBufferStub = sinon
    .stub(MLEngineParent, "getWasmArrayBuffer")
    .returns(new ArrayBuffer(16));

  let promiseStub = sinon
    .stub(MLEngineParent, "getWorkerConfig")
    .callsFake(function () {
      return { url: TELEMETRY_TEST_STUB_URL, options: { type: "module" } };
    });

  await createEngine({
    engineId: "main",
    taskName: "real-llama-text-generation",
    featureId: "link-preview",
    backend: "llama.cpp",
    modelId: "acme/bert",
    modelHubUrlTemplate: "{model}/resolve/{revision}",
    modelRevision: "v0.4",
    modelHubRootUrl:
      "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
    modelFile: "onnx/config.json",
  }).catch(() => {});

  let observed = Glean.firefoxAiRuntime.modelDownload.testGetValue();

  Assert.equal(
    observed?.length || 0,
    initialModelDownloadsCount + 1,
    "The download recorded one event"
  );
  const download = observed.at(-1).extra;
  Assert.equal(
    download.outcome,
    "success",
    "A download whose last file arrived is recorded as a success"
  );
  Assert.equal(Number(download.files), 1);
  Assert.equal(Number(download.failed_files), 1, "The failed file is counted");
  await EngineProcess.destroyMLEngine();
  await IndexedDBCache.init({ reset: true });

  wasmBufferStub.restore();
  promiseStub.restore();
});

/**
 * Check that downloads started through the model hub service, which native
 * callers use, are attributed to the task that asked for them.
 */
add_task(async function test_model_hub_service_download_attribution() {
  Services.env.set("MOZ_ALLOW_EXTERNAL_ML_HUB", "true");
  await SpecialPowers.pushPrefEnv({
    set: [
      [
        "browser.ml.modelHubRootUrl",
        "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
      ],
      ["browser.ml.modelHubUrlTemplate", "{model}/resolve/{revision}"],
    ],
  });
  await IndexedDBCache.init({ reset: true });

  let initialModelDownloadsCount =
    Glean.firefoxAiRuntime.modelDownload.testGetValue()?.length || 0;

  let hub = Cc["@mozilla.org/ml-modelhub;1"].getService(Ci.nsIMLModelHub);
  await new Promise((resolve, reject) => {
    hub.downloadModel(
      "main",
      "speech-recognition",
      "acme/bert",
      "v0.1",
      ["onnx/config.json"],
      "progress-token",
      null,
      {
        QueryInterface: ChromeUtils.generateQI([
          "nsIMLModelDownloadCompletionCallback",
        ]),
        onSuccess: resolve,
        onError: reject,
      }
    );
  });

  let observed = Glean.firefoxAiRuntime.modelDownload
    .testGetValue()
    .slice(initialModelDownloadsCount);

  Assert.greater(observed.length, 0, "The download was measured");
  Assert.deepEqual(
    observed.map(obj => obj.extra.purpose),
    ["speech-recognition"],
    "The download is attributed to the task that asked for it"
  );

  await IndexedDBCache.init({ reset: true });
  await SpecialPowers.popPrefEnv();
});

/**
 * Check that a telemetry instance keeps the flow id it is given, and makes
 * one up otherwise.
 */
add_task(async function test_ml_telemetry_flow_id() {
  const generated = new MLTelemetry({ featureId: "test-feature" }).flowId;
  Assert.ok(generated, "A flow id is generated when none is given");
  Assert.notEqual(
    new MLTelemetry({ featureId: "test-feature" }).flowId,
    generated,
    "Each instance generates its own"
  );
  Assert.equal(new MLTelemetry({ flowId: "given-flow" }).flowId, "given-flow");
});

add_task(async function test_run_with_generator_telemetry() {
  const { cleanup } = await setup();
  const { server: mockServer, port } = startMockOpenAI({
    echo: "Streaming response.",
  });

  info("Create the engine with OpenAI backend");
  const engineInstance = await createEngine({
    taskName: "text-generation",
    featureId: "about-inference",
    backend: "openai",
    modelId: "test-model",
    apiKey: "test-key",
    baseURL: `http://localhost:${port}/v1`,
  });

  info("Call runWithGenerator");
  const generator = engineInstance.runWithGenerator({
    args: [{ role: "user", content: "test streaming" }],
    streamOptions: { enabled: true },
  });

  info("Manually iterate to capture both chunks and return value");
  let iterResult;
  while (true) {
    iterResult = await generator.next();
    if (iterResult.done) {
      break;
    }
  }

  {
    info("Test the engine_run event for runWithGenerator");
    const value = Glean.firefoxAiRuntime.engineRun.testGetValue();
    Assert.ok(
      value && !!value.length,
      "At least one engine_run event was recorded"
    );
    const lastEngineRunEvent = value.at(-1);
    const { extra } = lastEngineRunEvent;
    info("Recorded Glean engine_run event: " + JSON.stringify(extra, null, 2));
    const checkNumber = key => {
      const value = extra[key];
      Assert.notEqual(value, null, `${key} should be present`);
      const number = Number(value);
      Assert.ok(!Number.isNaN(number), `${key} should be a number`);
      Assert.greater(number, 0, `${key} should be greater than 0`);
    };
    checkNumber("cpu_milliseconds");
    checkNumber("wall_milliseconds");
    checkNumber("memory_bytes");
    checkNumber("system_memory_mb");
    checkNumber("time_to_first_chunk");
    Assert.ok(!extra.output_tokens, "Token count is not implemented yet.");

    Assert.notEqual(
      extra.average_chunk_time,
      null,
      "average_chunk_time is recorded when more than one chunk streams"
    );
    Assert.greaterOrEqual(
      Number(extra.average_chunk_time),
      0,
      "average_chunk_time should be a non-negative number"
    );

    Assert.equal(extra.backend, "openai");
  }

  await EngineProcess.destroyMLEngine();
  await cleanup();
  await stopMockOpenAI(mockServer);
});
