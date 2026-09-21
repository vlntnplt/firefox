/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/// <reference path="head.js" />

const { sinon } = ChromeUtils.importESModule(
  "resource://testing-common/Sinon.sys.mjs"
);

const STUB_DIR =
  "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/";
const INIT_FAILURE_STUB_URL = `${STUB_DIR}ml_engine_init_failure_stub.worker.mjs`;
const LOAD_FAILURE_STUB_URL = `${STUB_DIR}ml_engine_load_failure_stub.worker.mjs`;

const OPTIONS = {
  taskName: "text-classification",
  modelId: "acme/bert",
  dtype: "q8",
  backend: "onnx-native",
  engineId: "init-failure-observability",
  modelHubUrlTemplate: "{model}/resolve/{revision}",
  timeoutMS: -1,
};

/**
 * Creates an engine on a stub worker and returns createEngine's rejection.
 *
 * @param {string} workerUrl
 * @returns {Promise<any>}
 */
async function createEngineOnStub(workerUrl) {
  const workerConfigStub = sinon
    .stub(MLEngineParent, "getWorkerConfig")
    .callsFake(() => ({ url: workerUrl, options: { type: "module" } }));
  try {
    return await createEngine(OPTIONS).then(
      () => null,
      error => error
    );
  } finally {
    workerConfigStub.restore();
  }
}

/**
 * Checks that a failed engine creation can be diagnosed: createEngine
 * rejected with something that has a name and a message containing
 * `expected`, and the engine kept that same error for whoever looks at it
 * afterwards.
 *
 * @param {any} rejection
 * @param {string} expected
 */
function assertDiagnosable(rejection, expected) {
  Assert.ok(rejection, "createEngine rejects when the worker cannot start.");
  Assert.ok(
    typeof rejection?.name == "string" && rejection.name,
    `The rejection has a name, got ${Object.prototype.toString.call(
      rejection
    )}.`
  );
  Assert.ok(
    String(rejection?.message).includes(expected),
    `The rejection carries the worker's message, got "${rejection?.message}".`
  );

  const engine = MLEngine.getInstance(OPTIONS.engineId);
  Assert.ok(engine, "The failed engine stays registered.");
  Assert.equal(engine.engineStatus, "error", "The engine is in error.");
  Assert.strictEqual(
    engine.error,
    rejection,
    "The engine records the error that failed it."
  );
}

/**
 * The worker starts but its backend throws while initializing.
 */
add_task(async function test_engine_init_failure_is_diagnosable() {
  const { cleanup } = await setup();
  try {
    assertDiagnosable(
      await createEngineOnStub(INIT_FAILURE_STUB_URL),
      "went away while loading"
    );
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

/**
 * The worker script itself fails to load, so the failure arrives through
 * the worker's error event rather than a request's rejection.
 */
add_task(async function test_engine_load_failure_is_diagnosable() {
  const { cleanup } = await setup();
  try {
    assertDiagnosable(
      await createEngineOnStub(LOAD_FAILURE_STUB_URL),
      "could not load"
    );
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});
