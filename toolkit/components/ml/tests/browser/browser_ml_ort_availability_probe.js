/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */
"use strict";
/// <reference path="head.js" />

/**
 * Build-configuration probe for the native onnxruntime.
 *
 * Unlike browser_ml_native.js and browser_ml_best_onnx.js -- which are
 * skip-if'd down to the platforms we already believe bundle the runtime --
 * this test runs EVERYWHERE. That is the point: CI covers far more build
 * configurations (asan, tsan, debug, x86, android, artifact builds, ...) than
 * our skip-if lists enumerate, and we currently have no signal at all about
 * whether libonnxruntime loads on them.
 *
 * It asserts that what the build claims (AppConstants.MOZ_ONNX_RUNTIME, driven
 * by the ONNX_RUNTIME define in toolkit/moz.configure) matches what actually
 * happens at runtime (InferenceSession::IsAvailable, i.e. a real dlopen). A
 * mismatch in either direction is a bug:
 *
 *   claims yes / loads no  -> we shipped a broken or unloadable library, and
 *                             every best-onnx consumer silently degrades to
 *                             wasm on this configuration.
 *   claims no  / loads yes -> packaging and configure disagree.
 *
 * It also emits a single machine-readable line so the whole CI matrix can be
 * scraped into a build-config table (see
 * toolkit/components/ml/tests/tools/ml_perf_report.py).
 */

const PROBE_MARKER = "ML_ORT_PROBE";

add_setup(async function () {
  await SpecialPowers.pushPrefEnv({
    set: [["browser.ml.enable", true]],
  });
  registerCleanupFunction(async () => {
    EngineProcess.resetNativeOnnxRuntimeAvailabilityForTests();
    await SpecialPowers.popPrefEnv();
  });
});

add_task(async function test_native_ort_availability_matches_build_config() {
  const expected = AppConstants.MOZ_ONNX_RUNTIME;
  const actual = await EngineProcess.requestIsNativeOnnxRuntimeAvailable();

  // One line, one JSON object, greppable across every CI job. Keep the shape
  // stable: the report tooling parses it.
  info(
    `${PROBE_MARKER} ${JSON.stringify({
      platform: AppConstants.platform,
      arch: Services.appinfo.XPCOMABI,
      isAndroid: AppConstants.platform === "android",
      debugBuild: AppConstants.DEBUG,
      buildApp: AppConstants.MOZ_BUILD_APP,
      bundled: expected,
      available: actual,
    })}`
  );

  Assert.equal(
    actual,
    expected,
    expected
      ? "The build bundles libonnxruntime and it loads at runtime"
      : "The build does not bundle libonnxruntime and it does not load"
  );
});

/**
 * The probe must not leave an inference process behind. It is called during
 * feature gating on paths that may never create an engine (see
 * EmbeddingsGenerator), so a leaked keep-alive would be a real regression.
 */
add_task(async function test_probe_does_not_leak_inference_process() {
  await EngineProcess.requestIsNativeOnnxRuntimeAvailable();

  await TestUtils.waitForCondition(
    () => EngineProcess.areAllEnginesTerminated(),
    "The availability probe did not keep an inference process alive"
  );

  Assert.ok(true, "No inference process outlived the probe");
});
