/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

"use strict";

function test() {
  waitForExplicitFinish();

  // Types of processes to test, taken from GeckoProcessTypes.h
  // GPU process might not run depending on the platform, so we need it to be
  // the last one of the list to allow the remainingTests logic below to work
  // as expected.
  //
  // For UtilityProcess, allow constructing a string made of the process type
  // and the sandbox variant we want to test, e.g.,
  // utility:0 for GENERIC_UTILITY
  // utility:1 for AppleMedia/WMF on macOS/Windows
  var processTypes = ["tab", "socket", "rdd", "gmplugin", "utility:0", "gpu"];

  const platform = SpecialPowers.Services.appinfo.OS;
  if (platform === "WINNT" || platform === "Darwin") {
    processTypes.push("utility:1");
  }
  // HW_INFERENCE. Its index is whatever the build put before it: on macOS
  // that is GENERIC_UTILITY, APPLE_MEDIA, and PKCS11_MODULE, the last of
  // which only exists on Nightly. Naming a kind past COUNT is refused by the
  // parent, which would hang the run rather than fail it.
  if (platform === "Darwin") {
    processTypes.push(AppConstants.NIGHTLY_BUILD ? "utility:3" : "utility:2");
  }

  // Results are fire-and-forget: a child that never runs a probe reports
  // nothing, and nothing is not a failure. So for the HWInference set the
  // parent holds the list of ids that must arrive and asserts it at the end,
  // otherwise a child that bails early -- or a policy tight enough to break
  // the reporting channel itself -- reads as a clean run.
  const expectedHWInferenceTests =
    platform === "Darwin"
      ? [
          "hwinference_sandbox_check",
          "hwinference_read_allowed_system",
          "hwinference_read_allowed_devurandom",
          "hwinference_read_denied_home",
          "hwinference_write_denied_home",
          "hwinference_write_denied_tmp",
          "hwinference_connect_denied_inet",
        ]
      : [];
  const seenTests = new Set();

  // A callback called after each test-result.
  let sandboxTestResult = (subject, topic, data) => {
    let { testid, passed, message } = JSON.parse(data);
    seenTests.add(testid);
    ok(
      passed,
      "Test " + testid + (passed ? " passed: " : " failed: ") + message
    );
  };
  Services.obs.addObserver(sandboxTestResult, "sandbox-test-result");

  var remainingTests = processTypes.length;

  // A callback that is notified when a child process is done running tests.
  let sandboxTestDone = () => {
    remainingTests = remainingTests - 1;
    if (remainingTests == 0) {
      // Clean up test file
      if (homeTestFile.exists()) {
        ok(homeTestFile.isFile(), "homeTestFile should be a file");
        if (homeTestFile.isFile()) {
          homeTestFile.remove(false);
        }
      }

      for (const testid of expectedHWInferenceTests) {
        ok(seenTests.has(testid), `HWInference sandbox test ran: ${testid}`);
      }

      Services.obs.removeObserver(sandboxTestResult, "sandbox-test-result");
      Services.obs.removeObserver(sandboxTestDone, "sandbox-test-done");

      // Notify SandboxTest component that it should terminate the connection
      // with the child processes.
      comp.finishTests();
      // Notify mochitest that all process tests are complete.
      finish();
    }
  };
  Services.obs.addObserver(sandboxTestDone, "sandbox-test-done");

  var comp = Cc["@mozilla.org/sandbox/sandbox-test;1"].getService(
    Ci.mozISandboxTest
  );

  let homeTestFile;
  try {
    homeTestFile = Services.dirsvc.get("Home", Ci.nsIFile);
    homeTestFile.append(".mozilla_gpu_sandbox_read_test");
    if (!homeTestFile.exists()) {
      homeTestFile.create(Ci.nsIFile.NORMAL_FILE_TYPE, 0o600);
    }
  } catch (e) {
    ok(false, "Failed to create home test file: " + e);
  }

  comp.startTests(processTypes);
}
