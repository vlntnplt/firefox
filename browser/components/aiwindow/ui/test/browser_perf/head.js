/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/* Unlike the functional suite's head.js, this installs no stub: the perf
 * suite measures the real paths, mocking only the MLPA network boundary
 * through withServer. */

const { AIWindowTestUtils } = ChromeUtils.importESModule(
  "resource://testing-common/AIWindowTestUtils.sys.mjs"
);

function openAIWindow(options) {
  return AIWindowTestUtils.openReadyAIWindow(SimpleTest, options);
}

function getAichatBrowser(browser) {
  return AIWindowTestUtils.getAichatBrowser(browser);
}

function selectExplicitSmartbarAction(browser, action) {
  return AIWindowTestUtils.selectExplicitSmartbarAction(
    SpecialPowers,
    browser,
    action
  );
}

function waitForSmartbarAction(browser, expectedAction) {
  return AIWindowTestUtils.waitForSmartbarAction(
    SpecialPowers,
    browser,
    expectedAction
  );
}

function withServer(serverOptions, task) {
  return AIWindowTestUtils.withServer(SpecialPowers, serverOptions, task);
}
