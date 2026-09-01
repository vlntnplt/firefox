/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// End-to-end performance coverage for ML Form Autofill: a focus in the
// production address form drives the real FormAutofill actor, so the engines
// are configured by production code and the reported span is user-perceived.
// Both shipping classifier architectures are measured.
//
// Parsed by the mozperftest static parser (vendored esprima, ES2017): no
// optional chaining, nullish coalescing, or object spread.

const { FormAutofillParent } = ChromeUtils.importESModule(
  "resource://autofill/FormAutofillParent.sys.mjs"
);
const { RemoteSettings } = ChromeUtils.importESModule(
  "resource://services-settings/remote-settings.sys.mjs"
);
const { MLPerfTestUtils } = ChromeUtils.importESModule(
  "resource://testing-common/MLPerfTestUtils.sys.mjs"
);

const ADDRESS_FORM_URL =
  "https://example.org/browser/browser/extensions/formautofill/test/fixtures/without_autocomplete_address_basic.html";

const FOCUS_TO_IDENTIFICATION = "focus-to-identification-time";

// The single-engine classifier resolves entirely from production code and
// hub defaults. The two-head engines are configured by Remote Settings and
// nowhere in code, so the test seeds the two shipping records at the
// revisions the in-tree fetch pins (create_local_hub.py serves the same
// layout locally that CI fetches).
const TWO_HEAD_INFERENCE_RECORDS = [
  {
    id: "formfill-encoder-local",
    last_modified: 1,
    featureId: "formfill-encoder",
    taskName: "feature-extraction",
    modelId: "mozilla/form-autofill-embed",
    modelRevision: "v0.1.0",
    tokenizerId: "mozilla/form-autofill-embed",
    tokenizerRevision: "v0.1.0",
    processorId: "mozilla/form-autofill-embed",
    processorRevision: "v0.1.0",
  },
  {
    id: "formfill-head-local",
    last_modified: 1,
    featureId: "formfill-head",
    taskName: "moz-formfill-head",
    modelId: "mozilla/form-autofill-head",
    modelRevision: "v0.1.0",
    tokenizerId: "mozilla/form-autofill-head",
    tokenizerRevision: "v0.1.0",
    processorId: "mozilla/form-autofill-head",
    processorRevision: "v0.1.0",
    dtype: "fp32",
  },
];

const perfMetadata = {
  owner: "Form Autofill Team",
  name: "browser_formautofill_ml_perf.js",
  description:
    "User-perceived latency and inference memory for ML Form Autofill field identification, driven through the production focus flow",
  options: {
    default: {
      perfherder: true,
      perfherder_metrics: [
        {
          name: "FORM-AUTOFILL-single-engine-focus-to-identification-time",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-single-engine-engine-creation",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-single-engine-inference-time",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-single-engine-peak-memory",
          unit: "MiB",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-double-engine-focus-to-identification-time",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-double-engine-encoder-engine-creation",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-double-engine-head-engine-creation",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-double-engine-encoder-inference-time",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-double-engine-head-inference-time",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "FORM-AUTOFILL-double-engine-peak-memory",
          unit: "MiB",
          shouldAlert: false,
        },
      ],
      verbose: true,
      manifest: "perftest.toml",
      manifest_flavor: "browser-chrome",
      try_platform: ["linux", "mac", "win"],
    },
  },
};

requestLongerTimeout(30);

function observeFieldsIdentified() {
  const deferred = Promise.withResolvers();
  const observer = {
    fieldsIdentified(_data, chromeWindow) {
      if (chromeWindow === window) {
        deferred.resolve(performance.now());
      }
    },
  };
  FormAutofillParent.addMessageObserver(observer);
  return {
    promise: deferred.promise,
    stop() {
      FormAutofillParent.removeMessageObserver(observer);
    },
  };
}

// One full user interaction: focus a field of the production address form
// and wait for FormAutofill to finish identifying the form's fields.
async function identifyFieldsOnce() {
  const tab = await BrowserTestUtils.openNewForegroundTab(
    gBrowser,
    ADDRESS_FORM_URL
  );
  const identified = observeFieldsIdentified();

  try {
    const start = performance.now();
    await SpecialPowers.spawn(tab.linkedBrowser, [], () => {
      content.document.querySelector("#given-name").focus();
    });
    const end = await identified.promise;

    const actor =
      tab.linkedBrowser.browsingContext.currentWindowGlobal.getActor(
        "FormAutofill"
      );
    const fields = Array.from(actor.sectionsByRootId.values())
      .flat()
      .flatMap(function (section) {
        return section.fieldDetails;
      });
    Assert.ok(
      fields.some(function (field) {
        return field.reason === "ml";
      }),
      "The production flow applied an ML classification"
    );

    const spans = {};
    spans[FOCUS_TO_IDENTIFICATION] = end - start;
    return spans;
  } finally {
    identified.stop();
    await BrowserTestUtils.removeTab(tab);
  }
}

add_task(async function test_formautofill_ml_perf() {
  const client = RemoteSettings("ml-inference-options");
  await client.db.importChanges({}, Date.now(), TWO_HEAD_INFERENCE_RECORDS, {
    clear: true,
  });
  registerCleanupFunction(function () {
    return client.db.clear();
  });

  await SpecialPowers.pushPrefEnv({
    set: [
      ["extensions.formautofill.addresses.enabled", true],
      ["extensions.formautofill.skipProgrammaticCheckForTests", true],
      ["extensions.formautofill.useml", true],
      // ML eligibility is still gated on native availability for a WASM run.
      ["extensions.formautofill.useml.nativeOnnxAvailable", true],
      // Exercise field identification without Form Autofill's first-use
      // early return.
      ["extensions.formautofill.useml.successful", true],
    ],
  });

  await SpecialPowers.pushPrefEnv({
    set: [["extensions.formautofill.useml.twoHead", false]],
  });
  await MLPerfTestUtils.runPerfScenario({
    info,
    Assert,
    metricPrefix: "FORM-AUTOFILL-single-engine",
    featureId: "formfill-classification",
    scenario: identifyFieldsOnce,
    warmIterations: 5,
  });
  await SpecialPowers.popPrefEnv();

  await SpecialPowers.pushPrefEnv({
    set: [["extensions.formautofill.useml.twoHead", true]],
  });
  await MLPerfTestUtils.runPerfScenario({
    info,
    Assert,
    metricPrefix: "FORM-AUTOFILL-double-engine",
    featureId: ["formfill-encoder", "formfill-head"],
    featureTags: {
      "formfill-encoder": "encoder-",
      "formfill-head": "head-",
    },
    scenario: identifyFieldsOnce,
    warmIterations: 5,
  });
  await SpecialPowers.popPrefEnv();
});
