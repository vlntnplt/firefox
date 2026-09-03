/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// End-to-end performance coverage for semantic history search: history is
// ingested through the production indexing task, and queries are typed into
// the real urlbar, so the embedding engine is configured and driven entirely
// by production code. Reported spans are user-perceived: how long the
// day-one index build takes, and how long until a semantic result shows in
// the urlbar view.
//
// Parsed by the mozperftest static parser (vendored esprima, ES2017): no
// optional chaining, nullish coalescing, or object spread.

const { UrlbarTestUtils } = ChromeUtils.importESModule(
  "resource://testing-common/UrlbarTestUtils.sys.mjs"
);

const { getPlacesSemanticHistoryManager } = ChromeUtils.importESModule(
  "resource://gre/modules/PlacesSemanticHistoryManager.sys.mjs"
);
const { MLPerfTestUtils } = ChromeUtils.importESModule(
  "resource://testing-common/MLPerfTestUtils.sys.mjs"
);

const METRIC_PREFIX = "SEMANTICHISTORY";
const EMBEDDER_FEATURE_ID = "simple-text-embedder";
const INDEX_BUILD_LATENCY = "index-build-latency";
const SEMANTIC_RESULT_LATENCY = "semantic-result-latency";

// The query is semantically close to the dessert entries below but shares no
// word with any title or URL: a textual history match on the same URL would
// win muxer dedup and hide the semantic result this test waits for.
const SEARCH_QUERY = "baking sugary cakes";

const DESSERT_TITLES = [
  "Rich chocolate brownie recipe with fudgy center",
  "How to bake the perfect vanilla layer cake",
  "Classic apple pie crust techniques explained",
  "Easy homemade cookie dough for beginners",
  "Lemon meringue tart step by step guide",
  "Cinnamon rolls that rise overnight in the fridge",
  "Cheesecake baking mistakes and how to avoid them",
  "Glazed donuts made without a deep fryer",
  "Banana bread with caramelized walnuts",
  "French macarons troubleshooting common failures",
  "Pastry cream filling for eclairs and profiteroles",
  "Gingerbread house construction for the holidays",
];

const FILLER_TITLES = [
  "Quarterly earnings report analysis for investors",
  "Mountain bike suspension setup and tuning",
  "Understanding container orchestration architecture",
  "Migratory bird patterns in northern wetlands",
  "Electric vehicle charging standards compared",
  "Ancient roman aqueduct engineering principles",
  "Beginner guide to watercolor landscape painting",
  "Marathon training schedule for first timers",
  "Deep sea exploration submersible technology",
  "Municipal recycling program participation rates",
  "Vintage synthesizer restoration and repair",
  "Constitutional law landmark decisions reviewed",
  "Greenhouse tomato cultivation temperature control",
  "Professional camera lens optics fundamentals",
  "Volcanic activity monitoring instrumentation",
  "Medieval manuscript preservation techniques",
  "Urban traffic flow optimization modeling",
  "Antarctic research station daily operations",
];

// Declared names match emitted series as substrings, so each entry covers its
// lifecycle variants (-first-use, -cold, -warm).
const perfMetadata = {
  owner: "GenAI Team",
  name: "browser_urlbar_semantic_history_perf.js",
  description:
    "User-perceived latency and inference memory for semantic history search, driven through history ingestion and the production urlbar flow",
  options: {
    default: {
      perfherder: true,
      perfherder_metrics: [
        {
          name: "SEMANTICHISTORY-index-build-latency",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "SEMANTICHISTORY-semantic-result-latency",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "SEMANTICHISTORY-peak-memory",
          unit: "MiB",
          shouldAlert: false,
        },
        {
          name: "SEMANTICHISTORY-engine-creation-time",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "SEMANTICHISTORY-engine-run-time",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "SEMANTICHISTORY-memory-after-run",
          unit: "MiB",
          shouldAlert: false,
        },
      ],
      verbose: true,
      ml_services: true,
      manifest: "perftest.toml",
      manifest_flavor: "browser-chrome",
      try_platform: ["linux", "mac", "win"],
    },
  },
};

requestLongerTimeout(30);

function historyEntries() {
  const titles = DESSERT_TITLES.concat(FILLER_TITLES);
  return titles.map((title, i) => ({
    url: `https://site${i}.example.com/page/${i}`,
    title,
    visits: [{ date: new Date() }],
  }));
}

// The urlbar provider builds the manager singleton lazily with production
// parameters, whose indexing cadence waits on organic history churn. Create
// the same singleton first with an eager indexing schedule; the provider
// then reuses it. The embedding and search paths are untouched.
function createSemanticManager() {
  return getPlacesSemanticHistoryManager(
    {
      rowLimit: 10000,
      samplingAttrib: "frecency",
      changeThresholdCount: 0,
      distanceThreshold: Services.prefs.getFloatPref(
        "places.semanticHistory.distanceThreshold",
        0.6
      ),
      testFlag: true,
    },
    true
  );
}

add_task(async function test_semantic_history_perf() {
  UrlbarTestUtils.init(this);
  await SpecialPowers.pushPrefEnv({
    set: [["places.semanticHistory.featureGate", true]],
  });

  const semanticManager = createSemanticManager();
  Assert.ok(
    semanticManager.qualifiedForSemanticSearch,
    "Semantic history search is available on this machine"
  );

  const entries = historyEntries();
  // The indexing task embeds one chunk of pending entries per engine run.
  const indexBuildRuns = Math.ceil(
    entries.length /
      Services.prefs.getIntPref(
        "places.semanticHistory.defaultBatchChunksize",
        25
      )
  );

  // One day-one index build: from history landing to every entry embedded
  // and searchable.
  async function buildIndexOnce() {
    const start = performance.now();
    await PlacesUtils.history.insertMany(entries);
    // Frecency recalculation normally waits for idle; the semantic query
    // filters out rows whose frecency is still 0, so settle it now.
    await Cc["@mozilla.org/places/frecency-recalculator;1"]
      .getService(Ci.nsIObserver)
      .wrappedJSObject.recalculateAnyOutdatedFrecencies();
    const conn = await semanticManager.getConnection();
    await TestUtils.waitForCondition(
      async () => {
        const rows = await conn.execute(
          "SELECT COUNT(*) AS n FROM vec_history_mapping"
        );
        return rows[0].getResultByName("n") >= entries.length;
      },
      // Each indexing sweep is a DeferredTask that can wait up to two
      // minutes for main-thread idle before running anyway; several sweeps
      // may be needed.
      "Waiting for every history entry to be embedded",
      1000,
      600
    );
    const measurements = {};
    measurements[INDEX_BUILD_LATENCY] = performance.now() - start;
    return measurements;
  }

  // One urlbar search: from typing the query to the semantic result showing
  // in the view.
  async function searchOnce() {
    const start = performance.now();
    await UrlbarTestUtils.promiseAutocompleteResultPopup({
      window,
      value: SEARCH_QUERY,
      waitForFocus: SimpleTest.waitForFocus,
    });
    await TestUtils.waitForCondition(
      async () => {
        const count = UrlbarTestUtils.getResultCount(window);
        for (let i = 0; i < count; i++) {
          const details = await UrlbarTestUtils.getDetailsOfResultAt(window, i);
          if (
            details.result.providerName == "UrlbarProviderSemanticHistorySearch"
          ) {
            return true;
          }
        }
        return false;
      },
      "Waiting for a semantic history result in the urlbar view",
      100,
      600
    );
    const latency = performance.now() - start;
    await UrlbarTestUtils.promisePopupClose(window, () => gURLBar.blur());
    const measurements = {};
    measurements[SEMANTIC_RESULT_LATENCY] = latency;
    return measurements;
  }

  registerCleanupFunction(async () => {
    await PlacesUtils.history.clear();
    Services.prefs.clearUserPref("places.semanticHistory.initialized");
  });

  await MLPerfTestUtils.runPerfScenario({
    info,
    Assert,
    metricPrefix: METRIC_PREFIX,
    scenario: buildIndexOnce,
    engines: [{ featureId: EMBEDDER_FEATURE_ID, expectedRuns: indexBuildRuns }],
    coldIterations: 0,
    memoryIterations: 0,
  });

  // The eager indexing schedule kept polling via testFlag; silence it so the
  // no-op sweeps stop perturbing the query measurements.
  semanticManager.testFlag = false;

  // The index build above already landed the model, so the query flow has no
  // measurable first use; its shipping states are cold (first search of a
  // session, engine not yet running) and warm (engine resident, timeoutMS -1).
  await MLPerfTestUtils.runPerfScenario({
    info,
    Assert,
    metricPrefix: METRIC_PREFIX,
    scenario: searchOnce,
    engines: [{ featureId: EMBEDDER_FEATURE_ID }],
    measureFirstUse: false,
    coldIterations: 5,
    warmIterations: 5,
    memoryIterations: 3,
  });
});
