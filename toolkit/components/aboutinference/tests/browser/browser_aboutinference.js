requestLongerTimeout(2);

/**
 * Checks that the page renders without issue, and that the expected elements
 * are there.
 */
add_task(async function test_about_inference_enabled() {
  await openAboutInference({
    runInPage: async ({ selectors }) => {
      const { document, window } = content;

      function checkElementIsVisible(expectVisible, name) {
        const expected = expectVisible ? "visible" : "hidden";
        const element = document.querySelector(selectors[name]);
        ok(Boolean(element), `Element ${name} was found.`);
        const { visibility } = window.getComputedStyle(element);
        is(
          visibility,
          expected,
          `Element ${name} was not ${expected} but should be.`
        );
      }
      checkElementIsVisible(true, "pageHeader");
      const element = document.querySelector(selectors.warning);
      const { display } = window.getComputedStyle(element);
      is(display, "none", "The warning should be hidden");
    },
  });
});

/**
 * Checks that the page renders with a warning when ml is disabled.
 */
add_task(async function test_about_inference_disabled() {
  await openAboutInference({
    prefs: [["browser.ml.enable", false]],
    runInPage: async ({ selectors }) => {
      const { document, window } = content;
      const element = document.querySelector(selectors.warning);
      const { display } = window.getComputedStyle(element);
      is(display, "block", "The warning should be visible");
      is(
        element.getAttribute("data-l10n-id"),
        "about-inference-ml-disabled",
        "The warning asks to enable local inference"
      );
      document.getElementById("warning.enable").click();
      await ContentTaskUtils.waitForCondition(
        () => window.getComputedStyle(element).display === "none",
        "Enabling local inference hides the warning"
      );
    },
  });
});
/**
 * Checks that the inference process is shown on the page
 */
add_task(async function test_about_inference_process() {
  await openAboutInference({
    runInference: true,
    runInPage: async ({ selectors }) => {
      function waitForInnerHTML(selector, substring, interval = 100) {
        return new Promise((resolve, reject) => {
          const { document } = content;
          const element = document.querySelector(selector);
          if (!element) {
            reject(new Error(`No element found with selector: ${selector}`));
            return;
          }
          const checkInnerHTML = () => {
            info(`Checking innerHTML of element with selector: ${selector}`);
            if (element.innerHTML.includes(substring)) {
              info(
                `Substring "${substring}" found in element with selector: ${selector}`
              );
              resolve();
            } else {
              // eslint-disable-next-line mozilla/no-arbitrary-setTimeout
              content.setTimeout(checkInnerHTML, interval);
            }
          };
          checkInnerHTML();
        });
      }

      // When the process is shown, we display its memory size in MB
      await waitForInnerHTML(selectors.processes, "MB");
    },
  });
});

/**
 * Lists the inference devices through the devices panel.
 */
add_task(async function test_about_inference_devices() {
  const { cleanup } = await setup({
    prefs: [["browser.ml.llama.hwInference", true]],
  });
  const tab = await BrowserTestUtils.openNewForegroundTab(
    gBrowser,
    "about:inference",
    true
  );
  try {
    await SpecialPowers.spawn(tab.linkedBrowser, [], async () => {
      const byId = id => content.document.getElementById(id);
      byId("subtab-devices").click();
      Assert.ok(byId("inference-pad").hidden, "The pad gives way");
      await ContentTaskUtils.waitForCondition(
        () => byId("devices.table").querySelector("td"),
        "The devices panel lists the inference devices",
        100,
        300
      );
      const kinds = [...byId("devices.table").querySelectorAll("tbody tr")].map(
        tr => tr.children[2].textContent
      );
      Assert.ok(kinds.includes("cpu"), "The CPU is listed");
    });
  } finally {
    BrowserTestUtils.removeTab(tab);
    await cleanup();
  }
});

/**
 * Runs a generation through the text generation panel.
 */
add_task(async function test_about_inference_text_generation() {
  const { cleanup } = await setup({
    prefs: [
      ["browser.ml.llama.hwInference", true],
      [
        "browser.ml.modelHubRootUrl",
        "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
      ],
    ],
  });
  const tab = await BrowserTestUtils.openNewForegroundTab(
    gBrowser,
    "about:inference",
    true
  );
  try {
    await SpecialPowers.spawn(tab.linkedBrowser, [], async () => {
      const byId = id => content.document.getElementById(id);
      byId("subtab-devices").click();
      await ContentTaskUtils.waitForCondition(
        () => byId("devices.table").querySelector("td"),
        "The devices panel lists the inference devices",
        100,
        300
      );
      const kinds = [...byId("devices.table").querySelectorAll("tbody tr")].map(
        tr => tr.children[2].textContent
      );
      Assert.equal(
        byId("tg.device").options.length,
        2 + kinds.filter(kind => kind !== "cpu").length,
        "Every accelerator joins the text generation device select"
      );
      byId("subtab-text-generation").click();
      // Let the accessibility tree pick up the unhidden panel.
      await new Promise(resolve => content.requestAnimationFrame(resolve));
      await new Promise(resolve => content.requestAnimationFrame(resolve));
      byId("tg.hub").value = "";
      byId("tg.model").value = "Mozilla/test-llama/TinyStories-656K.Q8_0.gguf";
      byId("tg.prompt").value = "Once upon a time";
      byId("tg.maxTokens").value = "8";
      byId("tg.threads").value = "1";
      byId("tg.engines").value = "2";
      byId("tg.runs").value = "2";
      Assert.ok(
        byId("tg.kvCacheDtype").querySelector("option[value=q8_0]").disabled,
        "A quantized KV cache needs flash attention"
      );
      byId("tg.flashAttn").click();
      Assert.ok(
        !byId("tg.kvCacheDtype").querySelector("option[value=q8_0]").disabled,
        "Flash attention allows a quantized KV cache"
      );
      byId("tg.flashAttn").click();
      Assert.ok(
        byId("tg.kvCacheDtype").querySelector("option[value=q8_0]").disabled,
        "Turning flash attention off disables it again"
      );
      byId("tg.run").click();
      await ContentTaskUtils.waitForCondition(
        () => !byId("tg.run").disabled,
        "The runs finish",
        500,
        120
      );
      Assert.ok(
        byId("tg.summary").querySelector("td"),
        `The summary appears after every run: ${byId("tg.output").value}`
      );
      const rows = [...byId("tg.results").querySelectorAll("tbody tr")].map(
        tr => [...tr.children].map(td => td.textContent)
      );
      Assert.equal(rows.length, 4, "Two engines ran twice each");
      Assert.notEqual(rows[0][2], "-", "An engine's first run reports init");
      Assert.equal(rows[1][2], "-", "Its second run does not");
      Assert.notEqual(rows[0][7], "-", "Output tokens are reported");
      Assert.notEqual(rows[0][9], "-", "Throughput is reported");
      const summary = byId("tg.summary").querySelectorAll("tbody tr");
      Assert.equal(summary.length, 6, "Every timing is summarised");
    });
  } finally {
    BrowserTestUtils.removeTab(tab);
    await cleanup();
  }
});

/**
 * Checks that the location hash selects a category or a subtab, and follows
 * the selection.
 */
add_task(async function test_about_inference_hash() {
  const tab = await BrowserTestUtils.openNewForegroundTab(
    gBrowser,
    "about:inference#text-generation",
    true
  );
  try {
    await SpecialPowers.spawn(tab.linkedBrowser, [], async () => {
      const { document, location } = content;
      const byId = id => document.getElementById(id);
      const selected = () => [
        document.querySelector(".category[selected=true]").id,
        document.querySelector(".subtabs [aria-selected=true]").id,
      ];
      Assert.deepEqual(
        selected(),
        ["category-local-inference", "subtab-text-generation"],
        "The hash opens the subtab"
      );
      Assert.ok(!byId("text-generation").hidden, "The subtab panel is shown");
      Assert.ok(byId("inference-pad").hidden, "The default panel is hidden");

      byId("subtab-devices").click();
      await ContentTaskUtils.waitForCondition(
        () => !byId("devices").hidden,
        "Clicking a subtab shows it"
      );
      Assert.equal(
        location.hash,
        "#devices",
        "Clicking a subtab sets the hash"
      );

      location.hash = "benchmark";
      await ContentTaskUtils.waitForCondition(
        () => selected()[0] === "category-benchmark",
        "Changing the hash to a category shows it"
      );
      Assert.ok(byId("local-inference").hidden, "The subtabs give way");

      location.hash = "memories";
      await ContentTaskUtils.waitForCondition(
        () => !byId("memories").hidden,
        "Changing the hash to memories shows the panel"
      );
      Assert.ok(
        document.body.classList.contains("memories"),
        "The panel takes the page"
      );
      Assert.equal(
        content.getComputedStyle(byId("categories")).display,
        "none",
        "The navigation gives way"
      );
      Assert.ok(
        !document.querySelector(".subtabs [aria-controls='memories']"),
        "The panel has no tab"
      );

      location.hash = "devices";
      await ContentTaskUtils.waitForCondition(
        () => byId("memories").hidden,
        "Leaving the hash hides the panel"
      );
      Assert.ok(
        !document.body.classList.contains("memories"),
        "The page is back"
      );
      Assert.deepEqual(
        selected(),
        ["category-local-inference", "subtab-devices"],
        "The hash's subtab is shown"
      );

      location.hash = "nowhere";
      await new Promise(resolve => content.requestAnimationFrame(resolve));
      Assert.deepEqual(
        selected(),
        ["category-local-inference", "subtab-devices"],
        "An unknown hash changes nothing"
      );
    });
  } finally {
    BrowserTestUtils.removeTab(tab);
  }
});

/**
 * Runs the memories pipeline on the fixtures through the memories panel, with
 * the model mocked, and checks it leaves the profile memory store alone.
 */
add_task(
  { skip_if: () => AppConstants.MOZ_BUILD_APP != "browser" },
  async function test_about_inference_memories() {
    const { MockEngineManager } = ChromeUtils.importESModule(
      "resource://testing-common/AIWindowTestUtils.sys.mjs"
    );
    const { PURPOSES } = ChromeUtils.importESModule(
      "moz-src:///browser/components/aiwindow/models/Utils.sys.mjs"
    );
    const { MemoryStore } = ChromeUtils.importESModule(
      "moz-src:///browser/components/aiwindow/services/MemoryStore.sys.mjs"
    );
    await SpecialPowers.pushPrefEnv({
      set: [
        ["browser.ml.enable", true],
        ["browser.ml.llama.hwInference", true],
      ],
    });
    const manager = new MockEngineManager();
    const tab = await BrowserTestUtils.openNewForegroundTab(
      gBrowser,
      "about:inference#memories",
      true
    );
    try {
      const input = await SpecialPowers.spawn(
        tab.linkedBrowser,
        [],
        async () => {
          const byId = id => content.document.getElementById(id);
          Assert.ok(!byId("memories").hidden, "The panel is shown on its own");
          const view = content.document.querySelector("memories-view");
          await ContentTaskUtils.waitForCondition(
            () => view.input && !view.shadowRoot.getElementById("run").disabled,
            "The fixtures are shown"
          );
          const text = view.shadowRoot.textContent;
          Assert.ok(
            text.includes("sourdough starter feeding schedule"),
            "The sessions' searches are shown"
          );
          Assert.ok(
            text.includes("Bakes bread at home"),
            "The scratch store's memories are shown"
          );
          view.shadowRoot.getElementById("run").click();
          return {
            sessions: view.input.sessions,
            memories: view.input.memories.length,
          };
        }
      );
      Assert.greater(input.sessions, 0, "The fixtures make sessions");
      Assert.equal(input.memories, 6, "The scratch store holds the seeds");

      const summaries = [
        "Likes outdoor running",
        "Bakes sourdough bread",
        "Plans city walking trips",
      ];
      const generation = await manager.captureRequest({
        purpose: PURPOSES.MEMORY_GENERATION,
      });
      Assert.ok(
        JSON.stringify(generation.request.args).includes("sourdough starter"),
        "The fixture sessions reach the generation prompt"
      );
      const engine = manager.engines.get(PURPOSES.MEMORY_GENERATION);
      let terminations = 0;
      engine.terminate = async () => terminations++;
      generation.respond(
        JSON.stringify(
          summaries.map(memory_summary => ({
            reasoning: "Read several pages on the topic",
            category: "Hobbies & Leisure",
            intent: "Research / Learn",
            memory_summary,
            score: 4,
            evidence: [],
          }))
        )
      );
      const filter = await manager.captureRequest({
        purpose: PURPOSES.MEMORY_GENERATION,
      });
      filter.respond(JSON.stringify({ kept_memories: summaries }));
      const merge = await manager.captureRequest({
        purpose: PURPOSES.MEMORY_GENERATION,
      });
      merge.respond(
        JSON.stringify([
          {
            component_statements: [
              "Likes outdoor running",
              "Enjoys hiking coastal trails",
            ],
            new_reasoning: "Both are about the outdoors",
            new_statement: "Enjoys running and hiking outdoors",
          },
        ])
      );
      // Two kept memories and the merged one are new to the store.
      const classified = [];
      for (let i = 0; i < 3; i++) {
        const classification = await manager.captureRequest({
          purpose: PURPOSES.MEMORY_GENERATION,
        });
        classified.push(JSON.stringify(classification.request.args));
        classification.respond(
          JSON.stringify({
            categories: ["Hobbies & Leisure"],
            intents: ["Research / Learn"],
          })
        );
      }
      Assert.ok(
        classified.some(request => request.includes("Bakes sourdough bread")),
        "The new memories are classified"
      );
      Assert.ok(
        classified.some(request =>
          request.includes("Enjoys running and hiking outdoors")
        ),
        "The merged memory is classified"
      );

      const shown = await SpecialPowers.spawn(
        tab.linkedBrowser,
        [],
        async () => {
          const view = content.document.querySelector("memories-view");
          await ContentTaskUtils.waitForCondition(
            () => view.run.status !== "running",
            "The run finishes",
            100,
            600
          );
          await view.updateComplete;
          const root = view.shadowRoot;
          const marks = [...root.querySelectorAll("#store tbody tr")].map(
            tr => [
              tr.children[0].textContent.trim(),
              tr.children[1].textContent.trim(),
            ]
          );
          return {
            status: view.run.status,
            error: String(view.run.error),
            calls: view.run.calls.map(call => call.step),
            counts: view.counts,
            rails: root.querySelectorAll(".steps .rail.done").length,
            marks,
          };
        }
      );
      Assert.equal(
        shown.status,
        "done",
        `The run has no error: ${shown.error}`
      );
      Assert.deepEqual(
        shown.calls,
        [
          "generation",
          "filter",
          "merge",
          "classification",
          "classification",
          "classification",
        ],
        "Each step made its model calls"
      );
      Assert.deepEqual(
        shown.counts,
        {
          sessions: input.sessions,
          candidates: 3,
          kept: 3,
          before: 9,
          memories: 8,
          merged: 1,
          new: 2,
          fresh: 3,
        },
        "The steps' counts follow the scratch store"
      );
      Assert.equal(shown.rails, 4, "The four steps are done");
      Assert.deepEqual(
        shown.marks
          .filter(([mark]) => mark === "new")
          .map(([, memory]) => memory),
        ["Bakes sourdough bread", "Plans city walking trips"],
        "The generated memories are marked new"
      );
      Assert.deepEqual(
        shown.marks
          .filter(([mark]) => mark === "merged")
          .map(([, memory]) => memory),
        ["Enjoys running and hiking outdoors"],
        "The merged memory is marked merged"
      );
      Assert.equal(terminations, 1, "The engine is terminated");
      Assert.deepEqual(
        await MemoryStore.getMemories({ includeSoftDeleted: true }),
        [],
        "The profile memory store stays empty"
      );
      manager.assertAllRequestsHandled();
    } finally {
      BrowserTestUtils.removeTab(tab);
      await manager.rejectAllRequests();
      manager.cleanupMocks();
      await SpecialPowers.popPrefEnv();
    }
  }
);
