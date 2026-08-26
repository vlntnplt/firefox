/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

const { MemoriesSchedulers } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/models/memories/MemoriesSchedulers.sys.mjs"
);
const { MemoryStore } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/services/MemoryStore.sys.mjs"
);

add_setup(async function () {
  await SpecialPowers.pushPrefEnv({
    set: [["browser.smartwindow.enabled", true]],
  });

  registerCleanupFunction(() => {
    AIWindow.uninit();
    AIWindow.init(window);
  });
});

describe("MemoriesSchedulers scheduling from AIWindow", () => {
  let stub;

  beforeEach(() => {
    stub = sinon.stub(MemoriesSchedulers, "maybeRunAndSchedule");
  });

  afterEach(() => {
    stub.restore();
    if (AIWindow.isAIWindowActive(window)) {
      AIWindow.toggleAIWindow(window, false);
    }
  });

  describe("toggling the main window", () => {
    it("calls maybeRunAndSchedule once when toggling to AI Window", () => {
      AIWindow.toggleAIWindow(window, true);

      Assert.ok(stub.calledOnce, "called once on toggle to AI Window");
      Assert.ok(
        document.documentElement.hasAttribute("ai-window"),
        "ai-window attribute is set after toggle"
      );
    });

    it("does not call maybeRunAndSchedule when toggling to Classic", () => {
      AIWindow.toggleAIWindow(window, true);
      // Only the toggle-off below should be counted
      stub.resetHistory();

      AIWindow.toggleAIWindow(window, false);

      Assert.ok(stub.notCalled, "not called on toggle to Classic");
      Assert.ok(
        !document.documentElement.hasAttribute("ai-window"),
        "ai-window attribute is removed after toggle"
      );
    });
  });

  describe("initializing against a new window", () => {
    let testWin;

    beforeEach(async () => {
      AIWindow.uninit();
      AIWindow.init(window);
      testWin = await BrowserTestUtils.openNewBrowserWindow({
        private: false,
      });
    });

    afterEach(async () => {
      await BrowserTestUtils.closeWindow(testWin);
      AIWindow.uninit();
      testWin = null;
    });

    it("calls maybeRunAndSchedule during init when window is AI Window", async () => {
      testWin.document.documentElement.setAttribute("ai-window", "");

      AIWindow.uninit();
      AIWindow.init(testWin);
      await testWin.delayedStartupPromise;

      Assert.ok(stub.calledOnce, "called once during AI Window init");
    });

    it("does not call maybeRunAndSchedule during init when window is not AI Window", () => {
      AIWindow.uninit();
      AIWindow.init(testWin);

      Assert.ok(stub.notCalled, "not called during classic window init");
    });

    it("still calls maybeRunAndSchedule for a later AI window even if an earlier window init() saw wasn't one", async () => {
      // beforeEach already called init() against `window` (not an AI
      // window) without an intervening uninit() - simulates a cold start
      // where the first window init() sees isn't yet the real AI window.
      testWin.document.documentElement.setAttribute("ai-window", "");

      AIWindow.init(testWin);
      await testWin.delayedStartupPromise;

      Assert.ok(
        stub.calledOnce,
        "called for a later AI window despite an earlier non-AI window init()"
      );
    });
  });

  describe("firstrun completing after an AI Window toggle", () => {
    beforeEach(async () => {
      await SpecialPowers.pushPrefEnv({
        set: [["browser.smartwindow.firstrun.hasCompleted", false]],
      });
      AIWindow.uninit();
      AIWindow.init(window);
    });

    afterEach(async () => {
      AIWindow.uninit();
      await SpecialPowers.popPrefEnv();
      AIWindow.init(window);
    });

    it("calls maybeRunAndSchedule when firstrun completes", () => {
      Services.prefs.setBoolPref(
        "browser.smartwindow.firstrun.hasCompleted",
        true
      );

      Assert.ok(stub.calledOnce, "called once when firstrun completes");
    });

    it("does not call maybeRunAndSchedule when firstrun is set to false", () => {
      Services.prefs.setBoolPref(
        "browser.smartwindow.firstrun.hasCompleted",
        true
      );
      // Only the false-transition below should be counted.
      stub.resetHistory();

      Services.prefs.setBoolPref(
        "browser.smartwindow.firstrun.hasCompleted",
        false
      );

      Assert.ok(stub.notCalled, "not called when firstrun is set to false");
    });
  });
});

describe("Memories embeddings warmup from AIWindow", () => {
  let warmupStub;
  let schedulersStub;

  beforeEach(async () => {
    warmupStub = sinon.stub(MemoryStore, "warmup").resolves();
    schedulersStub = sinon.stub(MemoriesSchedulers, "maybeRunAndSchedule");
    await SpecialPowers.pushPrefEnv({
      set: [["browser.smartwindow.tos.consentTime", 123]],
    });
    AIWindow.uninit();
    AIWindow.init(window);
  });

  afterEach(async () => {
    warmupStub.restore();
    schedulersStub.restore();
    if (AIWindow.isAIWindowActive(window)) {
      AIWindow.toggleAIWindow(window, false);
    }
    await SpecialPowers.popPrefEnv();
  });

  it("calls warmup when toggling to AI Window", () => {
    AIWindow.toggleAIWindow(window, true);

    Assert.ok(warmupStub.calledOnce, "warmup called once on toggle");
  });

  it("does not call warmup without ToS consent", async () => {
    await SpecialPowers.pushPrefEnv({
      set: [["browser.smartwindow.tos.consentTime", 0]],
    });

    AIWindow.toggleAIWindow(window, true);

    Assert.ok(warmupStub.notCalled, "warmup not called without consent");
    await SpecialPowers.popPrefEnv();
  });

  it("calls warmup when the memory store changes while an AI Window is active", () => {
    AIWindow.toggleAIWindow(window, true);
    warmupStub.resetHistory();

    Services.obs.notifyObservers(null, MemoryStore.MEMORY_STORE_CHANGED);

    Assert.ok(warmupStub.calledOnce, "warmup called on store change");
  });

  it("does not call warmup on store change without an active AI Window", () => {
    Services.obs.notifyObservers(null, MemoryStore.MEMORY_STORE_CHANGED);

    Assert.ok(
      warmupStub.notCalled,
      "warmup not called without an active AI Window"
    );
  });
});
