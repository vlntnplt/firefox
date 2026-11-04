/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

do_get_profile();

const { openMemoriesFixtures } = ChromeUtils.importESModule(
  "moz-src:///toolkit/components/aboutinference/content/MemoriesFixtures.sys.mjs"
);
const { getRecentHistory } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/models/memories/MemoriesHistorySource.sys.mjs"
);
const { getRecentChats } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/models/memories/MemoriesChatSource.sys.mjs"
);
const { buildSessions } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/models/memories/MemoriesSessions.sys.mjs"
);
const { runHeuristicGate } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/models/memories/MemoriesSessionGate.sys.mjs"
);
const { MemoriesManager } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/models/memories/MemoriesManager.sys.mjs"
);
const { MemoryStoreImpl } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/services/MemoryStore.sys.mjs"
);
const { GATE_KEEP } = ChromeUtils.importESModule(
  "moz-src:///browser/components/aiwindow/models/memories/MemoriesConstants.sys.mjs"
);

add_task(async function test_fixtures_feed_the_memories_sources() {
  const now = Date.now();
  const { db, chatStore, seedMemories, close } =
    await openMemoriesFixtures(now);
  const dir = PathUtils.parent(chatStore.databaseFilePath);
  try {
    const rows = await getRecentHistory({ days: 7, maxResults: 500, db });
    Assert.equal(
      rows.length,
      30,
      "The query reads a four hour window of visits"
    );
    Assert.equal(
      rows.filter(row => row.source === "search").length,
      8,
      "One web search per session"
    );
    const fourHoursAgo = (now - 4 * 60 * 60 * 1000) * 1000;
    Assert.ok(
      rows.every(row => row.visitDateMicros >= fourHoursAgo),
      "Every visit falls within the last four hours"
    );

    const chats = await getRecentChats(0, 50, 7, { chatStore });
    Assert.equal(chats.length, 15, "A four hour window of chat messages");

    const kept = buildSessions(rows, chats).filter(
      session => runHeuristicGate(session).decision === GATE_KEEP
    );
    Assert.equal(
      kept.length,
      8,
      "Every session stays apart and passes the gate"
    );

    const store = new MemoryStoreImpl(null);
    for (const memory of seedMemories) {
      await store.addMemory(memory);
    }
    await MemoriesManager.runMemoryMaintenance({ store });
    const maintained = await store.getMemories();
    Assert.equal(
      maintained.length,
      seedMemories.length,
      "Seed memories survive maintenance"
    );
    Assert.ok(
      maintained.every(m => m.updated_at > seedMemories[0].updated_at),
      "Maintenance ages every seed memory"
    );
  } finally {
    await close();
  }
  Assert.ok(!(await IOUtils.exists(dir)), "close removes the scratch files");
});
