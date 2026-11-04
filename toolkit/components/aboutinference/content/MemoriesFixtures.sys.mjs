/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * Synthetic browsing history, chats and memories for running the memories
 * pipeline on fixed data: the 30 visits and 15 chat messages of a typical
 * four hour window between two scheduler runs, written to a scratch Places
 * database and a scratch chat store, and enough aged memories for a merge
 * pass to run over the store.
 */

import { Sqlite } from "resource://gre/modules/Sqlite.sys.mjs";

const lazy = {};

ChromeUtils.defineESModuleGetters(lazy, {
  ChatConversation:
    // eslint-disable-next-line mozilla/no-browser-refs-in-toolkit
    "moz-src:///browser/components/aiwindow/ui/modules/ChatStore.sys.mjs",
  ChatStoreImpl:
    // eslint-disable-next-line mozilla/no-browser-refs-in-toolkit
    "moz-src:///browser/components/aiwindow/ui/modules/ChatStore.sys.mjs",
  MEMORY_SENSITIVITY_CATEGORY_NOT_SENSITIVE:
    // eslint-disable-next-line mozilla/no-browser-refs-in-toolkit
    "moz-src:///browser/components/aiwindow/models/memories/MemoriesConstants.sys.mjs",
  MEMORY_TYPE_SHORT_TERM_MEMORY:
    // eslint-disable-next-line mozilla/no-browser-refs-in-toolkit
    "moz-src:///browser/components/aiwindow/models/memories/MemoriesConstants.sys.mjs",
  SESSION:
    // eslint-disable-next-line mozilla/no-browser-refs-in-toolkit
    "moz-src:///browser/components/aiwindow/models/memories/MemoriesConstants.sys.mjs",
});

const MINUTE_MS = 60 * 1000;
const EVENT_GAP_MS = 2 * MINUTE_MS;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

/**
 * The Places tables and columns the memories history query reads.
 */
const PLACES_SCHEMA = [
  `CREATE TABLE moz_origins (
    id INTEGER PRIMARY KEY,
    host TEXT NOT NULL UNIQUE,
    frecency INTEGER NOT NULL DEFAULT -1
  )`,
  `CREATE TABLE moz_places (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    url_hash INTEGER NOT NULL,
    frecency INTEGER NOT NULL DEFAULT -1,
    origin_id INTEGER REFERENCES moz_origins(id)
  )`,
  `CREATE TABLE moz_historyvisits (
    id INTEGER PRIMARY KEY,
    place_id INTEGER NOT NULL REFERENCES moz_places(id),
    visit_date INTEGER NOT NULL
  )`,
  `CREATE TABLE moz_places_metadata (
    id INTEGER PRIMARY KEY,
    place_id INTEGER NOT NULL REFERENCES moz_places(id),
    created_at INTEGER NOT NULL,
    total_view_time INTEGER NOT NULL DEFAULT 0
  )`,
  `CREATE TABLE moz_inputhistory (
    place_id INTEGER NOT NULL REFERENCES moz_places(id),
    input TEXT NOT NULL,
    PRIMARY KEY (place_id, input)
  )`,
];

/**
 * One browsing session per entry, oldest first, half an hour apart with
 * their events two minutes apart, so the session builder keeps them separate.
 * Each session is one web search followed by its pages, with the chat
 * messages sent while browsing them.
 */
const SESSIONS = [
  {
    minutesAgo: 230,
    topic: "hiking",
    query: "trail running shoes for beginners",
    pages: [
      [
        "https://trails.example.com/beginner-guide",
        "Beginner Trail Running Guide",
      ],
      ["https://gear.example.org/trail-shoes", "Choosing Trail Running Shoes"],
      [
        "https://trails.example.com/coastal-day-hikes",
        "Best Day Hikes Along the Coast",
      ],
    ],
    chats: [
      "What should I pack for a half day hike along the coast?",
      "Are trail running shoes fine for a rocky coastal path?",
    ],
  },
  {
    minutesAgo: 200,
    topic: "baking",
    query: "sourdough starter feeding schedule",
    pages: [
      [
        "https://breadlab.example.com/starter",
        "Sourdough Starter From Scratch",
      ],
      [
        "https://breadlab.example.com/hydration",
        "Understanding Dough Hydration",
      ],
      ["https://kitchen.example.org/dutch-oven", "Dutch Oven Baking Tips"],
    ],
    chats: [
      "My sourdough loaf came out dense, how do I get a more open crumb?",
      "How often should I feed a starter I keep in the fridge?",
    ],
  },
  {
    minutesAgo: 170,
    topic: "travel",
    query: "lisbon walking tour alfama",
    pages: [
      [
        "https://travel.example.com/lisbon-itinerary",
        "Three Days in Lisbon Itinerary",
      ],
      [
        "https://travel.example.com/lisbon-trams",
        "Riding the Historic Trams of Lisbon",
      ],
      [
        "https://food.example.net/pastel-de-nata",
        "Custard Tart Bakeries Worth Visiting",
      ],
    ],
    chats: [
      "Plan a relaxed three day walking itinerary for Lisbon.",
      "Which Lisbon neighbourhood is quietest to stay in?",
    ],
  },
  {
    minutesAgo: 140,
    topic: "homelab",
    query: "raspberry pi home server setup",
    pages: [
      [
        "https://tech.example.com/pi-home-server",
        "Setting Up a Raspberry Pi Home Server",
      ],
      [
        "https://tech.example.com/self-hosted-media",
        "Self Hosting a Media Library",
      ],
      [
        "https://devblog.example.org/static-sites",
        "Static Site Generators Compared",
      ],
    ],
    chats: [
      "Which static site generator is easiest to self host on a small board?",
      "Is a reverse proxy worth it for two services at home?",
    ],
  },
  {
    minutesAgo: 110,
    topic: "hiking",
    query: "trail running training plan beginners",
    pages: [
      [
        "https://trails.example.com/training-plan",
        "Trail Running Training Plan for Beginners",
      ],
      ["https://gear.example.org/running-vests", "Running Vest Buying Guide"],
      [
        "https://trails.example.com/mountain-loops",
        "Scenic Mountain Loop Trails",
      ],
    ],
    chats: [
      "How many runs a week should a beginner trail plan have?",
      "Do I need a running vest for runs under two hours?",
    ],
  },
  {
    minutesAgo: 80,
    topic: "baking",
    query: "overnight focaccia recipe",
    pages: [
      ["https://kitchen.example.org/focaccia", "Baking Focaccia at Home"],
      [
        "https://breadlab.example.com/whole-wheat",
        "Whole Wheat Sourdough Recipe",
      ],
      ["https://kitchen.example.org/bread-flours", "A Guide to Bread Flours"],
    ],
    chats: [
      "Can I use my sourdough starter to make focaccia overnight?",
      "Which flour gives focaccia a crisp bottom?",
    ],
  },
  {
    minutesAgo: 50,
    topic: "travel",
    query: "lisbon to sintra day trip train",
    pages: [
      ["https://travel.example.com/sintra-day-trip", "A Day Trip to Sintra"],
      [
        "https://travel.example.com/lisbon-transport-pass",
        "Lisbon Public Transport Passes Explained",
      ],
    ],
    chats: [
      "Is Sintra doable as a half day from Lisbon by train?",
      "Should I buy a transport pass for four days in Lisbon?",
    ],
  },
  {
    minutesAgo: 20,
    topic: "homelab",
    query: "self hosted photo backup",
    pages: [
      [
        "https://tech.example.com/photo-backup",
        "Self Hosting a Photo Backup Service",
      ],
      [
        "https://devblog.example.org/docker-compose-basics",
        "Docker Compose Basics for Home Servers",
      ],
    ],
    chats: ["How much storage does a self hosted photo backup need?"],
  },
];

const SEED_MEMORIES = [
  {
    memory_summary: "Enjoys hiking coastal trails",
    reasoning: "Read several guides to coastal day hikes",
    tags: ["category:Sports & Fitness", "intent:Plan / Organize"],
    keywords: ["hiking", "coast"],
  },
  {
    memory_summary: "Bakes bread at home",
    reasoning: "Repeatedly looked up home bread baking techniques",
    tags: ["category:Food & Drink", "intent:Research / Learn"],
    keywords: ["bread", "baking"],
  },
  {
    memory_summary: "Interested in self hosting software",
    reasoning: "Compared tools for running services on home hardware",
    tags: ["category:Computers & Electronics", "intent:Research / Learn"],
    keywords: ["self hosting", "home server"],
  },
  {
    memory_summary: "Plans a winter trip to Lisbon",
    reasoning: "Compared Lisbon neighbourhoods and flights for January",
    tags: ["category:Travel", "intent:Plan / Organize"],
    keywords: ["lisbon", "trip"],
  },
  {
    memory_summary: "Prefers recipes without dairy",
    reasoning: "Searched for dairy free versions of several recipes",
    tags: ["category:Food & Drink", "intent:Research / Learn"],
    keywords: ["dairy free", "recipes"],
  },
  {
    memory_summary: "Reads long articles about urban planning",
    reasoning: "Spent a long time on several urban planning essays",
    tags: ["category:News & Society", "intent:Research / Learn"],
    keywords: ["urban planning", "cities"],
  },
];

/**
 * Writes the sessions' visits to a Places-shaped database.
 *
 * @param {object} db  Sqlite connection to an empty database
 * @param {number} now
 */
async function writeHistory(db, now) {
  for (const statement of PLACES_SCHEMA) {
    await db.execute(statement);
  }
  const origins = new Map();
  let placeId = 0;
  const addVisit = async (url, title, visitMs, viewTimeMs) => {
    const { host } = new URL(url);
    if (!origins.has(host)) {
      origins.set(host, origins.size + 1);
      await db.execute(
        "INSERT INTO moz_origins (id, host, frecency) VALUES (:id, :host, 1000)",
        { id: origins.get(host), host }
      );
    }
    placeId++;
    await db.execute(
      `INSERT INTO moz_places (id, url, title, url_hash, frecency, origin_id)
       VALUES (:id, :url, :title, :id, 100, :originId)`,
      { id: placeId, url, title, originId: origins.get(host) }
    );
    await db.execute(
      `INSERT INTO moz_historyvisits (place_id, visit_date)
       VALUES (:placeId, :visitDate)`,
      { placeId, visitDate: visitMs * 1000 }
    );
    if (viewTimeMs) {
      await db.execute(
        `INSERT INTO moz_places_metadata (place_id, created_at, total_view_time)
         VALUES (:placeId, :createdAt, :viewTime)`,
        { placeId, createdAt: visitMs, viewTime: viewTimeMs }
      );
    }
  };
  await db.executeTransaction(async () => {
    for (const { minutesAgo, query, pages } of SESSIONS) {
      const startMs = now - minutesAgo * MINUTE_MS;
      await addVisit(
        `https://www.google.com/search?q=${encodeURIComponent(query)}`,
        `${query} - Google Search`,
        startMs,
        0
      );
      for (const [i, [url, title]] of pages.entries()) {
        await addVisit(
          url,
          title,
          startMs + (i + 1) * EVENT_GAP_MS,
          (1 + i) * MINUTE_MS
        );
      }
    }
  });
}

/**
 * Writes the sessions' chats to a chat store, one conversation per session.
 *
 * @param {ChatStoreImpl} chatStore
 * @param {number} now
 */
async function writeChats(chatStore, now) {
  for (const { minutesAgo, topic, pages, chats } of SESSIONS) {
    const startMs = now - minutesAgo * MINUTE_MS;
    const pageUrl = new URL(pages.at(-1)[0]);
    const conversation = new lazy.ChatConversation({
      pageUrl,
      createdDate: startMs,
      updatedDate: startMs + (pages.length + chats.length) * EVENT_GAP_MS,
    });
    conversation.title = `About ${topic}`;
    for (const [i, chat] of chats.entries()) {
      const message = conversation.addUserMessage(chat, pageUrl);
      message.createdDate = startMs + (pages.length + 1 + i) * EVENT_GAP_MS;
    }
    await chatStore.updateConversation(conversation);
  }
}

/**
 * Writes the fixtures to a scratch directory, with dates relative to `now`.
 *
 * @param {number} [now=Date.now()]  Reference time in ms since epoch
 * @returns {Promise<{
 *   db: object,
 *   chatStore: ChatStoreImpl,
 *   seedMemories: Array<object>,
 *   close: () => Promise<void>,
 * }>}
 *   `db` is a connection to the Places-shaped database of the visits,
 *   `chatStore` the store of the chats, `seedMemories` are MemoryStore.addMemory
 *   partials last aged two days ago, and `close` removes the directory.
 */
export async function openMemoriesFixtures(now = Date.now()) {
  const dir = await IOUtils.createUniqueDirectory(
    PathUtils.tempDir,
    "memories-fixtures"
  );
  const db = await Sqlite.openConnection({
    path: PathUtils.join(dir, "places.sqlite"),
  });
  const chatStore = new lazy.ChatStoreImpl(PathUtils.join(dir, "chats.sqlite"));
  const close = async () => {
    await db.close();
    await chatStore.destroyDatabase();
    await IOUtils.remove(dir, { recursive: true, ignoreAbsent: true });
  };
  try {
    await writeHistory(db, now);
    await writeChats(chatStore, now);
  } catch (error) {
    await close();
    throw error;
  }

  const agedMs = now - 2 * DAY_MS;
  const seedMemories = SEED_MEMORIES.map(memory => ({
    ...memory,
    type: lazy.MEMORY_TYPE_SHORT_TERM_MEMORY,
    sources: [lazy.SESSION],
    source_ids: { history_source_ids: [], conversation_source_ids: [] },
    sensitivity_category: lazy.MEMORY_SENSITIVITY_CATEGORY_NOT_SENSITIVE,
    created_at: agedMs - DAY_MS,
    updated_at: agedMs,
    last_accessed: agedMs,
    lifetime_accessed_count: 3,
    merge_count: 0,
  }));

  return { db, chatStore, seedMemories, close };
}
