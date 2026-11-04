/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/* global HWInference */

import { html, nothing } from "chrome://global/content/vendor/lit.all.mjs";
import { MozLitElement } from "chrome://global/content/lit-utils.mjs";

const { AppConstants } = ChromeUtils.importESModule(
  "resource://gre/modules/AppConstants.sys.mjs"
);

const AVAILABLE = AppConstants.MOZ_BUILD_APP == "browser";

const lazy = {};
ChromeUtils.defineESModuleGetters(lazy, {
  DownloadUtils: "resource://gre/modules/DownloadUtils.sys.mjs",
  ModelHub: "chrome://global/content/ml/ModelHub.sys.mjs",
  ProgressStatusText: "chrome://global/content/ml/Utils.sys.mjs",
  ProgressType: "chrome://global/content/ml/Utils.sys.mjs",
  TextGenerationEngine:
    "moz-src:///toolkit/components/ml/textgeneration/TextGenerationEngine.sys.mjs",
  openMemoriesFixtures:
    "moz-src:///toolkit/components/aboutinference/content/MemoriesFixtures.sys.mjs",
});
if (AVAILABLE) {
  /* eslint-disable mozilla/no-browser-refs-in-toolkit */
  ChromeUtils.defineESModuleGetters(lazy, {
    buildConversation:
      "moz-src:///browser/components/aiwindow/models/PromptLoader.sys.mjs",
    MemoriesManager:
      "moz-src:///browser/components/aiwindow/models/memories/MemoriesManager.sys.mjs",
    MEMORIES_ON_DEVICE_ENGINE:
      "moz-src:///browser/components/aiwindow/models/memories/MemoriesManager.sys.mjs",
    MemoryStoreImpl:
      "moz-src:///browser/components/aiwindow/services/MemoryStore.sys.mjs",
    MODEL_FEATURES:
      "moz-src:///browser/components/aiwindow/models/Utils.sys.mjs",
    renderSessionsForPrompt:
      "moz-src:///browser/components/aiwindow/models/memories/Memories.sys.mjs",
  });
  /* eslint-enable mozilla/no-browser-refs-in-toolkit */
}

const FEATURE_ID = "memories-on-device-panel";
const ENGINE_ID = "about-inference-memories";
const STEPS = ["generation", "filter", "merge", "classification"];
const DASH = "—";

const numberFormat = new Intl.NumberFormat();
const rateFormat = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 1,
});

/**
 * The JSON a model call answered with, fenced or not, or null.
 *
 * @param {string} text
 */
function parseJSON(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  try {
    return JSON.parse(fenced ? fenced[1] : text);
  } catch {
    return null;
  }
}

/**
 * The memories pipeline on fixed sessions and a scratch store, driven through
 * its injection points and watched through its engine: every model call's
 * tokens, time and memory, and what it answered.
 */
export class MemoriesView extends MozLitElement {
  static properties = {
    input: { type: Object, state: true },
    run: { type: Object, state: true },
    cleanup: { type: Boolean, state: true },
  };

  constructor() {
    super();
    this.input = null;
    this.run = null;
    this.cleanup = true;
  }

  connectedCallback() {
    super.connectedCallback();
    if (AVAILABLE && !this.input) {
      this.#load();
    }
  }

  async #load() {
    const fixtures = await lazy.openMemoriesFixtures();
    try {
      const store = await this.#seedStore(fixtures.seedMemories);
      const { sessions } = await lazy.MemoriesManager.collectSessions(
        {},
        { store, chatStore: fixtures.chatStore, db: fixtures.db }
      );
      this.input = {
        sessions: sessions.length,
        searches: sessions.reduce((n, s) => n + s.search_queries.length, 0),
        titles: sessions.reduce((n, s) => n + s.titles.length, 0),
        chats: sessions.reduce((n, s) => n + s.chats.length, 0),
        prompt: lazy.renderSessionsForPrompt(sessions),
        memories: await store.getMemories(),
      };
    } finally {
      await fixtures.close();
    }
  }

  async #seedStore(seedMemories) {
    const store = new lazy.MemoryStoreImpl(null);
    for (const memory of seedMemories) {
      await store.addMemory(memory);
    }
    return store;
  }

  /** @param {object} data  A ProgressAndStatusCallbackParams of the model file. */
  #onProgress(data) {
    const { model } = this.run;
    const now = ChromeUtils.now();
    if (data.type === lazy.ProgressType.DOWNLOAD) {
      if (data.statusText === lazy.ProgressStatusText.INITIATE) {
        model.status = "downloading";
        model.downloadStartedAt = now;
      } else if (data.statusText === lazy.ProgressStatusText.DONE) {
        model.status = "loading";
        model.downloadFinishedAt = now;
      } else {
        model.loaded = data.totalLoaded;
        model.size = data.total;
      }
    } else if (data.type === lazy.ProgressType.LOAD_FROM_CACHE) {
      model.status = "loading";
    }
    this.requestUpdate();
  }

  /** Removes the model the run downloaded, so the next run downloads again. */
  async #removeModel() {
    const { modelHubRootUrl, modelHubUrlTemplate, modelId, modelRevision } =
      lazy.MEMORIES_ON_DEVICE_ENGINE;
    const hub = new lazy.ModelHub({
      rootUrl: modelHubRootUrl,
      urlTemplate: modelHubUrlTemplate,
    });
    await hub.deleteModels({
      model: `${new URL(modelHubRootUrl).hostname}/${modelId}`,
      revision: modelRevision,
      deletedBy: ENGINE_ID,
    });
  }

  #phase(step) {
    this.run.phase = step;
    this.requestUpdate();
  }

  #calls(step) {
    return this.run?.calls.filter(call => call.step === step) ?? [];
  }

  /** What the filter kept and what the merge formed, by memory summary. */
  get #answers() {
    const kept = parseJSON(this.#calls("filter")[0]?.text ?? "")?.kept_memories;
    const merged = parseJSON(this.#calls("merge")[0]?.text ?? "");
    return {
      kept: Array.isArray(kept) ? kept : [],
      merged: Array.isArray(merged)
        ? merged.map(candidate => candidate.new_statement)
        : [],
    };
  }

  /** The memories the run added to the store, by summary: new or merged. */
  get #fresh() {
    const memories = this.run?.after ?? this.run?.merged;
    if (!memories) {
      return null;
    }
    const seeds = new Set(this.input.memories.map(m => m.memory_summary));
    const { merged } = this.#answers;
    return new Map(
      memories
        .filter(({ memory_summary }) => !seeds.has(memory_summary))
        .map(({ memory_summary }) => [
          memory_summary,
          merged.includes(memory_summary) ? "merged" : "new",
        ])
    );
  }

  /** The counts the step rows show, read off the calls and the store. */
  get counts() {
    const { input, run } = this;
    const { kept } = this.#answers;
    const fresh = this.#fresh;
    const generated = this.#calls("generation").reduce(
      (total, call) => total + (parseJSON(call.text)?.length ?? 0),
      0
    );
    const count = (value, when) => (when ? value : DASH);
    const freshCount = kind =>
      fresh
        ? [...fresh.values()].filter(k => !kind || k === kind).length
        : DASH;
    return {
      sessions: input?.sessions ?? DASH,
      candidates: count(generated, this.#calls("generation").length),
      kept: count(kept.length, this.#calls("filter").length),
      before: count(run?.merging?.length, run?.merging),
      memories: count(run?.merged?.length, run?.merged),
      merged: freshCount("merged"),
      new: freshCount("new"),
      fresh: freshCount(),
    };
  }

  async #run() {
    if (this.run?.status === "running") {
      return;
    }
    const run = {
      status: "running",
      error: null,
      model: { status: "pending", startedAt: ChromeUtils.now() },
      phase: null,
      calls: [],
      merging: null,
      merged: null,
      after: null,
    };
    this.run = run;
    let fixtures;
    let engine;
    try {
      if (!lazy.TextGenerationEngine.shouldRoute({ backend: "llama.cpp" })) {
        throw new Error("browser.ml.llama.hwInference is off");
      }
      fixtures = await lazy.openMemoriesFixtures();
      const store = await this.#seedStore(fixtures.seedMemories);
      const generation = await lazy.buildConversation(
        lazy.MODEL_FEATURES.MEMORIES_INITIAL_GENERATION_SYSTEM,
        {
          onDevice: {
            ...lazy.MEMORIES_ON_DEVICE_ENGINE,
            engineId: ENGINE_ID,
            featureId: FEATURE_ID,
          },
          notificationsCallback: data => this.#onProgress(data),
        }
      );
      ({ engine } = generation);
      const devices = engine.device ? await HWInference.listDevices() : [];
      run.device = devices.find(({ id }) => id === engine.device) ?? null;
      const classification = await lazy.buildConversation(
        lazy.MODEL_FEATURES.MEMORIES_MESSAGE_CLASSIFICATION_SYSTEM,
        { engine }
      );
      const engineRun = engine.run.bind(engine);
      engine.run = async request => {
        const result = await engineRun(request);
        run.calls.push({
          step: run.phase,
          text: result.finalOutput ?? "",
          metrics: result.metrics ?? {},
          memory: result.resourcesAfter?.memory ?? null,
        });
        this.requestUpdate();
        return result;
      };
      run.model.status = "ready";
      run.model.readyAt = ChromeUtils.now();

      const deps = {
        store,
        chatStore: fixtures.chatStore,
        db: fixtures.db,
        conversation: generation,
      };
      await lazy.MemoriesManager.runMemoryMaintenance(deps);

      this.#phase("generation");
      await lazy.MemoriesManager.generateMemoriesFromSessions({}, deps);
      // Generation runs one call per batch of sessions, then the filter.
      const generated = this.#calls("generation");
      if (generated.length > 1) {
        generated.at(-1).step = "filter";
      }

      this.#phase("merge");
      run.merging = await store.getMemories();
      await lazy.MemoriesManager.mergeMemories({}, deps);
      run.merged = await store.getMemories();

      this.#phase("classification");
      const fresh = this.#fresh;
      for (const memory of run.merged) {
        if (fresh.has(memory.memory_summary)) {
          await lazy.MemoriesManager.enrichExistingMemory(
            memory.id,
            memory.memory_summary,
            { store, conversation: classification }
          );
        }
      }

      run.after = await store.getMemories();
      run.status = "done";
    } catch (error) {
      run.status = "failed";
      run.error = error;
    } finally {
      run.phase = null;
      try {
        // Nothing manages an HWInference engine's lifetime but its creator.
        await engine?.terminate();
        await fixtures?.close();
        if (this.cleanup && run.model.downloadStartedAt != null) {
          await this.#removeModel();
        }
      } finally {
        this.requestUpdate();
      }
    }
  }

  #l10n(id, args) {
    return html`<span
      data-l10n-id=${id}
      data-l10n-args=${args ? JSON.stringify(args) : nothing}
    ></span>`;
  }

  #duration(ms) {
    if (ms == null) {
      return DASH;
    }
    const seconds = ms / 1000;
    return seconds < 60
      ? this.#l10n("about-inference-memories-seconds", { seconds })
      : this.#l10n("about-inference-memories-minutes", {
          minutes: Math.floor(seconds / 60),
          seconds: Math.round(seconds % 60),
        });
  }

  #tokens(count) {
    return count == null ? DASH : numberFormat.format(count);
  }

  #bytes(count) {
    return count == null ? DASH : lazy.DownloadUtils.getTransferTotal(count);
  }

  #renderStore(memories) {
    const marked = Boolean(this.run?.after);
    const fresh = this.#fresh;
    const mark = ({ memory_summary }) => fresh?.get(memory_summary) ?? "";
    return html`
      <table>
        <thead>
          <tr>
            ${marked ? html`<th></th>` : nothing}
            <th
              class="text"
              data-l10n-id="about-inference-memories-col-memory"
            ></th>
            <th data-l10n-id="about-inference-memories-col-type"></th>
            <th data-l10n-id="about-inference-memories-col-strength"></th>
            <th data-l10n-id="about-inference-memories-col-sources"></th>
          </tr>
        </thead>
        <tbody>
          ${memories.map(
            memory => html`
              <tr>
                ${marked
                  ? html`<td class="mark ${mark(memory)}">
                      ${mark(memory)
                        ? this.#l10n(`about-inference-memories-${mark(memory)}`)
                        : nothing}
                    </td>`
                  : nothing}
                <td class="text">${memory.memory_summary}</td>
                <td>${memory.type}</td>
                <td>${memory.strength.toFixed(2)}</td>
                <td>${memory.sources.length}</td>
              </tr>
            `
          )}
        </tbody>
      </table>
    `;
  }

  #renderDetails() {
    const { input, run } = this;
    if (!input) {
      return nothing;
    }
    const after = run?.after;
    const store = after
      ? {
          id: "about-inference-memories-store-after",
          args: {
            stored: after.length,
            new: this.counts.new,
            merged: this.counts.merged,
          },
        }
      : {
          id: "about-inference-memories-store-summary",
          args: { count: input.memories.length },
        };
    return html`
      <moz-card
        type="accordion"
        data-l10n-id="about-inference-memories-sessions"
        data-l10n-args=${JSON.stringify({
          sessions: input.sessions,
          searches: input.searches,
          titles: input.titles,
          chats: input.chats,
        })}
      >
        <pre>${input.prompt}</pre>
      </moz-card>
      <moz-card
        id="store"
        type="accordion"
        data-l10n-id=${store.id}
        data-l10n-args=${JSON.stringify(store.args)}
      >
        ${this.#renderStore(after ?? input.memories)}
      </moz-card>
    `;
  }

  #tile(value, labelId) {
    return html`<div class="tile">
      <span class="value">${value}</span>
      <span class="label" data-l10n-id=${labelId}></span>
    </div>`;
  }

  #renderTiles() {
    const model = this.run?.model;
    const calls = this.run?.calls ?? [];
    const sum = key =>
      calls.reduce((total, call) => total + (call.metrics[key] ?? 0), 0);
    const prefillMs = sum("timeToFirstToken");
    const decodeMs = sum("decodingTime");
    const rate = (tokens, ms) =>
      ms ? rateFormat.format(tokens / (ms / 1000)) : DASH;
    const loaded = model?.readyAt != null;
    const downloaded =
      model?.downloadFinishedAt != null
        ? model.downloadFinishedAt - model.downloadStartedAt
        : null;
    return html`
      <div class="tiles">
        ${this.#tile(
          this.#duration(prefillMs + decodeMs || null),
          "about-inference-memories-tile-inference"
        )}
        ${this.#tile(
          this.#duration(
            loaded
              ? model.readyAt - (model.downloadFinishedAt ?? model.startedAt)
              : null
          ),
          "about-inference-memories-tile-load"
        )}
        ${downloaded != null
          ? this.#tile(
              this.#duration(downloaded),
              "about-inference-memories-tile-download"
            )
          : nothing}
        ${this.#tile(
          rate(sum("inputTokens"), prefillMs),
          "about-inference-memories-tile-prefill"
        )}
        ${this.#tile(
          rate(sum("outputTokens"), decodeMs),
          "about-inference-memories-tile-decode"
        )}
        ${this.#tile(
          this.#bytes(Math.max(0, ...calls.map(call => call.memory)) || null),
          "about-inference-memories-tile-memory"
        )}
      </div>
    `;
  }

  #renderStepRow(name, index) {
    const calls = this.#calls(name);
    const sum = key =>
      calls.some(call => call.metrics[key] != null)
        ? calls.reduce((total, call) => total + (call.metrics[key] ?? 0), 0)
        : null;
    const prefill = sum("timeToFirstToken");
    let status = "pending";
    if (this.run?.phase === name) {
      status = "active";
    } else if (calls.length) {
      status = "done";
    }
    let position = "";
    if (index === 0) {
      position = "first";
    } else if (index === STEPS.length - 1) {
      position = "last";
    }
    const cls = status === "pending" ? "r pending" : "r";
    return html`
      <div class="r rail ${status} ${position}"></div>
      <div
        class="${cls} l name"
        data-l10n-id="about-inference-memories-${name}"
      ></div>
      <div class="${cls} l">
        ${this.#l10n(`about-inference-memories-flow-${name}`, this.counts)}
      </div>
      <div class=${cls}>${this.#tokens(sum("inputTokens"))}</div>
      <div class=${cls}>${this.#tokens(sum("outputTokens"))}</div>
      <div class=${cls}>
        ${status === "active"
          ? "…"
          : this.#duration(
              prefill == null ? null : prefill + sum("decodingTime")
            )}
      </div>
    `;
  }

  #renderSteps() {
    return html`
      <moz-card>
        <div class="steps">
          <div class="h"></div>
          <div class="h"></div>
          <div class="h"></div>
          ${["in", "out", "time"].map(
            col =>
              html`<div
                class="h"
                data-l10n-id="about-inference-memories-col-${col}"
              ></div>`
          )}
          ${STEPS.map((name, index) => this.#renderStepRow(name, index))}
        </div>
      </moz-card>
    `;
  }

  #renderModelStatus() {
    const model = this.run?.model;
    if (this.run?.status !== "running" || !model) {
      return nothing;
    }
    if (model.status === "downloading") {
      return html`
        <div class="download">
          ${lazy.DownloadUtils.getTransferTotal(model.loaded ?? 0, model.size)}
          <progress
            value=${model.loaded ?? 0}
            max=${model.size || 1}
          ></progress>
        </div>
      `;
    }
    return model.status === "pending"
      ? nothing
      : html`<span
          class="status"
          data-l10n-id="about-inference-memories-model-${model.status}"
        ></span>`;
  }

  #renderSetup() {
    const { modelId, backend } = lazy.MEMORIES_ON_DEVICE_ENGINE;
    const device = this.run?.device;
    const error = this.run?.error;
    return html`
      <moz-card class="setup">
        <dl>
          <dt data-l10n-id="about-inference-memories-model"></dt>
          <dd>
            ${modelId
              .split("/")
              .pop()
              .replace(/-GGUF$/i, "")}
            ${this.#renderModelStatus()}
          </dd>
          <dt data-l10n-id="about-inference-memories-backend"></dt>
          <dd>${backend}</dd>
          <dt data-l10n-id="about-inference-memories-device-label"></dt>
          <dd>
            ${device
              ? this.#l10n("about-inference-memories-device", {
                  device: device.description,
                  kind: device.kind,
                })
              : html`<span
                  class="pending"
                  data-l10n-id="about-inference-memories-device-pending"
                ></span>`}
          </dd>
        </dl>
        ${error ? html`<p class="error">${String(error)}</p>` : nothing}
        <div class="actions">
          <moz-checkbox
            id="cleanup"
            data-l10n-id="about-inference-memories-cleanup"
            ?checked=${this.cleanup}
            @change=${event => (this.cleanup = event.target.checked)}
          ></moz-checkbox>
          <moz-button
            id="run"
            type="primary"
            data-l10n-id="about-inference-memories-run"
            ?disabled=${this.run?.status === "running" || !this.input}
            @click=${this.#run}
          ></moz-button>
        </div>
      </moz-card>
    `;
  }

  render() {
    if (!AVAILABLE) {
      return nothing;
    }
    return html`
      <link rel="stylesheet" href="chrome://global/content/memories-view.css" />
      <header>
        <h1 data-l10n-id="about-inference-memories-title"></h1>
        <p
          class="description"
          data-l10n-id="about-inference-memories-description"
        ></p>
      </header>
      <section>
        <h2 data-l10n-id="about-inference-memories-setup"></h2>
        ${this.#renderSetup()}
      </section>
      <section>
        <h2 data-l10n-id="about-inference-memories-results"></h2>
        ${this.#renderTiles()} ${this.#renderSteps()}
      </section>
      <section>
        <h2 data-l10n-id="about-inference-memories-inputs"></h2>
        ${this.#renderDetails()}
      </section>
    `;
  }
}

customElements.define("memories-view", MemoriesView);
