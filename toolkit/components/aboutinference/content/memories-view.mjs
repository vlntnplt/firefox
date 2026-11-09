/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

import { html, nothing } from "chrome://global/content/vendor/lit.all.mjs";
import { MozLitElement } from "chrome://global/content/lit-utils.mjs";

const { loadMemoriesWorkload, MEMORIES_MODEL } = ChromeUtils.importESModule(
  "moz-src:///toolkit/components/aboutinference/content/MemoriesWorkload.sys.mjs"
);

const WORKLOADS = ["generate", "merge", "filter"];
const CACHE_KEY = {
  model: `${new URL(MEMORIES_MODEL.modelHubRootUrl).hostname}/${MEMORIES_MODEL.modelId}`,
  revision: MEMORIES_MODEL.modelRevision,
};

const lazy = {};
ChromeUtils.defineESModuleGetters(lazy, {
  createEngine: "chrome://global/content/ml/EngineProcess.sys.mjs",
  DownloadUtils: "resource://gre/modules/DownloadUtils.sys.mjs",
  MLTelemetry: "chrome://global/content/ml/MLTelemetry.sys.mjs",
  ModelHub: "chrome://global/content/ml/ModelHub.sys.mjs",
  ProgressStatusText: "chrome://global/content/ml/Utils.sys.mjs",
  ProgressType: "chrome://global/content/ml/Utils.sys.mjs",
});

/** On-device memories experiment. */
export class MemoriesView extends MozLitElement {
  static properties = {
    run: { state: true },
    running: { type: Boolean, state: true },
    cacheStatus: { state: true },
    cacheSize: { state: true },
    cleanup: { state: true },
  };

  #controller;
  #engine;
  #hub;
  #cacheReady;

  constructor() {
    super();
    this.run = null;
    this.running = false;
    this.cacheStatus = "checking-cache";
    this.cacheSize = null;
    this.cleanup = null;
    this.#hub = new lazy.ModelHub({
      rootUrl: MEMORIES_MODEL.modelHubRootUrl,
    });
  }

  connectedCallback() {
    super.connectedCallback();
    this.documentGlobal.addEventListener("pagehide", this.#cancel);
    this.#cacheReady = this.#readCache();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.documentGlobal.removeEventListener("pagehide", this.#cancel);
    this.#cancel();
  }

  #cancel = () => {
    this.#controller?.abort();
  };

  async #readCache() {
    try {
      const { files } = await this.#hub.listFiles(CACHE_KEY);
      const file = files.find(({ path }) => path === MEMORIES_MODEL.modelFile);
      this.cacheStatus = file ? "cached" : "needs-download";
      this.cacheSize = file?.headers.fileSize ?? null;
    } catch {
      this.cacheStatus = "cache-unavailable";
      this.cacheSize = null;
    }
  }

  get cleaning() {
    return this.cleanup?.status === "cleaning";
  }

  async #cleanUp() {
    if (this.running || this.cleaning) {
      return;
    }
    this.cleanup = { status: "cleaning" };
    try {
      await this.#cacheReady;
      await this.#hub.deleteModels({
        ...CACHE_KEY,
        filterFn: ({ file }) => file === MEMORIES_MODEL.modelFile,
        deletedBy: MEMORIES_MODEL.engineId,
      });
      this.run = null;
      this.cacheStatus = "needs-download";
      this.cacheSize = null;
      this.cleanup = { status: "done" };
    } catch (error) {
      await this.#readCache();
      this.cleanup = { status: "failed", error: String(error) };
    }
  }

  #onProgress(model, data) {
    if (data.type === lazy.ProgressType.DOWNLOAD) {
      if (data.statusText === lazy.ProgressStatusText.INITIATE) {
        model.downloadStart = ChromeUtils.now();
      } else if (data.statusText === lazy.ProgressStatusText.DONE) {
        model.downloadEnd = ChromeUtils.now();
      }
      model.loaded = data.totalLoaded ?? 0;
      model.size = data.total ?? 0;
    } else if (data.type === lazy.ProgressType.LOAD_FROM_CACHE) {
      model.cached = true;
    }
    this.requestUpdate();
  }

  async #run() {
    if (this.running || this.cleaning) {
      return;
    }
    this.running = true;
    this.cleanup = null;
    const controller = new AbortController();
    this.#controller = controller;
    controller.signal.addEventListener(
      "abort",
      () => {
        this.#engine?.cancel();
        this.requestUpdate();
      },
      { once: true }
    );
    const telemetry = new lazy.MLTelemetry({
      featureId: MEMORIES_MODEL.featureId,
    });
    const run = { status: "running", model: {}, calls: [] };
    this.run = run;
    telemetry.sessionStart({ interaction: "button_click" });
    let timeout;
    try {
      await this.#cacheReady;
      const workload = await loadMemoriesWorkload();
      controller.signal.throwIfAborted();
      const start = ChromeUtils.now();
      this.#engine = await lazy.createEngine(
        { ...MEMORIES_MODEL, flowId: telemetry.flowId },
        data => this.#onProgress(run.model, data),
        controller.signal
      );
      controller.signal.throwIfAborted();
      run.model.loadMs = ChromeUtils.now() - (run.model.downloadEnd ?? start);
      run.model.device = this.#engine.deviceKind;
      this.cacheStatus = "cached";
      timeout = setTimeout(
        () => controller.abort(new DOMException("", "TimeoutError")),
        5 * 60 * 1000
      );
      for (const { id, featureId, prompt } of workload) {
        controller.signal.throwIfAborted();
        const call = { id, featureId, status: "running" };
        run.calls.push(call);
        this.requestUpdate();
        call.result = await this.#engine.run({
          args: prompt,
          featureId,
          inferenceParams: { temperature: 0 },
        });
        controller.signal.throwIfAborted();
        call.status = "done";
      }
      run.status = "done";
    } catch (error) {
      run.status = controller.signal.aborted ? "cancelled" : "failed";
      if (controller.signal.reason?.name === "TimeoutError") {
        run.status = "timed-out";
      }
      const call = run.calls.at(-1);
      if (call?.status === "running") {
        call.status = run.status;
      }
      if (!controller.signal.aborted) {
        run.error = String(error);
      }
    } finally {
      clearTimeout(timeout);
      try {
        await this.#engine?.terminate();
      } catch (error) {
        run.status = "failed";
        run.error = String(error);
      } finally {
        this.#engine = null;
        this.#controller = null;
        telemetry.endSession(run.status);
        await this.#readCache();
        this.running = false;
        this.requestUpdate();
      }
    }
  }

  get #status() {
    if (this.cleanup) {
      return this.cleanup.status;
    }
    if (!this.running) {
      return this.run?.status ?? "idle";
    }
    if (this.#controller?.signal.aborted) {
      return "cancelling";
    }
    const call = this.run.calls.at(-1);
    if (call?.status === "running") {
      return call.id;
    }
    if (this.run.status !== "running") {
      return "finishing";
    }
    return this.run.model.downloadStart != null &&
      this.run.model.downloadEnd == null
      ? "downloading"
      : "loading";
  }

  #duration(ms) {
    return ms == null
      ? "—"
      : html`<span
          data-l10n-id="about-inference-memories-duration"
          data-l10n-args=${JSON.stringify({ seconds: ms / 1000 })}
        ></span>`;
  }

  #bytes(bytes) {
    return bytes == null ? "—" : lazy.DownloadUtils.getTransferTotal(bytes);
  }

  #renderControls() {
    return html`
      <div class="actions">
        <moz-button
          id="run"
          type="primary"
          data-l10n-id="about-inference-memories-run"
          ?disabled=${this.running || this.cleaning}
          @click=${this.#run}
        ></moz-button>
        <moz-button
          id="cancel"
          ?hidden=${!this.running}
          data-l10n-id="about-inference-memories-cancel"
          ?disabled=${!this.running || this.run?.status !== "running"}
          @click=${this.#cancel}
        ></moz-button>
        <moz-button
          id="cleanup"
          data-l10n-id="about-inference-memories-cleanup"
          ?disabled=${this.running || this.cleaning}
          @click=${this.#cleanUp}
        ></moz-button>
        <span
          id="status"
          class="badge"
          role="status"
          data-l10n-id="about-inference-memories-${this.#status}"
        ></span>
      </div>
    `;
  }

  #renderRuntime() {
    const model = this.run?.model;
    const memory = this.run?.calls[0]?.result?.resourcesBefore?.memory;
    return html`
      <section class="runtime">
        <div class="model-file">
          <span data-l10n-id="about-inference-text-generation-model"></span>
          <code>${MEMORIES_MODEL.modelFile}</code>
          <span
            class="muted"
            data-l10n-id="about-inference-memories-${this.cacheStatus}"
          ></span>
        </div>
        <dl class="runtime-metrics">
          <div>
            <dt data-l10n-id="about-inference-text-generation-device"></dt>
            <dd>${model?.device?.toUpperCase() ?? "—"}</dd>
          </div>
          <div>
            <dt data-l10n-id="about-inference-memories-memory-before"></dt>
            <dd>${this.#bytes(memory)}</dd>
          </div>
          <div>
            <dt data-l10n-id="about-inference-memories-downloaded"></dt>
            <dd>${this.#bytes(this.cacheSize ?? model?.loaded)}</dd>
          </div>
        </dl>
        ${this.#status === "downloading"
          ? html`<progress
              id="download-progress"
              aria-labelledby="status"
              value=${model.size ? model.loaded : nothing}
              max=${model.size || 1}
            ></progress>`
          : nothing}
      </section>
    `;
  }

  #tokenRate(tokens, milliseconds) {
    const rate = (tokens * 1000) / milliseconds;
    return milliseconds > 0 && Number.isFinite(rate)
      ? html`<span
          data-l10n-id="about-inference-memories-token-rate"
          data-l10n-args=${JSON.stringify({ rate })}
        ></span>`
      : "—";
  }

  #renderSummary() {
    const model = this.run?.model;
    const completed = this.run?.calls.filter(call => call.result) ?? [];
    const total = name =>
      completed.reduce((sum, call) => sum + call.result.metrics[name], 0);
    return html`
      <section class="recap">
        <h2 data-l10n-id="about-inference-memories-summary"></h2>
        <dl class="summary">
          <div>
            <dt data-l10n-id="about-inference-memories-download"></dt>
            <dd>
              ${model?.cached
                ? html`<span
                    data-l10n-id="about-inference-memories-cached"
                  ></span>`
                : this.#duration(
                    model?.downloadEnd != null
                      ? model.downloadEnd - model.downloadStart
                      : null
                  )}
            </dd>
          </div>
          <div>
            <dt data-l10n-id="about-inference-memories-load"></dt>
            <dd>${this.#duration(model?.loadMs)}</dd>
          </div>
          <div>
            <dt data-l10n-id="about-inference-memories-col-time"></dt>
            <dd>
              ${this.#duration(
                completed.length
                  ? completed.reduce(
                      (sum, call) => sum + call.result.metrics.inferenceTime,
                      0
                    )
                  : null
              )}
            </dd>
          </div>
          <div>
            <dt data-l10n-id="about-inference-memories-prefill-speed"></dt>
            <dd>
              ${this.#tokenRate(
                total("inputTokens"),
                total("timeToFirstToken")
              )}
            </dd>
          </div>
          <div>
            <dt data-l10n-id="about-inference-memories-decode-speed"></dt>
            <dd>
              ${this.#tokenRate(total("outputTokens"), total("decodingTime"))}
            </dd>
          </div>
        </dl>
      </section>
    `;
  }

  #renderCall(id) {
    const call = this.run?.calls.find(item => item.id === id);
    const status =
      call?.status ?? (this.run && !this.running ? "skipped" : "queued");
    const result = call?.result;
    return html`
      <tr data-workload=${id} data-status=${status}>
        <th scope="row" data-l10n-id="about-inference-memories-${id}"></th>
        <td data-l10n-id="about-inference-memories-${status}"></td>
        <td>${result?.metrics.inputTokens ?? "—"}</td>
        <td>${result?.metrics.outputTokens ?? "—"}</td>
        <td>${this.#duration(result?.metrics.inferenceTime)}</td>
        <td>${this.#bytes(result?.resourcesAfter?.memory)}</td>
      </tr>
    `;
  }

  #renderDetails() {
    const error = this.cleanup?.error ?? this.run?.error;
    return html`
      <details ?open=${Boolean(error)}>
        <summary data-l10n-id="about-inference-memories-details"></summary>
        ${error ? html`<p class="error">${error}</p>` : nothing}
        <dl class="details">
          <dt data-l10n-id="about-inference-text-generation-col-engine"></dt>
          <dd><code>${MEMORIES_MODEL.engineId}</code></dd>
        </dl>
        ${this.run?.calls.map(
          ({ id, featureId, result }) =>
            html` <h3 data-l10n-id="about-inference-memories-${id}"></h3>
              <dl class="details">
                <dt data-l10n-id="about-inference-memories-feature"></dt>
                <dd><code>${featureId}</code></dd>
                <dt data-l10n-id="about-inference-memories-finish-reason"></dt>
                <dd>${result?.finishReason ?? "—"}</dd>
                <dt data-l10n-id="about-inference-memories-first-token"></dt>
                <dd>${this.#duration(result?.metrics.timeToFirstToken)}</dd>
                <dt data-l10n-id="about-inference-memories-throughput"></dt>
                <dd>
                  ${result?.metrics.tokensPerSecond != null
                    ? html`<span
                        data-l10n-id="about-inference-memories-token-rate"
                        data-l10n-args=${JSON.stringify({
                          rate: result.metrics.tokensPerSecond,
                        })}
                      ></span>`
                    : "—"}
                </dd>
              </dl>`
        )}
      </details>
    `;
  }

  render() {
    return html`
      <link rel="stylesheet" href="chrome://global/content/memories-view.css" />
      <header>
        <h1 data-l10n-id="about-inference-memories-title"></h1>
        <p
          class="muted"
          data-l10n-id="about-inference-memories-description"
        ></p>
      </header>
      ${this.#renderControls()} ${this.#renderRuntime()}
      <section>
        <h2
          id="results-title"
          data-l10n-id="about-inference-memories-results"
        ></h2>
        <div class="table-scroll">
          <table aria-labelledby="results-title">
            <thead>
              <tr>
                ${[
                  "about-inference-memories-col-task",
                  "about-inference-memories-col-status",
                  "about-inference-text-generation-col-prompt-tokens",
                  "about-inference-text-generation-col-output-tokens",
                  "about-inference-memories-col-time",
                  "about-inference-memories-col-memory",
                ].map(
                  column => html`<th scope="col" data-l10n-id=${column}></th>`
                )}
              </tr>
            </thead>
            <tbody>
              ${WORKLOADS.map(id => this.#renderCall(id))}
            </tbody>
          </table>
        </div>
      </section>
      ${this.#renderSummary()} ${this.#renderDetails()}
    `;
  }
}

customElements.define("memories-view", MemoriesView);
