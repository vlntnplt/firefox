/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// @ts-nocheck - TODO - Remove this to type check this file.

const lazy = {};

ChromeUtils.defineLazyGetter(lazy, "console", () => {
  return console.createInstance({
    maxLogLevelPref: "browser.ml.logLevel",
    logModule: "MLTelemetry",
  });
});

ChromeUtils.defineLazyGetter(lazy, "mlUtils", () => {
  return Cc["@mozilla.org/ml-utils;1"].getService(Ci.nsIMLUtils);
});

/**
 * MLTelemetry provides a mechanism tracking a "flow" of operations
 * related to a machine learning feature. A flow is a sequence of related
 * events that represent a single, complete user-level operation, for example
 * generating a summary for a page.
 *
 * This class uses a correlation ID pattern where flowId is passed to each
 * method, allowing flexible tracking across different parts of the system.
 */
export class MLTelemetry {
  static #systemMemoryMB = Math.round(
    Services.sysinfo.getProperty("memsize") / 1024 / 1024
  );

  /** @type {string} */
  #flowId;
  /** @type {string|undefined} */
  #featureId;
  /** @type {string|undefined} */
  #hostProcess;

  /**
   * Creates a new MLTelemetry instance.
   *
   * @param {object} [options] - Configuration options.
   * @param {string | null} [options.featureId] - The identifier for the ML feature.
   * @param {string | null} [options.flowId] - An optional unique identifier for
   * this flow. If not provided, a new UUID will be generated.
   * @param {string} [options.hostProcess] - The process serving the engine,
   * "inference" or "hwinference"; see host_process in metrics.yaml.
   */
  constructor(options = {}) {
    this.#featureId = options.featureId;
    this.#flowId = options.flowId || crypto.randomUUID();
    this.#hostProcess = options.hostProcess;

    this.logEventToConsole(this.constructor, {
      featureId: this.#featureId,
      flowId: this.#flowId,
    });
  }

  /**
   * The unique identifier for this flow.
   *
   * @returns {string} The flow ID.
   */
  get flowId() {
    return this.#flowId;
  }

  /**
   * The feature identifier for this telemetry instance.
   *
   * @returns {string} The feature ID.
   */
  get featureId() {
    return this.#featureId;
  }

  /**
   * The name events record for an error: its own name, never its message.
   *
   * @param {unknown} error
   * @returns {string}
   */
  static errorName(error) {
    const name = /** @type {any} */ (error)?.name;
    return typeof name === "string" && /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(name)
      ? name
      : "Error";
  }

  /**
   * Logs a debug message to the browser console, prefixed with the flow ID.
   *
   * @param {object} caller - The calling function or class.
   * @param {object} [data] - Optional data to be JSON-stringified and logged.
   */
  logEventToConsole(caller, data) {
    const flowId = data?.flowId || this.#flowId;
    const id = flowId.substring(0, 5);
    lazy.console.debug("flowId[%s]: %s", id, caller.name, data);
  }

  /**
   * Records the engine_creation event of an attempt to create an engine.
   *
   * @param {object} options
   * @param {string} [options.flowId] - The flow ID. Uses instance flowId if not provided.
   * @param {string | null} options.engineId
   * @param {string | null} [options.modelId]
   * @param {string | null} [options.backend]
   * @param {number | null} [options.duration] - How long the attempt took, in milliseconds.
   * @param {string | null} [options.deviceKind] - Kind of the device that loaded the model.
   * @param {unknown} [options.error] - What the attempt threw, when it failed.
   */
  recordEngineCreation({
    flowId = this.#flowId,
    engineId,
    modelId = null,
    backend = null,
    duration = null,
    deviceKind = null,
    error = null,
  }) {
    const extra = {
      flow_id: flowId,
      host_process: this.#hostProcess,
      engine_id: engineId,
      model_id: modelId,
      backend,
      device_kind: deviceKind,
      duration: duration === null ? null : Math.round(duration),
      outcome: error === null ? "success" : "error",
      error: error === null ? null : MLTelemetry.errorName(error),
    };
    Glean.firefoxAiRuntime.engineCreation.record(extra);
    this.logEventToConsole(this.recordEngineCreation, extra);
  }

  /**
   * Records an engine run event.
   *
   * See firefox.ai.runtime.engine_run in toolkit/components/ml/metrics.yaml for
   * documentation on each of the parameters here.
   *
   * @param {object} options - Engine run options.
   * @param {string} options.engineId
   * @param {string | null} options.modelId
   * @param {string | null} options.backend
   * @param {number} options.beforeRun
   * @param {{cpuTime: number | null, memory: number | null}} [options.resourcesBefore]
   * @param {{cpuTime: number | null, memory: number | null}} [options.resourcesAfter]
   * @param {number | null} [options.inputTokens]
   * @param {number | null} [options.outputTokens]
   * @param {number | null} [options.characterCount]
   * @param {number | null} [options.timeToFirstChunk]
   * @param {number | null} [options.averageChunkTime]
   * @param {string | null} [options.backendSourceRevision]
   * @param {string | null} [options.deviceKind]
   * @param {string} [options.flow_id]
   * @param {string} [options.feature_id]
   * @param {unknown} [options.error] - What the run threw, when it failed.
   */
  recordEngineRun({
    engineId,
    modelId,
    backend,
    beforeRun,
    resourcesBefore,
    resourcesAfter,
    inputTokens,
    outputTokens,
    characterCount,
    timeToFirstChunk = null,
    averageChunkTime = null,
    backendSourceRevision = null,
    deviceKind = null,
    flow_id = this.#flowId,
    feature_id = this.#featureId,
    error = null,
  }) {
    const outcome = error === null ? "success" : "error";
    const errorName = error === null ? null : MLTelemetry.errorName(error);

    let cpuMilliseconds = null;
    let cpuUtilization = null;
    const wallMilliseconds = ChromeUtils.now() - beforeRun;
    const cores = lazy.mlUtils.getOptimalCPUConcurrency();
    const memoryBytes = resourcesAfter?.memory ?? null;
    if (resourcesAfter?.cpuTime != null && resourcesBefore?.cpuTime != null) {
      cpuMilliseconds = resourcesAfter.cpuTime - resourcesBefore.cpuTime;
      cpuUtilization = (cpuMilliseconds / wallMilliseconds / cores) * 100;
    }

    /**
     * Round a potentially null number, preserving null for telemetry results.
     *
     * @param {number | null} number
     */
    function round(number) {
      if (number == null) {
        return null;
      }
      return Math.round(number);
    }

    const payload = {
      flow_id,
      cpu_milliseconds: round(cpuMilliseconds),
      wall_milliseconds: round(wallMilliseconds),
      cores: round(cores),
      cpu_utilization: round(cpuUtilization),
      memory_bytes: round(memoryBytes),
      feature_id,
      outcome,
      error: errorName,
      engine_id: engineId,
      model_id: modelId,
      backend,
      backend_source_revision: backendSourceRevision,
      device_kind: deviceKind,
      // Specifically use the "||" operator since Glean expects "null" rather than
      // "undefined". When the counts are 0, this can mean nothing was generated for
      // the counts. We should count these as null.
      input_tokens: inputTokens || null,
      output_tokens: outputTokens || null,
      character_count: characterCount || null,
      time_to_first_chunk: round(timeToFirstChunk),
      average_chunk_time: round(averageChunkTime),
      system_memory_mb: MLTelemetry.#systemMemoryMB,
      host_process: this.#hostProcess,
    };

    Glean.firefoxAiRuntime.engineRun.record(payload);

    this.logEventToConsole(this.recordEngineRun, payload);
  }
}
