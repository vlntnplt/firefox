/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// TextGenerator is a ChromeOnly WebIDL interface (dom/chrome-webidl).
/* global TextGenerator */

/** MLEngine-shaped engine over the HWInference utility process. */

const lazy = {};

ChromeUtils.defineESModuleGetters(lazy, {
  LLAMA_CPP_VERSION:
    "moz-src:///toolkit/components/ml/actors/MLEngineParent.sys.mjs",
  MLEngineParent:
    "moz-src:///toolkit/components/ml/actors/MLEngineParent.sys.mjs",
  MLTelemetry: "chrome://global/content/ml/MLTelemetry.sys.mjs",
  generateUUID: "chrome://global/content/ml/Utils.sys.mjs",
});

const toEngineResources = snapshot => ({
  cpuTime: snapshot.cpuTimeMs,
  memory: snapshot.memoryBytes,
});

// LlamaCppPipeline spells the float cache types fp16 and fp32; the
// generator, f16 and f32.
function toKVCacheDtype(dtype) {
  return dtype.replace(/^fp/, "f");
}

// The llama.cpp pipeline options TextGenerator.create takes. Unset ones
// are left out so the generator's defaults apply; create() rejects a
// quantized cache without flash attention rather than this quietly
// picking another.
function toCreateOptions(options) {
  const createOptions = {
    contextSize: options.numContext,
    batchSize: options.numBatch,
    ubatchSize: options.numUbatch,
    flashAttn: options.flashAttn,
    featureId: options.featureId ?? "",
  };
  if (options.kvCacheDtype) {
    createOptions.kvCacheDtype = toKVCacheDtype(options.kvCacheDtype);
  }
  if (options.numThreads) {
    createOptions.numThreads = options.numThreads;
  }
  if (options.numThreadsDecoding) {
    createOptions.numThreadsDecoding = options.numThreadsDecoding;
  }
  return createOptions;
}

function toGenerateRequest(request) {
  let prompt = request.prompt;
  if (!Array.isArray(prompt)) {
    prompt = [{ role: "user", content: String(prompt) }];
  }
  const generateRequest = {
    messages: prompt.map(message => ({
      role: message.role,
      content: message.content,
    })),
  };
  // Leave unset fields absent so TextGenerationRequest defaults apply.
  if (request.nPredict != null) {
    generateRequest.maxTokens = request.nPredict;
  }
  if (request.minOutputBufferSize != null) {
    generateRequest.bufferLength = request.minOutputBufferSize;
  }
  if (request.samplers != null) {
    generateRequest.samplers = request.samplers;
  }
  if (request.stopTokens != null) {
    generateRequest.stopTokens = request.stopTokens;
  }
  if (request.stopOnEndOfGenerationTokens != null) {
    generateRequest.stopOnEndOfGenerationTokens =
      request.stopOnEndOfGenerationTokens;
  }
  return generateRequest;
}

function toEngineMetrics(result, runTimestamps) {
  const { usage } = result;
  const outputTokens = usage.generatedTokens;
  const decodingTime = usage.timings.decodeMs;
  return {
    runTimestamps,
    inputTokens: usage.promptTokens,
    inputCharacters: usage.promptCharacters,
    outputTokens,
    inferenceTime: usage.timings.prefillMs + decodingTime,
    decodingTime,
    timeToFirstToken: usage.timings.prefillMs,
    tokensPerSecond: decodingTime
      ? outputTokens / (decodingTime / 1000)
      : undefined,
    timePerOutputToken: outputTokens ? decodingTime / outputTokens : undefined,
  };
}

/** Async-iterable sink that the delta callback pushes into. */
class ChunkQueue {
  #pending = [];
  #closed = false;
  #wakeUp = null;

  push(text) {
    this.#pending.push(text);
    this.#wakeUp?.();
  }

  close() {
    this.#closed = true;
    this.#wakeUp?.();
  }

  async *[Symbol.asyncIterator]() {
    while (!this.#closed || this.#pending.length) {
      if (!this.#pending.length) {
        await new Promise(resolve => {
          this.#wakeUp = resolve;
        });
        this.#wakeUp = null;
        continue;
      }
      yield this.#pending.shift();
    }
  }
}

/** What recordEngineRun reports about a streamed run. */
class StreamStats {
  firstChunkAt = 0;
  lastChunkAt = 0;
  chunkCount = 0;
  characterCount = 0;

  constructor() {
    this.beforeRun = ChromeUtils.now();
  }

  onChunk(text) {
    this.lastChunkAt = ChromeUtils.now();
    if (!this.firstChunkAt) {
      this.firstChunkAt = this.lastChunkAt;
    }
    this.chunkCount += 1;
    this.characterCount += text.length;
  }

  metrics(result) {
    return {
      tokenCount: result.metrics.outputTokens,
      characterCount: this.characterCount,
      timeToFirstChunk: this.firstChunkAt
        ? this.firstChunkAt - this.beforeRun
        : null,
      averageChunkTime:
        this.chunkCount > 1
          ? (this.lastChunkAt - this.firstChunkAt) / (this.chunkCount - 1)
          : null,
    };
  }
}

export class TextGenerationEngine {
  engineId;
  pipelineOptions;
  engineStatus = "uninitialized";
  notificationsCallback = null;
  telemetry;

  #generator = null;
  #initTimestamps = [];
  #inFlight = false;

  static shouldRoute(pipelineOptions) {
    if (!Services.prefs.getBoolPref("browser.ml.llama.hwInference", false)) {
      return false;
    }
    return pipelineOptions.backend === "llama.cpp";
  }

  // abortSignal stays undefined when absent: it ends up in WebIDL
  // dictionaries (StreamPipeOptions) where undefined means absent but
  // null throws.
  static async create(
    pipelineOptions,
    notificationsCallback = null,
    abortSignal = undefined
  ) {
    const engineId = pipelineOptions.engineId;
    const start = ChromeUtils.now();
    const engine = new TextGenerationEngine(
      pipelineOptions,
      notificationsCallback
    );
    try {
      await engine.#initialize(abortSignal);
      engine.telemetry.recordEngineCreationSuccessFlow({
        engineId,
        duration: ChromeUtils.now() - start,
      });
    } catch (error) {
      engine.telemetry.recordEngineCreationFailure({
        modelId: pipelineOptions.modelId,
        featureId: pipelineOptions.featureId,
        taskName: pipelineOptions.taskName,
        engineId,
        error,
      });
      throw error;
    }
    return engine;
  }

  constructor(pipelineOptions, notificationsCallback) {
    this.engineId = pipelineOptions.engineId;
    this.pipelineOptions = pipelineOptions;
    this.notificationsCallback = notificationsCallback;
    this.telemetry = new lazy.MLTelemetry({
      featureId: pipelineOptions.featureId,
      flowId: pipelineOptions.flowId,
    });
  }

  async #initialize(abortSignal) {
    abortSignal?.throwIfAborted();
    const options = this.pipelineOptions;
    this.#initTimestamps = [
      { name: "initializationStart", when: ChromeUtils.now() },
    ];

    const hub = await lazy.MLEngineParent.createModelHub({
      rootUrl: options.modelHubRootUrl,
      urlTemplate: options.modelHubUrlTemplate,
    });
    const sessionId = lazy.generateUUID();
    let modelBlob;
    try {
      [modelBlob] = await hub.getModelFileAsBlob({
        engineId: options.engineId,
        taskName: options.taskName,
        model: options.modelId,
        revision: options.modelRevision,
        file: options.modelFile,
        modelHubRootUrl: options.modelHubRootUrl,
        modelHubUrlTemplate: options.modelHubUrlTemplate,
        progressCallback: this.notificationsCallback,
        abortSignal,
        featureId: options.featureId,
        sessionId,
      });
    } finally {
      await hub.notifyModelDownloadComplete({
        engineId: options.engineId,
        model: options.modelId,
        revision: options.modelRevision,
        featureId: options.featureId,
        sessionId,
      });
    }

    const createOptions = toCreateOptions(options);
    this.#generator = await TextGenerator.create(modelBlob, createOptions);
    this.pipelineOptions.backend = "llama.cpp";
    this.#initTimestamps.push({
      name: "initializationEnd",
      when: ChromeUtils.now(),
    });
    this.engineStatus = "ready";
  }

  #assertRunnable() {
    if (this.engineStatus !== "ready") {
      throw new Error("TextGenerationEngine: engine is not ready");
    }
    if (this.#inFlight) {
      throw new Error("A generation is already in progress");
    }
  }

  async #execute(request, onDelta) {
    this.#assertRunnable();
    this.#inFlight = true;
    try {
      const runStart = ChromeUtils.now();
      this.#generator.clear();
      const result = await this.#generator.generate(
        toGenerateRequest(request),
        onDelta
      );
      const runEnd = ChromeUtils.now();
      return {
        done: true,
        finalOutput: result.content,
        ok: true,
        metrics: toEngineMetrics(result, [
          ...this.#initTimestamps,
          { name: "runStart", when: runStart },
          { name: "runEnd", when: runEnd },
        ]),
        resourcesBefore: toEngineResources(result.resources.before),
        resourcesAfter: toEngineResources(result.resources.after),
      };
    } finally {
      this.#inFlight = false;
    }
  }

  #recordRun(stats, result) {
    this.telemetry.recordRunInferenceSuccessFlow(this.engineId, result.metrics);
    this.telemetry.recordEngineRun({
      beforeRun: stats.beforeRun,
      resourcesBefore: result.resourcesBefore,
      resourcesAfter: result.resourcesAfter,
      engineId: this.engineId,
      modelId: this.pipelineOptions.modelId,
      backend: this.pipelineOptions.backend,
      backendSourceRevision: lazy.LLAMA_CPP_VERSION,
      ...(stats.streaming ? stats.metrics(result) : {}),
    });
  }

  async run(request) {
    const stats = new StreamStats();
    try {
      const result = await this.#execute(request);
      this.#recordRun(stats, result);
      return result;
    } catch (error) {
      this.telemetry.recordRunInferenceFailure(error);
      throw error;
    }
  }

  async *runWithGenerator(request) {
    const stats = new StreamStats();
    stats.streaming = true;
    const queue = new ChunkQueue();
    let completion = null;
    let settled = false;
    try {
      completion = this.#execute(request, text => {
        stats.onChunk(text);
        queue.push(text);
      });
      // Keeps the rejection handled; the await below rethrows it.
      completion.catch(() => {});
      const close = () => queue.close();
      completion.then(close, close);

      for await (const text of queue) {
        yield { text, tokens: [], isPrompt: false };
      }

      const result = await completion;
      // Link Preview keys its final flush on an empty terminal chunk.
      yield { text: "", tokens: [], isPrompt: false };

      this.#recordRun(stats, result);
      settled = true;
      return result;
    } catch (error) {
      settled = true;
      this.telemetry.recordRunInferenceFailure(error);
      throw error;
    } finally {
      // Breaking out of a `for await` loop returns this generator mid-decode.
      if (!settled) {
        this.cancel();
        await completion.catch(() => {});
      }
    }
  }

  /** Stops the generation; the run settles and the generator stays usable. */
  cancel() {
    this.#generator?.cancel();
  }

  async terminate() {
    if (this.#generator) {
      this.#generator.terminate();
      this.#generator = null;
    }
    this.engineStatus = "closed";
  }
}
