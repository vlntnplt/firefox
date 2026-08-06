/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// TextGenerator is a ChromeOnly WebIDL interface (dom/chrome-webidl).
/* global TextGenerator */

/** MLEngine-shaped engine over the HWInference utility process. */

const lazy = {};

ChromeUtils.defineESModuleGetters(lazy, {
  MLEngineParent: "resource://gre/actors/MLEngineParent.sys.mjs",
  MLTelemetry: "chrome://global/content/ml/MLTelemetry.sys.mjs",
});

const toEngineResources = snapshot => ({
  cpuTime: snapshot.cpuTimeMs,
  memory: snapshot.memoryBytes,
});

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

/** Per-run counters for the engine-run telemetry record. */
class RunTracker {
  settled = false;
  firstChunkAt = 0;
  chunkCount = 0;
  characterCount = 0;

  constructor(streaming) {
    this.beforeRun = ChromeUtils.now();
    this.streaming = streaming;
  }

  onChunk(text) {
    if (!this.firstChunkAt) {
      this.firstChunkAt = ChromeUtils.now();
    }
    this.chunkCount += 1;
    this.characterCount += text.length;
  }

  // tokenCount reports chunks, not tokens; see section 7 of the backlog.
  streamingMetrics() {
    const lastChunkAt = ChromeUtils.now();
    return {
      tokenCount: this.chunkCount,
      characterCount: this.characterCount,
      timeToFirstChunk: this.firstChunkAt
        ? this.firstChunkAt - this.beforeRun
        : undefined,
      averageChunkTime:
        this.chunkCount && this.firstChunkAt
          ? (lastChunkAt - this.firstChunkAt) / this.chunkCount
          : undefined,
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
    const [modelBlob] = await hub.getModelFileAsBlob({
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
    });

    // Leave unset fields absent so TextGeneratorCreateOptions defaults apply.
    const createOptions = { contextSize: options.numContext };
    if (options.numThreads) {
      createOptions.numThreads = options.numThreads;
    }
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

  #recordRun(tracker, result) {
    this.telemetry.recordRunInferenceSuccessFlow(this.engineId, result.metrics);
    this.telemetry.recordEngineRun({
      beforeRun: tracker.beforeRun,
      resourcesBefore: result.resourcesBefore,
      resourcesAfter: result.resourcesAfter,
      engineId: this.engineId,
      modelId: this.pipelineOptions.modelId,
      backend: this.pipelineOptions.backend,
      ...(tracker.streaming ? tracker.streamingMetrics() : {}),
    });
    tracker.settled = true;
  }

  async run(request) {
    const tracker = new RunTracker(false);
    try {
      const result = await this.#execute(request);
      this.#recordRun(tracker, result);
      return result;
    } catch (error) {
      this.telemetry.recordRunInferenceFailure(error);
      tracker.settled = true;
      throw error;
    }
  }

  async *runWithGenerator(request) {
    const tracker = new RunTracker(true);
    const queue = new ChunkQueue();
    let completion = null;
    try {
      completion = this.#execute(request, text => {
        tracker.onChunk(text);
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

      this.#recordRun(tracker, result);
      return result;
    } catch (error) {
      this.telemetry.recordRunInferenceFailure(error);
      tracker.settled = true;
      throw error;
    } finally {
      // Breaking out of a `for await` loop returns this generator mid-decode.
      if (!tracker.settled) {
        this.cancel();
        try {
          await completion;
        } catch (error) {
          // Ignored: the consumer already walked away.
        }
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
