/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// The ChromeOnly TextGenerator interface (dom/chrome-webidl) is not in
// eslint's environment yet.
/* global TextGenerator */

/**
 * MLEngine-shaped engine over the HWInference utility process. createEngine
 * routes every llama.cpp request here: consumers keep the surface they
 * already hold -- run, runWithGenerator, terminate, engineId,
 * pipelineOptions, engineStatus -- while generation happens in the
 * sandboxed HWInference process through the TextGenerator WebIDL surface
 * instead of the inference content process.
 */

const lazy = {};

ChromeUtils.defineESModuleGetters(lazy, {
  MLEngineParent: "resource://gre/actors/MLEngineParent.sys.mjs",
  MLTelemetry: "chrome://global/content/ml/MLTelemetry.sys.mjs",
  getInferenceProcessInfo: "chrome://global/content/ml/Utils.sys.mjs",
});

/**
 * Resources of the HWInference utility process, in the shape MLEngine
 * attaches to responses ({cpuTime, memory}).
 */
async function hwInferenceProcessResources() {
  try {
    const { cpuTime, memory } =
      await lazy.getInferenceProcessInfo("hwInference");
    if (cpuTime !== undefined) {
      return { cpuTime, memory };
    }
  } catch (e) {
    // Process listing is best-effort; resource fields stay zeroed.
  }
  return { cpuTime: 0, memory: 0 };
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

  /**
   * True when createEngine should route these options here: the backend
   * is llama.cpp.
   */
  static shouldRoute(pipelineOptions) {
    if (!Services.prefs.getBoolPref("browser.ml.llama.hwInference", false)) {
      return false;
    }
    return pipelineOptions.backend === "llama.cpp";
  }

  /**
   * Constructs and initializes a fresh engine. No engineId reuse: every
   * createEngine gets its own engine, a deliberate departure from
   * MLEngineParent.getEngine, and process lifetime is the IPC manager's
   * concern, not this wrapper's.
   */
  static async create(
    pipelineOptions,
    notificationsCallback = null,
    abortSignal = null
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
      progressCallback: this.notificationsCallback ?? undefined,
      abortSignal: abortSignal ?? undefined,
      featureId: options.featureId,
    });

    // Set fields pass through; unset ones fall to the defaults of the
    // WebIDL dictionary (TextGeneratorCreateOptions), the one defaults
    // authority.
    const createOptions = { contextSize: options.numContext };
    if (options.numThreads) {
      createOptions.numThreads = options.numThreads;
    }
    this.#generator = await TextGenerator.create(modelBlob, createOptions);
    // Mirror the resolvedBackend echo of the actor path.
    this.pipelineOptions.backend = "llama.cpp";
    this.#initTimestamps.push({
      name: "initializationEnd",
      when: ChromeUtils.now(),
    });
    this.engineStatus = "ready";
  }

  /**
   * Runs one generation on the wrapped generator and builds the
   * MLEngine-shaped response. Each MLEngine run is stateless, so the
   * generator history is cleared first.
   */
  async #execute(request, onDelta) {
    if (this.engineStatus !== "ready") {
      throw new Error("TextGenerationEngine: engine is not ready");
    }
    if (this.#inFlight) {
      throw new Error("A generation is already in progress");
    }
    this.#inFlight = true;
    let prompt = request.prompt;
    if (!Array.isArray(prompt)) {
      prompt = [{ role: "user", content: String(prompt) }];
    }

    // Set fields pass through; unset ones fall to the WebIDL request
    // defaults.
    const generateRequest = {
      messages: prompt.map(message => ({
        role: message.role,
        content: message.content,
      })),
    };
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

    const runStart = ChromeUtils.now();
    this.#generator.clear();
    const result = await this.#generator.generate(
      generateRequest,
      onDelta ?? undefined
    );
    const runEnd = ChromeUtils.now();

    const { usage } = result;
    const outputTokens = usage.generatedTokens;
    const decodingTime = usage.timings.decodeMs;
    const metrics = {
      runTimestamps: [
        ...this.#initTimestamps,
        { name: "runStart", when: runStart },
        { name: "runEnd", when: runEnd },
      ],
      inputTokens: usage.promptTokens,
      outputTokens,
      inferenceTime: usage.timings.prefillMs + decodingTime,
      decodingTime,
      timeToFirstToken: usage.timings.prefillMs,
      tokensPerSecond: decodingTime
        ? outputTokens / (decodingTime / 1000)
        : undefined,
      timePerOutputToken: outputTokens
        ? decodingTime / outputTokens
        : undefined,
    };
    return { done: true, finalOutput: result.content, ok: true, metrics };
  }

  async #executeGuarded(request, onDelta) {
    try {
      return await this.#execute(request, onDelta);
    } finally {
      this.#inFlight = false;
    }
  }

  /**
   * Shared telemetry bracket for both entry points: resource snapshots
   * around the generation, success flow and engine-run records on
   * completion, failure record on rejection. Streaming runs yield their
   * chunks through this generator and add the chunk metrics.
   */
  async *#instrumentedRun(request, streaming) {
    const beforeRun = ChromeUtils.now();
    const resourcesBefore = await hwInferenceProcessResources();
    let firstChunkAt = 0;
    let tokenCount = 0;
    let characterCount = 0;

    try {
      let result;
      if (streaming) {
        const pendingChunks = [];
        let wakeUp = null;

        const onDelta = text => {
          if (!firstChunkAt) {
            firstChunkAt = ChromeUtils.now();
          }
          characterCount += text.length;
          tokenCount += 1;
          pendingChunks.push(text);
          wakeUp?.();
        };

        const completion = this.#executeGuarded(request, onDelta);
        // Surface rejections through the loop below rather than unhandled.
        completion.catch(() => {});
        let done = false;
        const markDone = () => {
          done = true;
          wakeUp?.();
        };
        completion.then(markDone, markDone);

        while (!done || pendingChunks.length) {
          if (!pendingChunks.length) {
            await new Promise(resolve => {
              wakeUp = resolve;
            });
            wakeUp = null;
            continue;
          }
          const text = pendingChunks.shift();
          yield { text, tokens: [], isPrompt: false };
        }

        result = await completion;
        // Consumers (Link Preview) key their final flush on an empty
        // terminal chunk.
        yield { text: "", tokens: [], isPrompt: false };
      } else {
        result = await this.#executeGuarded(request, null);
      }

      const resourcesAfter = await hwInferenceProcessResources();
      this.telemetry.recordRunInferenceSuccessFlow(
        this.engineId,
        result.metrics
      );
      let streamingMetrics = {};
      if (streaming) {
        const lastChunkAt = ChromeUtils.now();
        streamingMetrics = {
          tokenCount,
          characterCount,
          timeToFirstChunk: firstChunkAt ? firstChunkAt - beforeRun : undefined,
          averageChunkTime:
            tokenCount && firstChunkAt
              ? (lastChunkAt - firstChunkAt) / tokenCount
              : undefined,
        };
      }
      this.telemetry.recordEngineRun({
        beforeRun,
        resourcesBefore,
        resourcesAfter,
        engineId: this.engineId,
        modelId: this.pipelineOptions.modelId,
        backend: this.pipelineOptions.backend,
        ...streamingMetrics,
      });
      return { ...result, resourcesBefore, resourcesAfter };
    } catch (error) {
      this.telemetry.recordRunInferenceFailure(error);
      throw error;
    }
  }

  async run(request) {
    return (await this.#instrumentedRun(request, false).next()).value;
  }

  runWithGenerator(request) {
    return this.#instrumentedRun(request, true);
  }

  async terminate() {
    if (this.#generator) {
      // The manager tears the utility process down once its last
      // generator dies; nothing to book-keep here.
      this.#generator.terminate();
      this.#generator = null;
    }
    this.engineStatus = "closed";
  }
}
