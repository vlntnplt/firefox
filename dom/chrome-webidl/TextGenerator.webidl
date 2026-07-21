/* -*- Mode: IDL; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * Chrome-only surface over a text generator running in the browser-keyed
 * HWInference utility process. Each TextGenerator wraps one
 * PTextGeneration actor: the model is pushed as an already-open file
 * descriptor extracted from a file-backed Blob at creation, generation
 * streams deltas through the optional callback and resolves with the
 * full result, and the generator accumulates messages until clear().
 */

enum TextGenerationRole {
  "system",
  "user",
  "assistant",
};

enum TextGenerationSamplerType {
  "logit-bias",
  "top-k",
  "top-p",
  "min-p",
  "temperature",
  "dist",
};

enum TextGenerationFinishReason {
  "eos",
  "length",
  "stop-token",
  "cancelled",
};

enum TextGenerationKVCacheDtype {
  "f32",
  "f16",
  "q8_0",
  "q5_1",
  "q5_0",
  "q4_1",
  "q4_0",
};

/**
 * Every option here is served verbatim by the generator, and this
 * dictionary is the one place the defaults are defined.
 */
dictionary TextGeneratorCreateOptions {
  unsigned long contextSize = 1024;
  /**
   * 0 resolves to the optimal CPU concurrency (the physical-core count)
   * at create; the wire always carries the resolved value. Covers both
   * generation phases unless numThreadsDecoding overrides the decode one.
   */
  unsigned long numThreads = 0;
  /**
   * Thread count for the token-by-token decode phase (llama.cpp
   * n_threads). 0 follows numThreads.
   */
  unsigned long numThreadsDecoding = 0;
  unsigned long batchSize = 2048;
  unsigned long ubatchSize = 512;
  /**
   * Data type of the K and V caches. Quantized values require flashAttn;
   * llama.cpp rejects the context otherwise and create() rejects with
   * its error.
   */
  TextGenerationKVCacheDtype kvCacheDtype = "f16";
  boolean flashAttn = false;
};

dictionary TextGenerationMessage {
  required TextGenerationRole role;
  required UTF8String content;
};

dictionary TextGenerationLogitBias {
  required long token;
  required float bias;
};

dictionary TextGenerationSampler {
  required TextGenerationSamplerType type;
  long topK = 40;
  float topP = 0.95;
  float minP = 0.05;
  float temp = 0.80;
  sequence<TextGenerationLogitBias> logitBias = [];
  unsigned long seed;
};

dictionary TextGenerationRequest {
  required sequence<TextGenerationMessage> messages;
  unsigned long maxTokens = 512;
  /**
   * Number of generated tokens batched into one delta callback.
   */
  unsigned long bufferLength = 20;
  sequence<TextGenerationSampler> samplers = [];
  sequence<long> stopTokens = [];
  boolean stopOnEndOfGenerationTokens = true;
};

[GenerateConversionToJS]
dictionary TextGenerationTimings {
  required double prefillMs;
  required double decodeMs;
};

[GenerateConversionToJS]
dictionary TextGenerationUsage {
  required unsigned long promptTokens;
  required unsigned long generatedTokens;
  required TextGenerationTimings timings;
};

[GenerateConversionToJS]
dictionary TextGenerationResult {
  required UTF8String content;
  required TextGenerationFinishReason reason;
  required TextGenerationUsage usage;
};

callback TextGenerationDeltaCallback = undefined (UTF8String text);

[ChromeOnly, Exposed=Window]
interface TextGenerator {
  /**
   * Creates a generator in the browser HWInference process, launching or
   * reusing the process. The Blob must be file-backed (ModelHub's OPFS
   * blobs are); its descriptor is what crosses to the utility process.
   * The model is loaded as part of construction: the promise resolves
   * once the load succeeded and rejects with the load error otherwise,
   * so a resolved generator always has its model.
   */
  static Promise<TextGenerator> create(Blob model,
                                       optional TextGeneratorCreateOptions options = {});

  /**
   * Appends the request's messages to the generator and generates.
   * Deltas stream through the callback while generation runs; the
   * promise resolves with the full result, or rejects with the error
   * message for runtime failures and actor death. One generation is in
   * flight at a time: a second generate() before the promise settles
   * rejects with InvalidStateError. The history is append-only and holds
   * request messages exclusively; a multi-turn caller re-sends the
   * assistant reply it received as an "assistant" message.
   */
  [Throws]
  Promise<TextGenerationResult> generate(TextGenerationRequest request,
                                         optional TextGenerationDeltaCallback onDelta);

  /**
   * Empties the accumulated message history.
   */
  undefined clear();

  /**
   * Best-effort cancel of the running generation; its promise still
   * resolves, with the tokens produced so far and reason "cancelled".
   */
  undefined cancel();

  /**
   * Destroys the underlying actor. Pending generates reject.
   */
  undefined terminate();
};
