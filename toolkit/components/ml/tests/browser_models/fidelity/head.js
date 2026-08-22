/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Shared helpers for fidelity tests.
// See toolkit/components/ml/docs/fidelity.md.

const { MLEngine, MLEngineParent } = ChromeUtils.importESModule(
  "resource://gre/actors/MLEngineParent.sys.mjs"
);

const TRACE_ROOT =
  "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser_models/fidelity/";

// Every bit nsSystemInfo exposes. AMX and aarch64 dotprod/i8mm are missing;
// see the known gaps in docs/fidelity.md.
const CPU_FEATURES = [
  "hasMMX",
  "hasSSE",
  "hasSSE2",
  "hasSSE3",
  "hasSSSE3",
  "hasSSE4A",
  "hasSSE4_1",
  "hasSSE4_2",
  "hasAVX",
  "hasAVX2",
  "hasFMA3",
  "hasAVXVNNI",
  "hasAVX512F",
  "hasAVX512VNNI",
  "hasAES",
  "hasEDSP",
  "hasARMv6",
  "hasARMv7",
  "hasNEON",
];

/**
 * Loads a trace and checks it against a schema version this harness knows.
 *
 * @param {string} filename - Trace file, relative to the fidelity directory.
 * @returns {Promise<object>}
 */
async function loadTrace(filename) {
  const trace = await (await fetch(TRACE_ROOT + filename)).json();
  Assert.equal(
    trace.schema_version,
    1,
    `${filename} uses a schema version this test understands.`
  );
  return trace;
}

/**
 * Turns on run recording for the duration of the test.
 */
function recordEngineRuns() {
  MLEngine.recordRuns = true;
  registerCleanupFunction(() => {
    MLEngine.recordRuns = false;
  });
}

/**
 * Checks that the models a trace references are the ones Remote Settings
 * ships, so a model bump without a new trace fails rather than measuring a
 * pairing that no longer exists.
 *
 * @param {object} trace
 * @param {Record<string, object>} configs - Trace model role to pipeline config.
 * @returns {Promise<void>}
 */
async function assertTraceMatchesShippedModels(trace, configs) {
  for (const [role, config] of Object.entries(configs)) {
    const shipped = await MLEngineParent.getInferenceOptions(
      config.featureId,
      config.taskName
    );
    const referenced = trace.models[role];
    info(
      `${role}: Remote Settings ships ${shipped.modelId}@${shipped.modelRevision}, ` +
        `trace references ${referenced.model_id}@${referenced.revision}`
    );
    Assert.deepEqual(
      { modelId: shipped.modelId, revision: shipped.modelRevision },
      { modelId: referenced.model_id, revision: referenced.revision },
      `The ${role} model the trace references is the one that ships.`
    );
  }
}

/**
 * Builds the `mlData` string detectFields expects: own tokens, previous
 * field's prefixed "bb", next field's prefixed "aa".
 *
 * @param {{current: string, previous: string, next: string}} text
 * @returns {string}
 */
function toMlData(text) {
  const prefix = (value, marker) =>
    value
      .split(/\s+/)
      .filter(Boolean)
      .map(word => marker + word);
  return [
    ...text.current.split(/\s+/).filter(Boolean),
    ...prefix(text.previous, "bb"),
    ...prefix(text.next, "aa"),
  ].join(" ");
}

/**
 * Resolves the engines a production entry point actually ran, and fails when
 * it selected a different deployment than the test pinned. Which classifier
 * runs is decided at runtime by a pref or a Nimbus flag, so a test that names
 * its engines up front checks whichever deployment it guessed rather than the
 * one production chose.
 *
 * @param {Record<string, string>} expected - Role to engineId, each of which
 *   must have run.
 * @param {string[]} [absent] - engineIds that must not have run.
 * @returns {Record<string, object>} Role to engine.
 */
function assertDeploymentRan(expected, absent = []) {
  const engines = {};
  for (const [role, engineId] of Object.entries(expected)) {
    const engine = MLEngine.getInstance(engineId);
    Assert.ok(engine, `The ${role} engine (${engineId}) was created.`);
    Assert.greater(
      engine.recordedRuns.length,
      0,
      `The ${role} engine (${engineId}) ran.`
    );
    engines[role] = engine;
  }
  for (const engineId of absent) {
    Assert.ok(
      !MLEngine.getInstance(engineId)?.recordedRuns.length,
      `No run went to ${engineId}.`
    );
  }
  return engines;
}

/**
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @returns {number}
 */
function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

/**
 * @param {ArrayLike<number>} v
 * @returns {number}
 */
function magnitude(v) {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  return Math.sqrt(sum);
}

/**
 * Compares live embeddings against a trace's reference vectors. Cosine alone
 * is scale-invariant, so pooling applied differently than the trace records
 * only shows up in the magnitudes.
 *
 * @param {string[]} texts - The texts, in the order they were embedded.
 * @param {Array<ArrayLike<number>>} embeddings
 * @param {Map<string, number[]>} reference
 * @returns {{minCosine: number, normDelta: number}}
 */
function compareEmbeddings(texts, embeddings, reference) {
  const cosines = texts.map((text, i) =>
    cosineSimilarity(embeddings[i], reference.get(text))
  );
  const normDelta = Math.max(
    ...texts.map((text, i) => {
      const expected = magnitude(reference.get(text));
      return Math.abs(magnitude(embeddings[i]) - expected) / expected;
    })
  );
  return { minCosine: Math.min(...cosines), normDelta };
}

/**
 * Splits requested-vs-served options into keys that never reached the backend
 * and keys served as a different value.
 *
 * @param {object} requested - The config the feature asked for.
 * @param {object} served - `resolvedOptions.options` from the engine.
 * @returns {{dropped: Array<object>, transformed: Array<object>}}
 */
function diffOptions(requested, served) {
  const dropped = [];
  const transformed = [];
  for (const [key, value] of Object.entries(requested)) {
    if (value == null) {
      continue;
    }
    if (!(key in served) || served[key] == null) {
      dropped.push({ key, requested: value });
    } else if (served[key] !== value) {
      transformed.push({ key, requested: value, served: served[key] });
    }
  }
  return { dropped, transformed };
}

/**
 * Reports what each engine was asked for against what its backend served.
 *
 * @param {Record<string, {engine: object, requested: object}>} engines
 * @returns {object} Per-role config report, for the artifact.
 */
function reportEngineConfigs(engines) {
  const report = {};
  for (const [role, { engine, requested }] of Object.entries(engines)) {
    Assert.ok(engine, `The ${role} engine was created.`);
    const resolved = engine.resolvedOptions;
    Assert.ok(resolved, `The ${role} backend reported resolved options.`);
    info(`${role} resolved: ${JSON.stringify(resolved.options)}`);
    info(`${role} effective: ${JSON.stringify(resolved.effective)}`);

    if (!requested) {
      report[role] = {
        resolved: resolved.options,
        effective: resolved.effective,
      };
      continue;
    }

    const diff = diffOptions(requested, resolved.options);
    for (const entry of diff.dropped) {
      info(
        `${role} option DROPPED, it had no effect: ${entry.key}=` +
          `${JSON.stringify(entry.requested)}`
      );
    }
    for (const entry of diff.transformed) {
      info(
        `${role} option transformed: ${entry.key} ` +
          `${JSON.stringify(entry.requested)} -> ${JSON.stringify(entry.served)}`
      );
    }
    report[role] = {
      requested,
      resolved: resolved.options,
      effective: resolved.effective,
      dropped: diff.dropped,
      transformed: diff.transformed,
    };
  }
  return report;
}

/**
 * @param {ArrayLike<number>} values
 * @param {number} offset
 * @param {number} length
 * @returns {number[]} log softmax of one row
 */
function logSoftmaxRow(values, offset, length) {
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    max = Math.max(max, values[offset + i]);
  }
  let sumExp = 0;
  for (let i = 0; i < length; i++) {
    sumExp += Math.exp(values[offset + i] - max);
  }
  const logSum = max + Math.log(sumExp);
  const out = new Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = values[offset + i] - logSum;
  }
  return out;
}

/**
 * Kullback-Leibler divergence of a recorded model output from the reference,
 * in nats, over every row of the output.
 *
 * KLD compares the whole predicted distribution rather than the class that
 * happens to win, so it does not care which token or label was picked, and it
 * is in the same unit for every model.
 *
 * @param {{data: ArrayLike<number>, dims: number[]}} output
 * @param {{logits: number[], dims: number[]}} reference
 * @param {string} message
 * @returns {{kldMax: number, kldMean: number}}
 */
function compareDistributions(output, reference, message) {
  Assert.deepEqual(output.dims, reference.dims, message);
  const classes = reference.dims[reference.dims.length - 1];
  const rows = reference.logits.length / classes;

  let kldMax = 0;
  let kldSum = 0;
  for (let row = 0; row < rows; row++) {
    const offset = row * classes;
    const logP = logSoftmaxRow(reference.logits, offset, classes);
    const logQ = logSoftmaxRow(output.data, offset, classes);
    let kld = 0;
    for (let i = 0; i < classes; i++) {
      kld += Math.exp(logP[i]) * (logP[i] - logQ[i]);
    }
    kld = Math.max(kld, 0);
    kldMax = Math.max(kldMax, kld);
    kldSum += kld;
  }
  return { kldMax, kldMean: kldSum / rows };
}

/**
 * KL divergence of a live full-vocabulary output from a reference stored as
 * its top-k logits plus the full log-partition.
 *
 * Truncating to the reference's own top-k is exact for the mass that matters:
 * terms where the reference has no mass contribute nothing however far the
 * live distribution moves there, and terms where it does are all retained.
 *
 * @param {{data: ArrayLike<number>, dims: number[]}} output
 * @param {{vocab: number, top_indices: number[], top_logits: number[],
 *          log_partition: number}} reference
 * @param {string} message
 * @returns {number} divergence in nats
 */
function compareTruncatedDistribution(output, reference, message) {
  Assert.equal(output.dims[output.dims.length - 1], reference.vocab, message);
  let max = -Infinity;
  for (let i = 0; i < output.data.length; i++) {
    max = Math.max(max, output.data[i]);
  }
  let sumExp = 0;
  for (let i = 0; i < output.data.length; i++) {
    sumExp += Math.exp(output.data[i] - max);
  }
  const logPartitionQ = max + Math.log(sumExp);

  let kld = 0;
  for (let k = 0; k < reference.top_indices.length; k++) {
    const index = reference.top_indices[k];
    const logP = reference.top_logits[k] - reference.log_partition;
    const logQ = output.data[index] - logPartitionQ;
    kld += Math.exp(logP) * (logP - logQ);
  }
  return Math.max(kld, 0);
}

/**
 * Compares one run of a topic model -- an encoder-decoder that names a group
 * of tabs -- against the reference a trace holds.
 *
 * The encoder output is pooled over token positions, so the comparison does
 * not depend on the sequence length the tokenizer produced. Only the first
 * decoder step is compared: later steps are conditioned on what was generated
 * before them, so they stop being the same question as soon as two
 * configurations pick different tokens.
 *
 * @param {Array<{session: string, outputs: object}>} modelOutput
 * @param {{id: string, encoder_pooled: number[], step0: object}} example
 * @returns {{encoderCosine: number, kld: number}}
 */
function compareTopicModelRun(modelOutput, example) {
  const encoder = modelOutput.find(run => run.session === "model")?.outputs
    ?.last_hidden_state;
  const step0 = modelOutput.find(run => run.session === "decoder_model_merged")
    ?.outputs?.logits;
  Assert.ok(encoder, `${example.id}: the encoder reported its output.`);
  Assert.ok(step0, `${example.id}: the decoder reported its output.`);

  const [, seq, width] = encoder.dims;
  const pooled = new Array(width).fill(0);
  for (let t = 0; t < seq; t++) {
    for (let d = 0; d < width; d++) {
      pooled[d] += encoder.data[t * width + d] / seq;
    }
  }

  return {
    encoderCosine: cosineSimilarity(pooled, example.encoder_pooled),
    kld: compareTruncatedDistribution(
      step0,
      example.step0,
      `${example.id}: the decoder produced the reference vocabulary.`
    ),
  };
}

/**
 * The backend an engine resolved to, for labelling a measurement.
 *
 * @param {object} engine
 * @returns {string}
 */
function resolvedBackend(engine) {
  return engine?.resolvedOptions?.options?.backend ?? "unknown";
}

/**
 * @param {ArrayLike<number>} values
 * @param {number} offset
 * @param {number} length
 * @returns {number[]}
 */
function softmaxRow(values, offset, length) {
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    max = Math.max(max, values[offset + i]);
  }
  let sum = 0;
  const out = new Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = Math.exp(values[offset + i] - max);
    sum += out[i];
  }
  for (let i = 0; i < length; i++) {
    out[i] /= sum;
  }
  return out;
}

/**
 * Compares a recorded model output against the reference the trace holds.
 *
 * Both metrics are scale-free, so they read the same way for any model, unlike
 * a difference in logit space. Cosine is the shape of the output vector;
 * probability delta is how far the prediction itself moved, and is the metric
 * in the space softmax actually uses.
 *
 * @param {{data: ArrayLike<number>, dims: number[]}} output
 * @param {{logits: number[], dims: number[]}} reference
 * @param {string} message
 * @returns {{cosineMin: number, probMaxDelta: number}}
 */
function compareModelOutput(output, reference, message) {
  Assert.deepEqual(output.dims, reference.dims, message);
  const classes = reference.dims[reference.dims.length - 1];
  const rows = reference.logits.length / classes;

  let cosineMin = 1;
  let probMaxDelta = 0;
  for (let row = 0; row < rows; row++) {
    const offset = row * classes;
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < classes; i++) {
      const a = output.data[offset + i];
      const b = reference.logits[offset + i];
      dot += a * b;
      normA += a * a;
      normB += b * b;
    }
    cosineMin = Math.min(
      cosineMin,
      dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1)
    );

    const probA = softmaxRow(output.data, offset, classes);
    const probB = softmaxRow(reference.logits, offset, classes);
    for (let i = 0; i < classes; i++) {
      probMaxDelta = Math.max(probMaxDelta, Math.abs(probA[i] - probB[i]));
    }
  }
  return { cosineMin, probMaxDelta };
}

/**
 * Architecture, instruction set and core counts of the machine.
 *
 * @returns {Promise<object>}
 */
async function hostDescription() {
  const property = name => {
    try {
      return Services.sysinfo.getProperty(name);
    } catch (e) {
      return null;
    }
  };
  let cpu = {};
  try {
    cpu = await Services.sysinfo.processInfo;
  } catch (e) {
    info(`Could not read processInfo: ${e}`);
  }

  const features = {};
  for (const name of CPU_FEATURES) {
    const value = property(name);
    if (value !== null) {
      features[name] = value;
    }
  }

  return {
    os: Services.appinfo.OS,
    arch: property("arch"),
    is_nightly: AppConstants.NIGHTLY_BUILD,
    cpu: {
      name: cpu.name ?? null,
      vendor: cpu.vendor ?? null,
      family: cpu.family ?? null,
      model: cpu.model ?? null,
      stepping: cpu.stepping ?? null,
      count: cpu.count ?? null,
      cores: cpu.cores ?? null,
      pcount: cpu.pcount ?? null,
      ecount: cpu.ecount ?? null,
      l2_cache_kb: cpu.l2cacheKB ?? null,
    },
    cpu_features: features,
  };
}

/**
 * Writes a run's configuration and measurements to MOZ_UPLOAD_DIR, which
 * Taskcluster uploads as a task artifact. No-op when the variable is unset.
 *
 * @param {string} feature
 * @param {object} report
 * @returns {Promise<void>}
 */
async function writeFidelityArtifact(feature, report) {
  const full = {
    schema_version: 1,
    feature,
    host: await hostDescription(),
    ...report,
  };
  const uploadDir = Services.env.get("MOZ_UPLOAD_DIR");
  if (!uploadDir) {
    info(`Fidelity report (MOZ_UPLOAD_DIR unset): ${JSON.stringify(full)}`);
    return;
  }
  const path = PathUtils.join(uploadDir, `ml-fidelity-${feature}.json`);
  await IOUtils.writeJSON(path, full);
  info(`Wrote fidelity report to ${path}`);
}
