/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Shared helpers for fidelity tests.
// See toolkit/components/ml/docs/fidelity.md.

const { MLEngine, MLEngineParent } = ChromeUtils.importESModule(
  "moz-src:///toolkit/components/ml/actors/MLEngineParent.sys.mjs"
);

const TRACE_ROOT =
  "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser_models/fidelity/";

// Set when validating a model that has not shipped yet: the trace is then
// allowed to name models Remote Settings does not serve. See docs/fidelity.md.
const UNPINNED = Services.env.get("MOZ_ML_FIDELITY_UNPINNED") === "1";

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
 * Every tolerance a trace declares, as dotted paths, so that one keyed by role
 * is one entry per role.
 *
 * @param {object} trace
 * @returns {string[]}
 */
function declaredTolerances(trace) {
  return Object.entries(trace.tolerances).flatMap(([key, value]) =>
    value !== null && typeof value === "object"
      ? Object.keys(value).map(role => `${key}.${role}`)
      : [key]
  );
}

/**
 * @param {object} trace
 * @param {string} path - A tolerance, as a dotted path when keyed by role.
 * @returns {?number}
 */
function toleranceAt(trace, path) {
  const value = path
    .split(".")
    .reduce((held, key) => held?.[key], trace.tolerances);
  return typeof value === "number" ? value : null;
}

/**
 * Loads a trace and checks that it holds every section and tolerance the test
 * reads.
 *
 * @param {string} filename - Trace file, relative to the fidelity directory.
 * @param {object} expected
 * @param {string[]} expected.tolerances - Every tolerance this test reads, as
 *   a dotted path when a tolerance is keyed by role, e.g. "kld_max.intent". A
 *   trace that declares a tolerance outside this list fails.
 * @returns {Promise<object>}
 */
async function loadTrace(filename, { tolerances }) {
  const response = await fetch(TRACE_ROOT + filename);
  Assert.ok(response.ok, `${filename} was found next to the test.`);
  const trace = await response.json();

  Assert.equal(
    trace.schema_version,
    1,
    `${filename} uses a schema version this test understands.`
  );
  for (const section of ["models", "reference", "tolerances", "examples"]) {
    Assert.ok(
      trace[section],
      `${filename} has a "${section}" section. See docs/fidelity.md.`
    );
  }
  for (const path of tolerances) {
    Assert.notEqual(
      toleranceAt(trace, path),
      null,
      `${filename} sets tolerances.${path}, which this test measures against.`
    );
  }
  Assert.deepEqual(
    declaredTolerances(trace).filter(path => !tolerances.includes(path)),
    [],
    `${filename} declares no tolerance the test leaves unchecked.`
  );
  return trace;
}

/**
 * Turns on run recording for the duration of the test. Must run before the
 * feature creates its engines.
 */
function recordEngineRuns() {
  MLEngine.recordRuns = true;
  registerCleanupFunction(() => {
    MLEngine.recordRuns = false;
  });
}

/**
 * Checks that the models a trace references are the ones Remote Settings
 * ships.
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
    Assert.ok(referenced, `The trace describes a "${role}" model.`);
    info(
      `${role}: Remote Settings ships ${shipped.modelId}@${shipped.modelRevision}, ` +
        `trace references ${referenced.model_id}@${referenced.revision}`
    );
    if (UNPINNED) {
      info(`MOZ_ML_FIDELITY_UNPINNED is set, not pinning the ${role} model.`);
      continue;
    }
    Assert.deepEqual(
      { modelId: shipped.modelId, revision: shipped.modelRevision },
      { modelId: referenced.model_id, revision: referenced.revision },
      `The ${role} model the trace references is the one that ships.`
    );
  }
}

/**
 * Checks that a recorded run was requested with the options the trace records
 * for that role.
 *
 * @param {object} run - A recorded run.
 * @param {object} trace
 * @param {string} role
 * @param {string} message
 */
function assertRequestOptions(run, trace, role, message) {
  const expected = trace.request_options?.[role];
  Assert.ok(expected, `The trace records request options for ${role}.`);
  const requested = {};
  for (const key of Object.keys(expected)) {
    requested[key] = run.request.options?.[key];
  }
  Assert.deepEqual(requested, expected, message);
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
 * How many runs an engine has recorded so far. Take it before driving the
 * feature and pass it to `runsSince` afterwards.
 *
 * @param {string} engineId
 * @returns {number}
 */
function runCount(engineId) {
  return MLEngine.getInstance(engineId)?.recordedRuns.length ?? 0;
}

/**
 * The runs one call to a feature produced.
 *
 * @param {string} engineId
 * @param {number} mark - A count from `runCount`, taken before the call.
 * @param {string} message
 * @param {?number} [expected] - How many runs the call should have produced.
 * @returns {Array<object>}
 */
function runsSince(engineId, mark, message, expected = null) {
  const engine = MLEngine.getInstance(engineId);
  Assert.ok(engine, `${message} (${engineId} was created.)`);
  const runs = engine.recordedRuns.slice(mark);
  if (expected !== null) {
    Assert.equal(runs.length, expected, message);
  } else {
    Assert.greater(runs.length, 0, message);
  }
  return runs;
}

/**
 * The tensor one recorded session run returned. The failure message lists the
 * sessions and outputs that were recorded.
 *
 * @param {object} run - A recorded run.
 * @param {string} message
 * @param {object} [where]
 * @param {?string} [where.session] - Which of the model's sessions, when it
 *   declares more than one. Defaults to the first one that ran.
 * @param {string} [where.key] - Which output of that session.
 * @returns {{data: ArrayLike<number>, dims: number[]}}
 */
function modelOutputOf(run, message, { session = null, key = "logits" } = {}) {
  const recorded = run?.modelOutput;
  Assert.ok(
    recorded?.length,
    `${message} (nothing was recorded: recordEngineRuns() has to run before ` +
      `the feature creates its engines, and only the ONNX pipeline records.)`
  );
  const entry = session
    ? recorded.find(item => item.session === session)
    : recorded[0];
  Assert.ok(
    entry,
    `${message} (the sessions that ran were: ` +
      `${recorded.map(item => item.session).join(", ")})`
  );
  const tensor = entry.outputs[key];
  Assert.ok(
    tensor,
    `${message} (session ${entry.session} returned: ` +
      `${Object.keys(entry.outputs).join(", ")})`
  );
  return tensor;
}

/**
 * Returns the engines a production entry point ran, and fails if an expected
 * engine did not run or an excluded one did. Which deployment runs is decided
 * at runtime by a pref or a Nimbus flag.
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
    Assert.equal(runCount(engineId), 0, `No run went to ${engineId}.`);
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
 * Compares live embeddings against the reference vectors a trace holds.
 * Cosine ignores scale; a normalization difference shows up in the
 * magnitudes.
 *
 * @param {Array<{id: string, live: ArrayLike<number>, reference: number[]}>}
 *   pairs
 * @returns {{minCosine: number, normDelta: number, worstId: ?string}}
 */
function compareEmbeddings(pairs) {
  Assert.deepEqual(
    pairs.filter(pair => !pair.live).map(pair => pair.id),
    [],
    "Every text the trace holds a vector for was embedded."
  );
  let minCosine = 1;
  let normDelta = 0;
  let worstId = null;
  for (const { id, live, reference } of pairs) {
    const cosine = cosineSimilarity(live, reference);
    if (cosine < minCosine) {
      minCosine = cosine;
      worstId = id;
    }
    const expected = magnitude(reference);
    normDelta = Math.max(
      normDelta,
      Math.abs(magnitude(live) - expected) / expected
    );
  }
  return { minCosine, normDelta, worstId };
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
  // An option can hold an object, so compare by value rather than by identity.
  const same = (a, b) =>
    a === b ||
    (a !== null &&
      b !== null &&
      typeof a === "object" &&
      typeof b === "object" &&
      JSON.stringify(a) === JSON.stringify(b));

  const dropped = [];
  const transformed = [];
  for (const [key, value] of Object.entries(requested)) {
    if (value == null) {
      continue;
    }
    if (!(key in served) || served[key] == null) {
      dropped.push({ key, requested: value });
    } else if (!same(served[key], value)) {
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
    Assert.ok(
      resolved,
      `The ${role} backend (${resolvedBackend(engine)}) reports its resolved ` +
        `options. Only the ONNX pipeline implements getResolvedOptions; see ` +
        `docs/fidelity.md.`
    );
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
 * in nats, over every row of the output. It compares whole distributions and
 * has the same unit for every model.
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
 * its top-k logits plus the full log-partition. The terms retained are exact,
 * and the terms left out have no mass in the reference and contribute
 * nothing.
 *
 * @param {{data: ArrayLike<number>, dims: number[]}} output - One row, so that
 *   the partition below is over the distribution the reference describes.
 * @param {{vocab: number, top_indices: number[], top_logits: number[],
 *          log_partition: number}} reference
 * @param {string} message
 * @returns {number} divergence in nats
 */
function compareTruncatedDistribution(output, reference, message) {
  Assert.equal(output.dims[output.dims.length - 1], reference.vocab, message);
  Assert.equal(
    output.data.length,
    reference.vocab,
    `${message} (a truncated reference describes one distribution, but the ` +
      `output holds ${output.data.length / reference.vocab} of them.)`
  );
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
 * Compares one run of an encoder-decoder model against the reference a trace
 * holds for it.
 *
 * The encoder output is averaged over positions, which makes the comparison
 * independent of the sequence length. Only the first decoder step is
 * compared; later steps depend on the tokens generated before them.
 *
 * @param {object} run - A recorded run of the engine.
 * @param {{encoder_pooled: number[], step0: object}} reference - The role's
 *   entry in `example.outputs`.
 * @param {string} id - The example, for messages.
 * @returns {{encoderCosine: number, kld: number}}
 */
function compareSeq2SeqRun(run, reference, id) {
  // transformers.js names a seq2seq encoder session "model".
  const encoder = modelOutputOf(run, `${id}: the encoder ran.`, {
    session: "model",
    key: "last_hidden_state",
  });
  const step0 = modelOutputOf(run, `${id}: the decoder ran.`, {
    session: "decoder_model_merged",
  });

  const [, seq, width] = encoder.dims;
  const pooled = new Array(width).fill(0);
  for (let t = 0; t < seq; t++) {
    for (let d = 0; d < width; d++) {
      pooled[d] += encoder.data[t * width + d] / seq;
    }
  }

  return {
    encoderCosine: cosineSimilarity(pooled, reference.encoder_pooled),
    kld: compareTruncatedDistribution(
      step0,
      reference.step0,
      `${id}: the decoder produced the reference vocabulary.`
    ),
  };
}

/**
 * Runs a whole fidelity test for a topic-model slot. The tab strip and Smart
 * Window drive the same encoder-decoder through different entry points and
 * separate Remote Settings entries.
 *
 * @param {object} slot
 * @param {string} slot.feature - Names the artifact.
 * @param {string} slot.traceFile
 * @param {string} slot.engineId
 * @param {object} slot.config - The pipeline config the feature points at the
 *   topic model.
 * @param {function(object): Promise<string>} slot.predict - Drives production
 *   for one example and returns the label it produced.
 * @returns {Promise<void>}
 */
async function runTopicModelFidelity({
  feature,
  traceFile,
  engineId,
  config,
  predict,
}) {
  const trace = await loadTrace(traceFile, {
    tolerances: ["encoder_cosine_min.topic", "step0_kld_max.topic"],
  });
  await assertTraceMatchesShippedModels(trace, { topic: config });

  let encoderCosineMin = 1;
  let kldMax = 0;
  let worstId = null;
  let resultMatches = 0;

  for (const example of trace.examples) {
    const mark = runCount(engineId);
    const label = await predict(example);
    const [run] = runsSince(
      engineId,
      mark,
      `${example.id}: naming the group took one run.`,
      1
    );

    const { encoderCosine, kld } = compareSeq2SeqRun(
      run,
      example.outputs.topic,
      example.id
    );
    encoderCosineMin = Math.min(encoderCosineMin, encoderCosine);
    if (kld > kldMax) {
      kldMax = kld;
      worstId = example.id;
    }

    if (label === example.result) {
      resultMatches++;
    } else {
      info(
        `${example.id}: label ${JSON.stringify(label)}, ` +
          `trace says ${JSON.stringify(example.result)}`
      );
    }
  }

  const engine = MLEngine.getInstance(engineId);
  const backend = resolvedBackend(engine);
  info(`Resolved backend on this machine: ${backend}`);
  info(`Worst example: ${worstId}`);

  await reportFidelity({
    feature,
    trace,
    backend,
    engines: reportEngineConfigs({ topic: { engine, requested: config } }),
    metrics: {
      "encoder_cosine_min.topic": bounded(
        trace,
        "encoder_cosine_min.topic",
        encoderCosineMin
      ),
      "step0_kld_max.topic": bounded(trace, "step0_kld_max.topic", kldMax),
      results_matched: resultMatches,
    },
  });
}

/**
 * The backend an engine resolved to, for labelling a measurement.
 *
 * @param {object} engine
 * @returns {string}
 */
function resolvedBackend(engine) {
  return (
    engine?.resolvedOptions?.options?.backend ??
    engine?.pipelineOptions?.backend ??
    "unknown"
  );
}

/**
 * A measurement checked against the tolerance of the same name. A name ending
 * in `_min` is a floor and one ending in `_max` a ceiling.
 *
 * @param {object} trace
 * @param {string} path - The tolerance, as a dotted path when keyed by role.
 * @param {number} value
 * @param {object} [options]
 * @param {string} [options.todo] - Reports the measurement as a `todo`, for a
 *   documented gap: it is expected to be outside its tolerance and shows up
 *   as an unexpected pass once the gap is closed.
 * @returns {object}
 */
function bounded(trace, path, value, { todo } = {}) {
  const bound = toleranceAt(trace, path);
  Assert.notEqual(bound, null, `The trace sets tolerances.${path}.`);
  const name = path.split(".")[0];
  const floor = name.endsWith("_min");
  Assert.ok(
    floor || name.endsWith("_max"),
    `${path} names a floor or a ceiling.`
  );
  return { value, bound, floor, todo };
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
 * Ends a fidelity test: logs the measurements, writes the report to
 * MOZ_UPLOAD_DIR, then checks each measurement against its tolerance. The
 * report is written first, and a failing run still leaves its numbers behind.
 *
 * @param {object} run
 * @param {string} run.feature - Names the artifact, `ml-fidelity-<feature>.json`.
 * @param {object} run.trace
 * @param {string} run.backend - What the engines resolved to on this machine.
 * @param {object} run.engines - From `reportEngineConfigs`.
 * @param {Record<string, object|number>} run.metrics - Each measurement,
 *   keyed by the tolerance it is checked against and built with `bounded`,
 *   or a plain number under a name of its own to report it without checking
 *   it.
 * @returns {Promise<void>}
 */
async function reportFidelity({ feature, trace, backend, engines, metrics }) {
  Assert.deepEqual(
    Object.keys(metrics)
      .filter(name => typeof metrics[name] !== "number")
      .sort(),
    declaredTolerances(trace).sort(),
    "Every tolerance the trace declares bounds a measurement."
  );

  const reported = {};
  const line = [];
  for (const [name, metric] of Object.entries(metrics)) {
    if (typeof metric === "number") {
      reported[name] = metric;
    } else {
      reported[name] = { value: metric.value, bound: metric.bound };
      if (metric.todo) {
        reported[name].todo = metric.todo;
      }
    }
    const value = typeof metric === "number" ? metric : metric.value;
    line.push(`${name}=${Number.isInteger(value) ? value : value.toFixed(6)}`);
  }
  info(`FIDELITY ${backend}: ${line.join(" ")}`);

  const report = {
    schema_version: 1,
    feature,
    host: await hostDescription(),
    trace: {
      models: trace.models,
      reference: trace.reference,
      example_count: trace.examples.length,
    },
    backend,
    engines,
    metrics: reported,
  };
  const uploadDir = Services.env.get("MOZ_UPLOAD_DIR");
  if (uploadDir) {
    const path = PathUtils.join(uploadDir, `ml-fidelity-${feature}.json`);
    await IOUtils.writeJSON(path, report);
    info(`Wrote fidelity report to ${path}`);
  } else {
    info(`Fidelity report (MOZ_UPLOAD_DIR unset): ${JSON.stringify(report)}`);
  }

  for (const [name, metric] of Object.entries(metrics)) {
    if (typeof metric === "number") {
      continue;
    }
    const { value, bound, floor } = metric;
    const within = floor ? value >= bound : value <= bound;
    const message =
      `${backend}: ${name} is within the trace tolerance ` +
      `(${value} ${floor ? ">=" : "<="} ${bound}).`;
    if (metric.todo) {
      todo(within, `${message} Known gap: ${metric.todo}`);
    } else {
      Assert.ok(within, message);
    }
  }
}
