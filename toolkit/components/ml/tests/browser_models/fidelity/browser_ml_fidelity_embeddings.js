/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the contextual text embedder against its reference trace. Every
// EmbeddingsGenerator consumer shares this engine, so one test covers them all.
// See toolkit/components/ml/docs/fidelity.md.

const { EngineProcess } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);

const { embeddingsGeneratorFactory } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EmbeddingsGenerator.sys.mjs"
);

// Cold workers download ~23 MB of ONNX first.
requestLongerTimeout(4);

add_task(async function test_embeddings_fidelity() {
  recordEngineRuns();

  const trace = await loadTrace("embeddings_trace.json");
  const generator = embeddingsGeneratorFactory.forGeneral();
  await assertTraceMatchesShippedModels(trace, { encoder: generator.options });

  const texts = trace.embeddings.map(entry => entry.text);
  const reference = new Map(
    trace.embeddings.map(entry => [entry.text, entry.vector])
  );

  try {
    const vectors = await generator.embedMany(texts);

    const engine = MLEngine.getInstance(generator.options.engineId);
    const backend = resolvedBackend(engine);
    info(`Resolved backend on this machine: ${backend}`);

    const engines = reportEngineConfigs({
      encoder: { engine, requested: generator.options },
    });

    Assert.equal(
      engine.recordedRuns.length,
      1,
      "embedMany embedded every text in a single run."
    );
    const [run] = engine.recordedRuns;
    const { pooling, normalize, max_length } = trace.preprocessing;
    Assert.deepEqual(
      {
        pooling: run.request.options?.pooling,
        normalize: run.request.options?.normalize,
        max_length: run.request.options?.max_length,
      },
      { pooling, normalize, max_length },
      "The embedder was run with the pooling the trace records."
    );

    Assert.equal(
      vectors.length,
      texts.length,
      "One vector came back per text."
    );
    const { minCosine, normDelta } = compareEmbeddings(
      texts,
      vectors,
      reference
    );

    await writeFidelityArtifact("embeddings", {
      trace: { models: trace.models, text_count: texts.length },
      backend,
      engines,
      metrics: {
        embedding_cosine_min: minCosine,
        embedding_norm_rel_delta: normDelta,
      },
      tolerances: trace.tolerances,
    });

    info(
      `FIDELITY ${backend}: embedding-cosine-min=${minCosine.toFixed(6)} ` +
        `embedding-norm-rel-delta=${normDelta.toFixed(5)}`
    );

    Assert.greaterOrEqual(
      minCosine,
      trace.tolerances.embedding_cosine_min,
      `${backend}: embedding cosine meets the trace tolerance.`
    );
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
