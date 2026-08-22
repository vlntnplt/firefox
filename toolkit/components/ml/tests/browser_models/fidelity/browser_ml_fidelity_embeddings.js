/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the contextual text embedder against its reference trace. Every
// EmbeddingsGenerator consumer shares this engine.
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

  const trace = await loadTrace("embeddings_trace.json", {
    tolerances: [
      "embedding_cosine_min.encoder",
      "embedding_norm_rel_max.encoder",
    ],
  });
  const generator = embeddingsGeneratorFactory.forGeneral();
  await assertTraceMatchesShippedModels(trace, { encoder: generator.options });

  const texts = trace.examples.map(example => example.input.text);

  try {
    const vectors = await generator.embedMany(texts);

    // embedMany created the engine, and this is its only run.
    const { engineId } = generator.options;
    const engine = MLEngine.getInstance(engineId);
    const backend = resolvedBackend(engine);
    info(`Resolved backend on this machine: ${backend}`);

    const engines = reportEngineConfigs({
      encoder: { engine, requested: generator.options },
    });

    const [run] = runsSince(
      engineId,
      0,
      "embedMany embedded every text in a single run.",
      1
    );
    assertRequestOptions(
      run,
      trace,
      "encoder",
      "The embedder was run with the pooling the trace records."
    );

    Assert.equal(
      vectors.length,
      texts.length,
      "One vector came back per text."
    );
    const { minCosine, normDelta, worstId } = compareEmbeddings(
      trace.examples.map((example, i) => ({
        id: example.id,
        live: vectors[i],
        reference: example.outputs.encoder,
      }))
    );
    info(`Worst example: ${worstId}`);

    // onnx-native ignores the normalize option (docs/fidelity.md, known
    // gaps); the magnitude check is a todo there.
    await reportFidelity({
      feature: "embeddings",
      trace,
      backend,
      engines,
      metrics: {
        "embedding_cosine_min.encoder": bounded(
          trace,
          "embedding_cosine_min.encoder",
          minCosine
        ),
        "embedding_norm_rel_max.encoder": bounded(
          trace,
          "embedding_norm_rel_max.encoder",
          normDelta,
          backend === "onnx-native"
            ? { todo: "onnx-native ignores normalize, see docs/fidelity.md." }
            : {}
        ),
      },
    });
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
