/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/* global HWInference */

const { BACKENDS, EngineProcess, PipelineOptions, createEngine } =
  ChromeUtils.importESModule(
    "chrome://global/content/ml/EngineProcess.sys.mjs"
  );

const { runOnBothInferenceProcesses } = ChromeUtils.importESModule(
  "resource://testing-common/MLTestUtils.sys.mjs"
).MLTestUtils;

// CI sets this on workers that have an accelerator llama.cpp can use.
const EXPECT_ACCELERATOR =
  Services.env.get("MOZ_ML_EXPECT_ACCELERATOR") === "1";

// A cold-start CI worker has to fetch a ~386 MB GGUF before inference can run,
// so allow more than the default 45 s browser-chrome per-test timeout.
requestLongerTimeout(4);

async function runChat(device) {
  const engine = await createEngine(
    new PipelineOptions({
      engineId: "smollm2-chat-e2e",
      taskName: "text-generation",
      backend: BACKENDS.llamaCpp,
      modelId: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
      modelRevision: "main",
      modelHubUrlTemplate: "{model}/{revision}",
      modelHubRootUrl: "https://model-hub.mozilla.org/",
      modelFile: "smollm2-360m-instruct-q8_0.gguf",
      device,
    })
  );
  info("Engine created");

  const prompt = [
    {
      role: "system",
      content: "You are a helpful assistant. Answer briefly.",
    },
    {
      role: "user",
      content: "What color is a clear daytime sky? Answer in one word.",
    },
  ];
  const request = { prompt, nPredict: 24 };

  let text = "";
  let metrics;

  try {
    const generator = engine.runWithGenerator(request);
    let result;
    do {
      result = await generator.next();
      if (result.done) {
        metrics = result.value?.metrics;
      } else if (!result.value.isPrompt) {
        text += result.value.text ?? "";
      }
    } while (!result.done);

    info(`SmolLM2 chat output: ${text.trim()}`);
    info(`SmolLM2 chat metrics: ${JSON.stringify(metrics)}`);

    Assert.greater(text.trim().length, 0, "Model produced non-empty output.");
    Assert.ok(metrics, "The run returned metrics.");
    Assert.greater(metrics.outputTokens, 0, "Real tokens were decoded.");
    const promptText = prompt.map(m => m.content).join(" ");
    Assert.notEqual(
      text.trim(),
      promptText,
      "Output is not just the prompt echoed."
    );
  } finally {
    await engine.terminate();
    await EngineProcess.destroyMLEngine();
  }
}

add_task(async function test_smollm2_real_chat_generation() {
  info(`MOZ_ML_EXPECT_ACCELERATOR: ${EXPECT_ACCELERATOR}`);
  await runOnBothInferenceProcesses(async hwInference => {
    const placements = [null];
    if (hwInference) {
      info(`Devices: ${JSON.stringify(await HWInference.listDevices())}`);
      if (EXPECT_ACCELERATOR) {
        placements.push({ kinds: ["gpu"] });
      }
    }
    for (const device of placements) {
      info(
        `Chat in the ${hwInference ? "HWInference" : "inference content"} ` +
          `process, placement: ${JSON.stringify(device)}`
      );
      await runChat(device);
    }
  });
});

const MEMORIES_MERGE_SCHEMA = {
  type: "array",
  items: {
    type: "object",
    additionalProperties: false,
    required: ["component_statements", "new_reasoning", "new_statement"],
    properties: {
      component_statements: {
        type: "array",
        minItems: 2,
        items: { type: "string" },
      },
      new_reasoning: { type: "string" },
      new_statement: { type: "string", maxLength: 100 },
    },
  },
};

// Only the HWInference engine honours response_format.
add_task(async function test_smollm2_json_schema() {
  await SpecialPowers.pushPrefEnv({
    set: [["browser.ml.llama.hwInference", true]],
  });
  const engine = await createEngine(
    new PipelineOptions({
      engineId: "smollm2-json-schema",
      taskName: "text-generation",
      backend: BACKENDS.llamaCpp,
      modelId: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
      modelRevision: "main",
      modelHubUrlTemplate: "{model}/{revision}",
      modelHubRootUrl: "https://model-hub.mozilla.org/",
      modelFile: "smollm2-360m-instruct-q8_0.gguf",
      numContext: 1024,
    })
  );
  try {
    const merged = await engine.run({
      args: [
        {
          role: "system",
          content:
            "You merge related statements about a user into one. Return JSON only.",
        },
        {
          role: "user",
          content:
            "Merge these statements into exactly one entry:\n" +
            "- Statement: Shops for trail running shoes\n" +
            "- Statement: Compares hiking boots online",
        },
      ],
      inferenceParams: {
        temperature: 0,
        max_completion_tokens: 256,
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "MemoriesMergeCandidates",
            schema: MEMORIES_MERGE_SCHEMA,
          },
        },
      },
    });
    info(`SmolLM2 merge output: ${merged.finalOutput}`);
    const candidates = JSON.parse(merged.finalOutput);
    Assert.ok(Array.isArray(candidates), "The output is a JSON array");
    for (const candidate of candidates) {
      Assert.deepEqual(
        Object.keys(candidate).sort(),
        ["component_statements", "new_reasoning", "new_statement"],
        "Each candidate has exactly the required keys"
      );
      Assert.greaterOrEqual(
        candidate.component_statements.length,
        2,
        "minItems holds"
      );
      Assert.lessOrEqual(
        candidate.new_statement.length,
        100,
        "maxLength holds"
      );
    }

    const object = await engine.run({
      args: [
        {
          role: "user",
          content:
            'Reply with a JSON object with one key "color" naming the sky.',
        },
      ],
      inferenceParams: {
        temperature: 0,
        max_completion_tokens: 64,
        response_format: { type: "json_object" },
      },
    });
    info(`SmolLM2 json_object output: ${object.finalOutput}`);
    const parsed = JSON.parse(object.finalOutput);
    Assert.equal(typeof parsed, "object", "json_object yields a JSON object");
  } finally {
    await engine.terminate();
    await EngineProcess.destroyMLEngine();
  }
});
