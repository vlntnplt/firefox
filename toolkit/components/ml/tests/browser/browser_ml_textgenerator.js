/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/* global TextGenerator */

requestLongerTimeout(10);

// top-k:1 only filters; without a final `dist` sampler llama.cpp crashes.
const TINYSTORIES_GREEDY_SAMPLERS = [
  { type: "top-k", topK: 1 },
  { type: "dist" },
];

const TINYSTORIES_STORYTELLER_PROMPT = [
  { role: "system", content: "You are a friendly storyteller." },
  { role: "user", content: "Once upon a time there was a small mouse who" },
];

const TINYSTORIES_CONTEXT_SIZE = 512;

async function createTinyStoriesGenerator(
  options = { contextSize: TINYSTORIES_CONTEXT_SIZE }
) {
  const modelPath = getTestFilePath(
    "data/Mozilla/test-llama/main/TinyStories-656K.Q8_0.gguf"
  );
  const modelFile = await File.createFromFileName(modelPath);
  return TextGenerator.create(modelFile, options);
}

add_task(async function test_textgenerator_webidl_surface() {
  const generator = await createTinyStoriesGenerator();
  let firstPromptTokens;
  try {
    let streamed = "";
    const result = await generator.generate(
      {
        messages: TINYSTORIES_STORYTELLER_PROMPT,
        maxTokens: 16,
        bufferLength: 4,
        samplers: TINYSTORIES_GREEDY_SAMPLERS,
      },
      text => {
        streamed += text;
      }
    );
    firstPromptTokens = result.usage.promptTokens;

    info(`Generated: ${result.content}`);
    Assert.greater(result.content.length, 0, "Generation produced text");
    Assert.equal(
      streamed,
      result.content,
      "Streamed deltas join to the full content"
    );
    Assert.greater(result.usage.promptTokens, 0, "promptTokens populated");
    Assert.greater(
      result.usage.generatedTokens,
      0,
      "generatedTokens populated"
    );
    Assert.ok(
      ["eos", "length", "stop-token"].includes(result.reason),
      `Finish reason is sane (got ${result.reason})`
    );

    Assert.greaterOrEqual(
      result.resources.after.cpuTimeMs,
      result.resources.before.cpuTimeMs,
      "CPU time is a counter that only moves forward across a generation"
    );
    Assert.greater(
      result.resources.after.memoryBytes,
      0,
      "The generator process reports its own memory"
    );

    const second = await generator.generate({
      messages: TINYSTORIES_STORYTELLER_PROMPT,
      maxTokens: 8,
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
    });
    Assert.greater(
      second.usage.promptTokens,
      firstPromptTokens,
      "Second generate prefills the accumulated history"
    );

    generator.clear();
    const afterClear = await generator.generate({
      messages: TINYSTORIES_STORYTELLER_PROMPT,
      maxTokens: 8,
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
    });
    Assert.equal(
      afterClear.usage.promptTokens,
      firstPromptTokens,
      "clear() empties the history: promptTokens is back to the first " +
        "run's value"
    );
  } finally {
    generator.terminate();
  }

  await Assert.rejects(
    generator.generate({ messages: TINYSTORIES_STORYTELLER_PROMPT }),
    /terminated/,
    "Generate on a terminated generator rejects"
  );
});

// maxTokens is an unsigned long, and neither end of that range may turn into
// a budget the caller did not ask for.
add_task(async function test_textgenerator_max_tokens_bounds() {
  const generator = await createTinyStoriesGenerator();
  try {
    const none = await generator.generate({
      messages: TINYSTORIES_STORYTELLER_PROMPT,
      maxTokens: 0,
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
    });
    Assert.equal(
      none.usage.generatedTokens,
      0,
      "maxTokens: 0 generates nothing"
    );
    Assert.equal(none.content, "", "maxTokens: 0 produces no content");

    generator.clear();

    const huge = await generator.generate({
      messages: TINYSTORIES_STORYTELLER_PROMPT,
      maxTokens: 0xffffffff,
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
    });
    info(`reason=${huge.reason} generated=${huge.usage.generatedTokens}`);
    Assert.greater(
      huge.usage.generatedTokens,
      1,
      "The largest legal budget does not cap the run at a single token"
    );
    Assert.lessOrEqual(
      huge.usage.generatedTokens,
      TINYSTORIES_CONTEXT_SIZE,
      "A budget past the context is served up to the context, not beyond"
    );
  } finally {
    generator.terminate();
  }
});

add_task(async function test_textgenerator_cancel() {
  const generator = await createTinyStoriesGenerator();
  try {
    let streamed = "";
    let cancelRequested = false;
    const result = await generator.generate(
      {
        messages: TINYSTORIES_STORYTELLER_PROMPT,
        maxTokens: 512,
        bufferLength: 1,
        samplers: TINYSTORIES_GREEDY_SAMPLERS,
      },
      text => {
        streamed += text;
        if (!cancelRequested) {
          cancelRequested = true;
          generator.cancel();
        }
      }
    );

    Assert.equal(
      result.reason,
      "cancelled",
      "A cancelled generation resolves with reason 'cancelled'"
    );
    Assert.equal(
      result.content,
      streamed,
      "The cancelled result carries exactly the streamed deltas"
    );
    Assert.less(
      result.usage.generatedTokens,
      512,
      "Cancel stopped the generation before maxTokens"
    );
  } finally {
    generator.terminate();
  }
});

add_task(async function test_textgenerator_overlap_rejects() {
  const generator = await createTinyStoriesGenerator();
  try {
    const first = generator.generate({
      messages: TINYSTORIES_STORYTELLER_PROMPT,
      maxTokens: 32,
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
    });
    await Assert.rejects(
      generator.generate({
        messages: TINYSTORIES_STORYTELLER_PROMPT,
        maxTokens: 8,
        samplers: TINYSTORIES_GREEDY_SAMPLERS,
      }),
      err => err.name === "InvalidStateError" && /in flight/.test(err.message),
      "A second generate() while one is pending rejects with " +
        "InvalidStateError"
    );

    const result = await first;
    Assert.greater(
      result.content.length,
      0,
      "The first generation still resolves normally after the rejected " +
        "overlap"
    );
    Assert.ok(
      ["eos", "length", "stop-token"].includes(result.reason),
      `The first generation finishes normally (got ${result.reason})`
    );
  } finally {
    generator.terminate();
  }
});
