/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// The ChromeOnly TextGenerator interface (dom/chrome-webidl) is not in
// eslint's environment yet.
/* global TextGenerator */

// Direct chrome-JS drive of the TextGenerator WebIDL surface: the
// in-tree TinyStories model as a file-backed File, a generator in the
// HWInference utility process, greedy generation with streamed deltas.
// The MLEngine integration (createEngine routing) is tested separately;
// this covers the binding itself. The greedy samplers and storyteller
// prompt come from head.js.

requestLongerTimeout(120);

async function createTinyStoriesGenerator(options = { contextSize: 512 }) {
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
