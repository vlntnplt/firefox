/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Unstubbed Link Preview generation. The sibling tests stub createEngine
// or generateTextAI to describe the feature's control flow; this one
// drives LinkPreviewModel.generateTextAI end-to-end -- real createEngine,
// real llama.cpp engine, the in-tree TinyStories model served over the
// chrome:// model hub -- so it fails if the engine behind Link Preview
// stops producing text. The browser.ml.linkPreview.config pref is the
// feature's own override hook (spread last onto its PipelineOptions), so
// the redirection exercises the exact options path shipping code uses.

// Loads setup() and the ML remote-settings mocks; the ML harness itself
// loads helpers cross-directory this way.
Services.scriptloader.loadSubScript(
  "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/head.js",
  this
);

const { LinkPreviewModel } = ChromeUtils.importESModule(
  "moz-src:///browser/components/genai/LinkPreviewModel.sys.mjs"
);

requestLongerTimeout(120);

const GENERATION_CONFIG = {
  backend: "llama.cpp",
  modelId: "Mozilla/test-llama",
  modelFile: "TinyStories-656K.Q8_0.gguf",
  modelRevision: "main",
  taskName: "text-generation",
  modelHubRootUrl:
    "chrome://mochitests/content/browser/toolkit/components/ml/tests/browser/data",
  modelHubUrlTemplate: "{model}/{revision}",
  numContext: 512,
};

const ARTICLE_TEXT =
  "Once upon a time there was a small mouse who lived in a cozy house. " +
  "Every day the mouse would explore the great green forest nearby. " +
  "One morning it found a shiny treasure hidden behind a waterfall. " +
  "The mouse carried the treasure home and shared it with all its friends. " +
  "From that day on, the little house was the happiest place in the valley.";

add_task(async function test_link_preview_real_generation() {
  const { cleanup } = await setup();
  try {
    await SpecialPowers.pushPrefEnv({
      set: [
        ["browser.ml.linkPreview.blockListEnabled", false],
        ["browser.ml.linkPreview.config", JSON.stringify(GENERATION_CONFIG)],
      ],
    });

    const sentences = [];
    const errors = [];
    await LinkPreviewModel.generateTextAI(ARTICLE_TEXT, {
      onText: sentence => sentences.push(sentence),
      onError: error => errors.push(error),
    });

    Assert.deepEqual(
      errors.map(e => String(e?.message ?? e)),
      [],
      "Generation completed without errors"
    );
    Assert.greater(sentences.length, 0, "Generation produced key points");

    const text = sentences.join(" ");
    info(`Generated key points: ${text}`);
    // printableRatio comes from the ML head.js loaded above.
    const ratio = printableRatio(text);
    Assert.greater(
      ratio,
      0.9,
      `Generated text should be mostly printable (got ${ratio.toFixed(3)})`
    );
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});
