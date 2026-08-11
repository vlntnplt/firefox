/* Any copyright is dedicated to the Public Domain.
http://creativecommons.org/publicdomain/zero/1.0/ */
"use strict";

// Latency test for the PDF.js alt-text generator
// (toolkit/components/pdfjs/content/PdfjsParent.sys.mjs).
//
// In production this feature requests "onnx-native" with NO fallback: if the
// native runtime is missing, #createAIEngine logs and returns null and the
// feature silently does nothing. Measuring it on both backends is therefore
// not just a perf question -- the wasm numbers describe the experience this
// feature would have if it were given a fallback, which is the open question.
//
// The image is synthetic. distilvit's processor resizes to a fixed input
// resolution before inference, so the source dimensions and pixel values do
// not affect latency materially; a deterministic buffer keeps the test
// hermetic and reproducible.

const IMAGE_WIDTH = 512;
const IMAGE_HEIGHT = 512;
const CHANNELS = 4;

/**
 * Builds a deterministic RGBA buffer. imageToText() in ONNXPipeline accepts
 * either a `url` or a raw `{data, width, height, channels}` triple; the raw
 * form avoids needing an image fixture or any I/O in the measured path.
 *
 * @returns {Uint8ClampedArray}
 */
function buildImage() {
  const data = new Uint8ClampedArray(IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS);
  for (let y = 0; y < IMAGE_HEIGHT; y++) {
    for (let x = 0; x < IMAGE_WIDTH; x++) {
      const i = (y * IMAGE_WIDTH + x) * CHANNELS;
      data[i] = (x * 7) % 256;
      data[i + 1] = (y * 11) % 256;
      data[i + 2] = (x + y) % 256;
      data[i + 3] = 255;
    }
  }
  return data;
}

const perfMetadata = {
  owner: "GenAI Team",
  name: "browser_ml_pdfjs_alt_text_perf.js",
  description: "Latency for the PDF.js alt-text (image-to-text) model",
  options: {
    default: {
      perfherder: true,
      perfherder_metrics: [
        {
          name: "latency",
          unit: "ms",
          shouldAlert: false,
        },
        {
          name: "memory",
          unit: "MiB",
          shouldAlert: false,
        },
        {
          name: "tokenSpeed",
          unit: "tokens/s",
          shouldAlert: false,
          lowerIsBetter: false,
        },
        {
          name: "charactersSpeed",
          unit: "chars/s",
          shouldAlert: false,
          lowerIsBetter: false,
        },
      ],
      verbose: true,
      manifest: "perftest.toml",
      manifest_flavor: "browser-chrome",
      try_platform: ["linux", "mac", "win"],
    },
  },
};

requestLongerTimeout(20);

add_task(async function test_ml_pdfjs_alt_text() {
  const options = new PipelineOptions({
    taskName: "moz-image-to-text",
    featureId: "pdfjs-alt-text",
    engineId: "pdfjs",
    // Lowercase, matching the moz-image-to-text engine configuration: the
    // tokenizer and processor ids default to it, and the model hub is served
    // from a case-sensitive filesystem in CI.
    modelId: "mozilla/distilvit",
    modelHubUrlTemplate: "{model}/{revision}",
    modelRevision: "main",
    dtype: "q8",
    timeoutMS: -1,
  });

  const request = {
    data: buildImage(),
    width: IMAGE_WIDTH,
    height: IMAGE_HEIGHT,
    channels: CHANNELS,
  };

  // Alt text is generated on demand when the user opens the alt-text dialog,
  // so the engine is usually cold on first use.
  await runMLPerfTest({
    name: "pdfjs-alt-text",
    options,
    request,
    iterations: 2,
    addColdStart: true,
  });
});
