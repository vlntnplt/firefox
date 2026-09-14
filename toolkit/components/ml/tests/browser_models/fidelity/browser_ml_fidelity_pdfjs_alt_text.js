/* Any copyright is dedicated to the Public Domain.
   https://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// Checks the pdf.js alt text captioner against its reference trace. Drives
// PdfJsParent._mlGuess on the actor of a loaded PDF with the request the
// viewer's image editor builds: the canvas pixels of the image, at its own
// size.
// See toolkit/components/ml/docs/fidelity.md.

const { EngineProcess, FEATURES } = ChromeUtils.importESModule(
  "chrome://global/content/ml/EngineProcess.sys.mjs"
);

const { PdfJsParent } = ChromeUtils.importESModule(
  "resource://pdf.js/PdfJsParent.sys.mjs"
);

// Cold workers download ~185 MB of ONNX first.
requestLongerTimeout(10);

const PDF_URL =
  "http://mochi.test:8888/browser/toolkit/components/pdfjs/test/file_pdfjs_test.pdf";

// The service name the viewer sends with a guess request.
const IMAGE_TO_TEXT_TASK = "moz-image-to-text";

const FEATURE_ID = "pdfjs-alt-text";
const ENGINE_ID = FEATURES[FEATURE_ID].engineId;

/**
 * Decodes an image the way the pdf.js editor hands one to the model: the
 * canvas pixels, RGBA, at the image's own size.
 *
 * @param {string} dataUrl
 * @returns {Promise<{data: Uint8ClampedArray, width: number, height: number,
 *   channels: number}>}
 */
async function imageRequest(dataUrl) {
  const image = new Image();
  image.src = dataUrl;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  const { data, width, height } = context.getImageData(
    0,
    0,
    canvas.width,
    canvas.height
  );
  return { data, width, height, channels: data.length / (width * height) };
}

/**
 * @param {object} browser - A browser showing a PDF.
 * @returns {Promise<object>} Its PdfJs actor, once the viewer has created it.
 */
function pdfJsActor(browser) {
  return TestUtils.waitForCondition(() => {
    try {
      return browser.browsingContext.currentWindowGlobal.getActor("PdfJs");
    } catch (e) {
      return null;
    }
  }, "The pdf.js viewer created its actor.");
}

add_task(async function test_pdfjs_alt_text_fidelity() {
  recordEngineRuns();
  // pdf.js answers guesses with a fixed string under automation.
  PdfJsParent.stubAIEngine = false;
  registerCleanupFunction(() => {
    PdfJsParent.stubAIEngine = Cu.isInAutomation;
  });

  const trace = await loadTrace("pdfjs_alt_text_trace.json", {
    tolerances: ["encoder_cosine_min.captioner", "step0_kld_max.captioner"],
  });
  await assertTraceMatchesShippedModels(trace, {
    captioner: { featureId: FEATURE_ID, taskName: IMAGE_TO_TEXT_TASK },
  });

  try {
    await BrowserTestUtils.withNewTab(PDF_URL, async browser => {
      const actor = await pdfJsActor(browser);

      let encoderCosineMin = 1;
      let kldMax = 0;
      let worstId = null;
      let resultMatches = 0;

      for (const example of trace.examples) {
        const request = await imageRequest(example.input.image);
        const mark = runCount(ENGINE_ID);
        const response = await actor._mlGuess({
          data: { service: IMAGE_TO_TEXT_TASK, request },
        });
        Assert.ok(
          response && !response.error,
          `${example.id}: the guess produced a caption.`
        );
        const [run] = runsSince(
          ENGINE_ID,
          mark,
          `${example.id}: captioning took one run.`,
          1
        );

        const { encoderCosine, kld } = compareSeq2SeqRun(
          run,
          example.outputs.captioner,
          example.id
        );
        info(
          `${example.id} (${request.width}x${request.height}): ` +
            `encoder cosine ${encoderCosine.toFixed(6)}, ` +
            `step0 kld ${kld.toFixed(6)}`
        );
        encoderCosineMin = Math.min(encoderCosineMin, encoderCosine);
        if (kld > kldMax) {
          kldMax = kld;
          worstId = example.id;
        }

        if (response.output === example.result) {
          resultMatches++;
        } else {
          info(
            `${example.id}: caption ${JSON.stringify(response.output)}, ` +
              `trace says ${JSON.stringify(example.result)}`
          );
        }
      }

      const { captioner } = assertDeploymentRan({ captioner: ENGINE_ID });
      const backend = resolvedBackend(captioner);
      info(`Resolved backend on this machine: ${backend}`);
      info(`Worst example: ${worstId}`);

      await reportFidelity({
        feature: "pdfjs-alt-text",
        trace,
        backend,
        engines: reportEngineConfigs({
          // The requested options are read off the engine the actor created.
          captioner: {
            engine: captioner,
            requested: captioner.pipelineOptions.getOptions(),
          },
        }),
        metrics: {
          "encoder_cosine_min.captioner": bounded(
            trace,
            "encoder_cosine_min.captioner",
            encoderCosineMin
          ),
          "step0_kld_max.captioner": bounded(
            trace,
            "step0_kld_max.captioner",
            kldMax
          ),
          results_matched: resultMatches,
        },
      });
    });
  } finally {
    await EngineProcess.destroyMLEngine();
  }
});
