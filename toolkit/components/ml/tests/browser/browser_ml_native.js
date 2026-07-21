/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

/// <reference path="head.js" />

// Shared fixtures (greedy samplers, storyteller prompt, TinyStories
// options, stream drain, printableRatio) live in head.js.
const LLAMA_SMOKE_OPTIONS = tinyStoriesOptions({
  engineId: "ml-smoke-test-llama-smoke",
});

const LLAMA_SMOKE_PROMPT_B = [
  { role: "system", content: "You are a friendly storyteller." },
  { role: "user", content: "Deep in the forest, a tall green tree" },
];

async function sha256Hex(str) {
  const bytes = new TextEncoder().encode(str);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");
}

// Fraction of whitespace-split tokens that are distinct. Low values
// mean the decoder is looping ("the the the the ...").
function distinctTokenRatio(text) {
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  if (!tokens.length) {
    return 0;
  }
  return new Set(tokens).size / tokens.length;
}

/**
 * Runs a full end-to-end test on the native ONNX backend
 */
add_task(async function test_ml_smoke_test_onnx() {
  const { cleanup } = await setup();

  info("Get the engine");
  const engineInstance = await createEngine({
    taskName: "text-classification",
    modelId: "acme/bert",
    dtype: "q8",
    backend: "onnx-native",
    modelHubUrlTemplate: "{model}/resolve/{revision}",
  });
  const inferencePromise = engineInstance.run({ args: ["dummy data"] });

  const res = await inferencePromise;
  Assert.equal(res[0].label, "LABEL_0", "The text gets classified");

  await EngineProcess.destroyMLEngine();
  await cleanup();
});

async function llama_crash() {
  const { cleanup } = await setup();

  SimpleTest.expectChildProcessCrash();

  try {
    const crashMan = Services.crashmanager;
    // The crash check is process-agnostic: whichever process serves the run
    // dies abnormally. Both shutdown topics can also fire for normal
    // teardowns (with no crash information), so each arm filters for the
    // abnormal one: content sets an explicit flag, utility carries a
    // dumpID when the crash reporter generated one.
    const contentShutdown = TestUtils.topicObserved(
      "ipc:content-shutdown",
      (subject, data) => {
        info(`ipc:content-shutdown: data=${data} subject=${subject}`);
        return subject instanceof Ci.nsIPropertyBag2 && subject.get("abnormal");
      }
    );
    const utilityShutdown = TestUtils.topicObserved(
      "ipc:utility-shutdown",
      (subject, data) => {
        info(`ipc:utility-shutdown: data=${data} subject=${subject}`);
        return (
          subject instanceof Ci.nsIPropertyBag2 &&
          (!!subject.get("dumpID") || !AppConstants.MOZ_CRASHREPORTER)
        );
      }
    );
    const servingProcessGone = Promise.race([contentShutdown, utilityShutdown]);

    const engine = await createEngine({
      modelId: "Mozilla/test-llama",
      taskName: "text-classification",
      modelFile: "crash-me.gguf",
      modelRevision: "main",
      backend: "llama.cpp",
      logLevel: "Debug",
    });
    const prompt = [
      { role: "system", content: "blah" },
      {
        role: "user",
        content: "This is a test that crashes",
      },
    ];
    info("Calling runWithGenerator");
    let sawCrash = false;
    try {
      for await (const val of engine.runWithGenerator({
        prompt,
      })) {
        info(val.text);
      }
    } catch (err) {
      sawCrash = true;
      info(`failed with error ${err.message}`);

      let [subject, data] = await servingProcessGone;

      info(`serving process gone: data=${data} subject=${subject}`);

      const dumpID = subject.get("dumpID");
      if (AppConstants.MOZ_CRASHREPORTER && dumpID === null) {
        // The content path does not appear to generate minidumps, it is
        // unclear why. We should turn this into an `ok()` call once we fix
        // the underlying issue in bug 2003271.
        dump("There should be a dumpID");
      }

      if (AppConstants.MOZ_CRASHREPORTER && dumpID !== null) {
        await crashMan.ensureCrashIsPresent(dumpID);
        let minidumpDirectory = Services.dirsvc.get("ProfD", Ci.nsIFile);
        minidumpDirectory.append("minidumps");

        let dumpfile = minidumpDirectory.clone();
        dumpfile.append(dumpID + ".dmp");
        if (dumpfile.exists()) {
          info(`Removal of ${dumpfile.path}`);
          dumpfile.remove(false);
        }
        let extrafile = minidumpDirectory.clone();
        extrafile.append(dumpID + ".extra");
        info(`Removal of ${extrafile.path}`);
        if (extrafile.exists()) {
          extrafile.remove(false);
        }
        info(`cleaning up ${subject} ${data}`);
      }
    }
    Assert.ok(sawCrash, "the crash model must crash the serving process");
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
}

async function llama_works({
  prompt = [
    { role: "system", content: "blah" },
    {
      role: "user",
      content: "This is a test that works",
    },
  ],
  expectMultiChunkPrefill = false,
} = {}) {
  const { cleanup } = await setup();
  try {
    info("Create the engine for a normal run");
    const engine = await createEngine({
      taskName: "text-classification",
      modelId: "Mozilla/test-llama",
      modelFile: "TinyStories-656K.Q8_0.gguf",
      modelRevision: "main",
      backend: "llama.cpp",
      logLevel: "Debug",
    });

    const samplers = [
      {
        type: "top-k",
        topK: 3,
      },
      {
        type: "top-p",
        topP: 0.95,
      },

      {
        type: "logit-bias",
        logitBias: [{ token: 5, bias: -1000 }],
      },

      {
        type: "dist",
      },
    ];

    info("Calling runWithGenerator for normal run");
    const generator = engine.runWithGenerator({
      prompt,
      samplers,
    });
    let result;
    do {
      result = await generator.next();
      if (!result.done) {
        info(result.value.text);
      }
    } while (!result.done);

    info("Normal run worked");

    const { metrics } = result.value;
    Assert.ok(metrics, "metrics should be present on the run result");
    Assert.ok(
      Array.isArray(metrics.runTimestamps),
      "metrics.runTimestamps should be an array"
    );
    const timestampNames = metrics.runTimestamps.map(t => t.name);
    for (const name of [
      "initializationStart",
      "initializationEnd",
      "runStart",
      "runEnd",
    ]) {
      Assert.ok(
        timestampNames.includes(name),
        `metrics.runTimestamps should include ${name}`
      );
    }
    Assert.greater(metrics.inputTokens, 0, "inputTokens should be > 0");
    Assert.greater(metrics.outputTokens, 0, "outputTokens should be > 0");
    Assert.greaterOrEqual(
      metrics.inferenceTime,
      0,
      "inferenceTime should be >= 0"
    );
    Assert.greaterOrEqual(
      metrics.decodingTime,
      0,
      "decodingTime should be >= 0"
    );
    Assert.greaterOrEqual(
      metrics.timeToFirstToken,
      0,
      "timeToFirstToken should be >= 0"
    );

    if (expectMultiChunkPrefill) {
      // Default minOutputBufferSize is 20: a prompt above that exercises
      // the multi-chunk prefill path where the runner flushes prompt chunks
      // before isPhaseCompleted=true.
      Assert.greater(
        metrics.inputTokens,
        20,
        "inputTokens should exceed the default minOutputBufferSize"
      );
    }
  } finally {
    info("Destroy the engine");
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
}

async function llama_fails_with_wrong_samplers() {
  await EngineProcess.destroyMLEngine();
  await IndexedDBCache.init({ reset: true });

  const { cleanup } = await setup();
  try {
    info("Create the engine for a normal run");
    const engine = await createEngine({
      taskName: "text-classification",
      modelId: "Mozilla/test-llama",
      modelFile: "TinyStories-656K.Q8_0.gguf",
      modelRevision: "main",
      backend: "llama.cpp",
      logLevel: "Debug",
    });

    const prompt = [
      { role: "system", content: "blah" },
      {
        role: "user",
        content: "This is a test that works",
      },
    ];

    const samplers = [
      {
        type: "top-k",
        topK: 3,
      },
      {
        type: "top-p",
        topP: 0.95,
      },

      {
        type: "logit-bias",
        logitBias: [{ token: 5, bias: -1000 }],
      },

      {
        type: "dist-invalid",
      },
    ];

    info("Calling runWithGenerator for normal run with expected failure");
    const runEngine = async () => {
      await engine.run({ prompt, samplers });
    };

    // Assert the rejection names the invalid sampler, not which class
    // rejected it: the error surface is the contract, the class is an
    // implementation detail.
    await Assert.rejects(
      runEngine(),
      err => String(err?.message ?? err).includes("'dist-invalid'"),
      "The call should be rejected because it used an invalid sampler"
    );
  } finally {
    info("Destroy the engine");
    await EngineProcess.destroyMLEngine();
    await IndexedDBCache.init({ reset: true });
    await cleanup();
  }
}

/**
 * Runs a full end-to-end test on the llama.cpp backend with samplers and expected failure.
 */
add_task(async function test_ml_smoke_test_llama_fails() {
  await llama_fails_with_wrong_samplers();
});

add_task(async function test_ml_smoke_test_llama_sequential_runs() {
  const { cleanup } = await setup();
  try {
    const engine = await createEngine({
      taskName: "text-generation",
      modelId: "Mozilla/test-llama",
      modelFile: "TinyStories-656K.Q8_0.gguf",
      modelRevision: "main",
      backend: "llama.cpp",
      numContext: 128,
    });

    const request = {
      prompt: [
        { role: "system", content: "blah" },
        { role: "user", content: "Once upon a time there was" },
      ],
      nPredict: 16,
    };

    await engine.run(request);
    await engine.run(request);
    Assert.ok(true, "Two sequential run() calls completed without rejection");
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

add_task(async function test_ml_smoke_test_llama_overlap_guard() {
  const { cleanup } = await setup();
  try {
    const engine = await createEngine({
      taskName: "text-generation",
      modelId: "Mozilla/test-llama",
      modelFile: "TinyStories-656K.Q8_0.gguf",
      modelRevision: "main",
      backend: "llama.cpp",
      numContext: 128,
    });

    const request = {
      prompt: [
        { role: "system", content: "blah" },
        { role: "user", content: "Once upon a time there was" },
      ],
      nPredict: 128,
    };

    const results = await Promise.allSettled([
      engine.run(request),
      engine.run(request),
    ]);

    const rejections = results
      .filter(r => r.status === "rejected")
      .map(r => String(r.reason?.message ?? r.reason));

    Assert.ok(
      rejections.some(m => m.includes("A generation is already in progress")),
      `Expected a rejection from the LlamaRunner guard, got: ${JSON.stringify(rejections)}`
    );
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

/**
 * Runs a full end-to-end test on the llama.cpp backend with a model that loads in llama but crashes during inference.
 */
add_task(async function test_ml_smoke_test_llama_crash() {
  info("Doing a crash call");
  await llama_crash();
  info(
    "Doing a normal call after the crash to verify it's up and running again"
  );
  await llama_works();
});

/**
 * Verifies metrics are correct when the prompt exceeds the runner's
 * default minOutputBufferSize (20), forcing the prefill phase to be
 * split across multiple chunks before isPhaseCompleted is set.
 */
add_task(async function test_ml_smoke_test_llama_long_prompt_metrics() {
  await llama_works({
    prompt: [
      { role: "system", content: "You are a friendly storyteller." },
      {
        role: "user",
        content:
          "Tell me a short story about a brave little mouse who travels " +
          "across a great forest, meets many friends along the way, and " +
          "finally finds a tiny treasure chest hidden behind a waterfall " +
          "at the top of the tallest hill in the whole valley.",
      },
    ],
    expectMultiChunkPrefill: true,
  });
});

// Structural smoke for the native llama.cpp backend with greedy
// decoding. The tests above cover engine-API correctness; the tasks
// below cover output-content invariants: the generation is printable
// and diverse, different prompts produce different outputs, and the
// greedy output matches a pinned golden text + SHA-256 hash.

add_task(async function test_ml_smoke_test_llama_output_looks_like_text() {
  const { cleanup } = await setup();
  try {
    const engine = await createEngine(LLAMA_SMOKE_OPTIONS);
    try {
      const { text } = await drainGenerator(engine);
      info(`Output: ${text}`);

      Assert.greater(text.length, 0, "Generation produced text");
      Assert.notEqual(
        text.trim(),
        TINYSTORIES_STORYTELLER_PROMPT[1].content.trim(),
        "Output is not a verbatim echo of the user prompt"
      );

      const pr = printableRatio(text);
      info(`Printable-ASCII ratio: ${pr.toFixed(3)}`);
      Assert.greater(
        pr,
        0.9,
        `Output should be mostly printable text (got ${pr.toFixed(3)})`
      );

      const dr = distinctTokenRatio(text);
      info(`Distinct-token ratio: ${dr.toFixed(3)}`);
      Assert.greater(
        dr,
        0.3,
        `Output should not be a degenerate loop (got ${dr.toFixed(3)})`
      );
    } finally {
      await engine.terminate?.();
    }
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

add_task(async function test_ml_smoke_test_llama_prompt_sensitive() {
  const { cleanup } = await setup();
  try {
    const engine = await createEngine(LLAMA_SMOKE_OPTIONS);
    try {
      const { text: a } = await drainGenerator(engine);
      const { text: b } = await drainGenerator(engine, {
        prompt: LLAMA_SMOKE_PROMPT_B,
      });
      info(`Prompt A output: ${a}`);
      info(`Prompt B output: ${b}`);
      Assert.notEqual(
        a,
        b,
        "Different prompts should produce different greedy outputs"
      );
    } finally {
      await engine.terminate?.();
    }
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

// Golden-output check. Catches silent changes in token selection
// (quantization drift, attention bug, upstream llama.cpp roll).
//
// Greedy outputs diverge by CPU architecture: different SIMD vector
// widths in llama.cpp's matmul/attention kernels (NEON on aarch64 vs
// AVX on x86_64) shift logits enough to flip argmax on a handful of
// tokens, so the constants dispatch on the CPU architecture. macOS
// Intel 10.15 is a third bucket and exhibits within-engine
// non-determinism for TinyStories, so the golden assertion
// runtime-skips there. Re-pin both text and hash from a try push
// when a llama.cpp roll legitimately changes outputs.
const LLAMA_SMOKE_IS_AARCH64 =
  Services.sysinfo.getProperty("arch") === "aarch64";
const LLAMA_SMOKE_EXPECTED_TEXT = LLAMA_SMOKE_IS_AARCH64
  ? "Suddenly, the mouse stopped! \nThe little mouse was scared, but he was scared. He had been so brave. He was scared, but he was still wearing his "
  : "Suddenly, the mouse stopped! \nThe old lady was surprised to see a beautiful song. The old lady was so surprised. She had never seen a little mouse and ";
const LLAMA_SMOKE_EXPECTED_HASH = LLAMA_SMOKE_IS_AARCH64
  ? "a59803f3d1c9d983f14c54fadbe9de3b73a1053780f01bc7768fa2bc498f6005"
  : "67b4cf2e37139e20795938ab3dedbdd040ffc2143e2ded210b0838de37581fa9";

// Cross-mode invariant: run() must return exactly the text that
// runWithGenerator() streams for the same greedy request. The future
// engine's reply carries the full content alongside the streamed
// deltas, so this equality is the contract that lets a caller ignore
// the stream.
add_task(async function test_ml_smoke_test_llama_run_matches_generator() {
  const { cleanup } = await setup();
  try {
    const engine = await createEngine(LLAMA_SMOKE_OPTIONS);
    try {
      const { text: streamed } = await drainGenerator(engine);
      const res = await engine.run({
        prompt: TINYSTORIES_STORYTELLER_PROMPT,
        samplers: TINYSTORIES_GREEDY_SAMPLERS,
        nPredict: 32,
      });
      info(`Streamed: ${streamed}`);
      info(`run() finalOutput: ${res.finalOutput}`);
      Assert.equal(
        res.finalOutput,
        streamed,
        "run() returns the same greedy text runWithGenerator() streams"
      );
    } finally {
      await engine.terminate?.();
    }
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

// Statelessness invariant: two identical greedy run() calls on one warm
// engine produce identical text, i.e. one run leaves no context behind
// for the next. An engine with session state must reset it between
// runs to keep this holding.
add_task(async function test_ml_smoke_test_llama_runs_are_stateless() {
  const { cleanup } = await setup();
  try {
    const engine = await createEngine(LLAMA_SMOKE_OPTIONS);
    try {
      const request = {
        prompt: TINYSTORIES_STORYTELLER_PROMPT,
        samplers: TINYSTORIES_GREEDY_SAMPLERS,
        nPredict: 32,
      };
      const first = await engine.run(request);
      const second = await engine.run(request);
      info(`First: ${first.finalOutput}`);
      info(`Second: ${second.finalOutput}`);
      Assert.equal(
        second.finalOutput,
        first.finalOutput,
        "Identical greedy runs on a warm engine produce identical text"
      );
    } finally {
      await engine.terminate?.();
    }
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});

add_task(async function test_ml_smoke_test_llama_golden_text() {
  const isMacIntel =
    AppConstants.platform === "macosx" &&
    Services.sysinfo.getProperty("arch") !== "aarch64";
  if (isMacIntel) {
    ok(
      true,
      "Skipping golden-text on macOS Intel 10.15: TinyStories greedy " +
        "output is in a third arch bucket and within-engine determinism " +
        "is unreliable there (Bug 2047025)."
    );
    return;
  }

  const { cleanup } = await setup();
  try {
    const engine = await createEngine(LLAMA_SMOKE_OPTIONS);
    try {
      const { text } = await drainGenerator(engine);
      const hash = await sha256Hex(text);
      info(`Greedy text: ${text}`);
      info(`Greedy SHA-256: ${hash}`);
      Assert.equal(
        text,
        LLAMA_SMOKE_EXPECTED_TEXT,
        "Greedy output matches the pinned golden text"
      );
      Assert.equal(
        hash,
        LLAMA_SMOKE_EXPECTED_HASH,
        "Greedy output hash matches the pinned golden hash"
      );
    } finally {
      await engine.terminate?.();
    }
  } finally {
    await EngineProcess.destroyMLEngine();
    await cleanup();
  }
});
