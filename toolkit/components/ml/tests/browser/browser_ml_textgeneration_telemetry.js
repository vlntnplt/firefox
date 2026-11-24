/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

// One task per telemetry contract of the HWInference text generation path:
// what a creation, a run and a stream record in firefox.ai.runtime, then what
// each failure mode records. Creation failures record the error name, run
// failures the message, and each mode must come out as its own value.
//
// Not covered, for lack of an injection point: the process refusing to launch,
// the process dying before the model load reports, and the backend refusing a
// model (llama.cpp aborts on an unreadable file instead of reporting).

const HWI_PREFS = [["browser.ml.llama.hwInference", true]];

const { LLAMA_CPP_VERSION } = ChromeUtils.importESModule(
  "moz-src:///toolkit/components/ml/actors/MLEngineParent.sys.mjs"
);

const RUN_REQUEST = {
  prompt: TINYSTORIES_STORYTELLER_PROMPT,
  samplers: TINYSTORIES_GREEDY_SAMPLERS,
  nPredict: 16,
};

function events(name) {
  return Glean.firefoxAiRuntime[name].testGetValue() ?? [];
}

function only(name) {
  const all = events(name);
  Assert.equal(all.length, 1, `Exactly one ${name} event`);
  return all[0].extra;
}

function failedCreations() {
  return events("engineCreation")
    .map(event => event.extra)
    .filter(creation => creation.outcome === "error");
}

function unsuccessfulRuns() {
  return events("engineRun").filter(event => event.extra.outcome !== "success");
}

// crash-me.gguf takes the process down mid-generation. The harness counts
// the minidump as a failure unless the test claims it, as browser_ml_native
// does for the same model.
async function removeCrashDump(shutdown) {
  const [subject] = await shutdown;
  const props = subject.QueryInterface(Ci.nsIPropertyBag2);
  if (!props.hasKey("dumpID")) {
    return;
  }
  const dumpID = props.getPropertyAsAString("dumpID");
  await Services.crashmanager.ensureCrashIsPresent(dumpID);
  const minidumps = Services.dirsvc.get("ProfD", Ci.nsIFile);
  minidumps.append("minidumps");
  for (const suffix of [".dmp", ".extra"]) {
    const file = minidumps.clone();
    file.append(dumpID + suffix);
    if (file.exists()) {
      file.remove(false);
    }
  }
}

add_task(async function test_creation_records_success_and_download() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  await IndexedDBCache.init({ reset: true });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-create",
      featureId: "link-preview",
    });
    await engine.terminate();

    const created = only("engineCreation");
    Assert.equal(created.outcome, "success");
    Assert.equal(created.engine_id, "hwi-tel-create");
    Assert.equal(created.host_process, "hwinference");
    Assert.equal(
      created.device_kind,
      "cpu",
      "The creation records the device kind"
    );
    Assert.greaterOrEqual(Number(created.duration), 0);

    const download = only("modelDownload");
    Assert.equal(download.outcome, "success", "The model download succeeded");
    Assert.equal(Number(download.files), 1);
    Assert.equal(Number(download.failed_files), 0);
  } finally {
    await cleanup();
  }
});

add_task(async function test_run_records_the_inference_flow() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-run",
      featureId: "link-preview",
    });
    const { metrics } = await engine.run(RUN_REQUEST);
    await engine.terminate();

    Assert.greater(metrics.outputTokens, 0, "The run generated");
    const recorded = only("engineRun");
    Assert.equal(recorded.host_process, "hwinference");
    Assert.equal(Number(recorded.input_tokens), metrics.inputTokens);
    Assert.equal(Number(recorded.output_tokens), metrics.outputTokens);
    Assert.equal(unsuccessfulRuns().length, 0);
  } finally {
    await cleanup();
  }
});

add_task(async function test_run_records_the_engine_run() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-resources",
      featureId: "link-preview",
    });
    const result = await engine.run(RUN_REQUEST);
    await engine.terminate();

    const run = only("engineRun");
    Assert.equal(run.feature_id, "link-preview");
    Assert.ok(
      ["eos", "length"].includes(result.finishReason),
      `The generation ran to its end (${result.finishReason})`
    );
    Assert.equal(
      run.finish_reason,
      result.finishReason,
      "The run records why the generation stopped"
    );
    Assert.equal(run.engine_id, "hwi-tel-resources");
    Assert.equal(run.model_id, "Mozilla/test-llama");
    Assert.equal(run.backend, "llama.cpp");
    Assert.equal(run.host_process, "hwinference");
    Assert.equal(
      run.backend_source_revision,
      LLAMA_CPP_VERSION,
      "The run names the llama.cpp revision, as the MLEngine path does"
    );

    for (const key of ["wall_milliseconds", "memory_bytes"]) {
      Assert.greater(Number(run[key]), 0, `${key} is measured`);
    }
    Assert.notEqual(run.cpu_milliseconds, null, "cpu_milliseconds is measured");
    Assert.greaterOrEqual(Number(run.cpu_milliseconds), 0);
    Assert.greater(Number(run.system_memory_mb), 0);

    Assert.greater(
      Number(run.output_tokens),
      0,
      "output_tokens is reported whether or not the run streams"
    );
    for (const key of ["time_to_first_chunk", "average_chunk_time"]) {
      Assert.equal(run[key], null, `${key} is only reported for a stream`);
    }
  } finally {
    await cleanup();
  }
});

add_task(async function test_run_records_the_request_feature() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-feature",
      featureId: "test-feature",
    });
    await engine.run(RUN_REQUEST);
    await collectGeneratedText(engine.runWithGenerator(RUN_REQUEST));
    await engine.run({ ...RUN_REQUEST, featureId: "hwi-prompt" });
    await collectGeneratedText(
      engine.runWithGenerator({ ...RUN_REQUEST, featureId: "hwi-prompt" })
    );

    Assert.deepEqual(
      events("engineRun").map(({ extra }) => extra.feature_id),
      ["test-feature", "test-feature", "hwi-prompt", "hwi-prompt"],
      "A run records the request's feature id or falls back to the engine's"
    );
    await engine.terminate();
  } finally {
    await cleanup();
  }
});

add_task(async function test_run_records_its_flow() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-flow",
    });
    await engine.run(RUN_REQUEST);
    await engine.run(RUN_REQUEST, { flowId: "flow-b" });
    await collectGeneratedText(
      engine.runWithGenerator(RUN_REQUEST, { flowId: "flow-b" })
    );
    await engine.terminate();

    const engineFlow = only("engineCreation").flow_id;
    Assert.deepEqual(
      events("engineRun").map(({ extra }) => extra.flow_id),
      [engineFlow, "flow-b", "flow-b"],
      "A run records the flow it is given, or else its engine's"
    );
  } finally {
    await cleanup();
  }
});

add_task(async function test_stream_records_the_chunk_metrics() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-stream",
      featureId: "link-preview",
    });

    let chunks = 0;
    for await (const chunk of engine.runWithGenerator({
      ...RUN_REQUEST,
      minOutputBufferSize: 1,
    })) {
      chunks += chunk.text ? 1 : 0;
    }
    Assert.greater(chunks, 1, "The stream came in several chunks");

    const [run] = events("engineRun");
    Assert.greater(
      Number(run.extra.output_tokens),
      0,
      "output_tokens is the generated token count"
    );
    Assert.equal(
      run.extra.device_kind,
      "cpu",
      "The run records the device kind"
    );
    Assert.greaterOrEqual(Number(run.extra.time_to_first_chunk), 0);
    Assert.notEqual(run.extra.average_chunk_time, null);
    Assert.greaterOrEqual(Number(run.extra.average_chunk_time), 0);

    for await (const chunk of engine.runWithGenerator({
      ...RUN_REQUEST,
      nPredict: 4,
      minOutputBufferSize: 20,
    })) {
      void chunk;
    }
    await engine.terminate();

    const single = events("engineRun")[1].extra;
    Assert.equal(Number(single.output_tokens), 4);
    Assert.equal(
      single.average_chunk_time,
      null,
      "One chunk has no chunk cadence to report"
    );
  } finally {
    await cleanup();
  }
});

add_task(async function test_download_failure() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    let thrown;
    await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-download",
      featureId: "link-preview",
      modelId: "acme-not-found/bert",
      modelFile: "config.json",
    }).catch(e => (thrown = e));
    Assert.ok(thrown, "A missing model fails engine creation");

    Assert.equal(failedCreations().length, 1);
    Assert.equal(
      failedCreations().at(-1).error,
      thrown.name,
      "The creation failure records the download error"
    );
    Assert.equal(failedCreations().at(-1).host_process, "hwinference");
    Assert.equal(
      failedCreations().at(-1).failed_step,
      "model-download",
      "The creation failed while downloading the model"
    );
    const download = events("modelDownload").at(-1).extra;
    Assert.equal(download.outcome, "error", "The download is recorded failed");
    Assert.equal(Number(download.failed_files), 1, "One file failed");
    Assert.equal(unsuccessfulRuns().length, 0);
  } finally {
    await cleanup();
  }
});

add_task(async function test_aborted_creation() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const controller = new AbortController();
    controller.abort();
    await Assert.rejects(
      createEngine(
        {
          ...TINYSTORIES_ENGINE_OPTIONS,
          engineId: "hwi-tel-abort",
          featureId: "link-preview",
        },
        null,
        controller.signal
      ),
      error => error.name === "AbortError"
    );
    Assert.equal(failedCreations().length, 0, "An abort is not a failure");
    const creation = only("engineCreation");
    Assert.equal(
      creation.outcome,
      "aborted",
      "A creation its caller gave up is recorded as aborted"
    );
    Assert.equal(creation.error, undefined, "and names no error");
  } finally {
    await cleanup();
  }
});

add_task(async function test_process_crash_mid_generation() {
  // Sanitizer builds have no crash reporter, so their signal handler turns
  // the deliberate crash into a sanitizer report that fails the run.
  if (AppConstants.ASAN || AppConstants.TSAN) {
    ok(true, "Skipping the deliberate crash on a sanitizer build");
    return;
  }
  SimpleTest.expectChildProcessCrash();
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-crash",
      featureId: "link-preview",
      modelFile: "crash-me.gguf",
    });
    const { pid } = await getInferenceProcessInfo("hwInference");
    const shutdown = TestUtils.topicObserved(
      "ipc:utility-shutdown",
      (subject, data) => parseInt(data, 10) === pid
    );

    await Assert.rejects(
      collectGeneratedText(
        engine.runWithGenerator({ ...RUN_REQUEST, nPredict: 64 })
      ),
      error => error.name === "AbortError"
    );
    await removeCrashDump(shutdown);
    noteIntentionalUtilityCrash(pid);

    Assert.equal(unsuccessfulRuns().length, 1);
    const failed = unsuccessfulRuns()[0].extra;
    Assert.equal(failed.outcome, "error", "A dead process is a failed run");
    Assert.equal(failed.error, "AbortError", "The error is recorded by name");
    Assert.equal(failed.host_process, "hwinference");
    await engine.terminate();
  } finally {
    await cleanup();
  }
});

add_task(async function test_cancel_mid_generation() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-cancel",
      featureId: "link-preview",
    });
    const pending = engine.run({
      prompt: TINYSTORIES_STORYTELLER_PROMPT,
      samplers: TINYSTORIES_GREEDY_SAMPLERS,
      nPredict: 4096,
      stopOnEndOfGenerationTokens: false,
    });
    engine.cancel();
    const result = await pending;
    await engine.terminate();

    Assert.equal(
      result.finishReason,
      "cancelled",
      "The generation was stopped before its end"
    );
    const run = only("engineRun");
    Assert.equal(
      run.outcome,
      "aborted",
      "A generation its caller cancelled is aborted, not a success"
    );
    Assert.equal(run.finish_reason, "cancelled");
  } finally {
    await cleanup();
  }
});

add_task(async function test_terminate_mid_generation() {
  const { cleanup } = await setup({ prefs: HWI_PREFS });
  try {
    const engine = await createEngine({
      ...TINYSTORIES_ENGINE_OPTIONS,
      engineId: "hwi-tel-terminate",
      featureId: "link-preview",
    });
    await Assert.rejects(
      (async () => {
        for await (const chunk of engine.runWithGenerator({
          prompt: TINYSTORIES_STORYTELLER_PROMPT,
          samplers: TINYSTORIES_GREEDY_SAMPLERS,
          nPredict: 4096,
          minOutputBufferSize: 1,
          stopOnEndOfGenerationTokens: false,
        })) {
          void chunk;
          await engine.terminate();
        }
      })(),
      error => error.name === "AbortError"
    );

    Assert.equal(unsuccessfulRuns().length, 1);
    Assert.equal(
      unsuccessfulRuns()[0].extra.outcome,
      "aborted",
      "A run whose engine the caller tore down is aborted, not failed"
    );
  } finally {
    await cleanup();
  }
});
