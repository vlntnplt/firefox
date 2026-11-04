# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

about-inference-title = Inference Manager
about-inference-header = ✨ Inference Manager
about-inference-description = This page gathers all information about local inference
about-inference-ml-disabled =
  .message = To use this page, enable local inference.
about-inference-ml-disabled-enable = Enable
about-inference-simd-supported = SIMD supported
about-inference-onnx-runtime-file = ONNX Runtime file
about-inference-downloads-description = You can delete models here to free up disk space. They will be downloaded again on first inference.
about-inference-delete-button = 🗑️
  .title = Delete model
about-inference-processes-title = Running Inference Processes
about-inference-models-title = Models downloaded
about-inference-yes = Yes
about-inference-no = No
about-inference-total = Total
about-inference-no-processes = No inference processes are running.
about-inference-memory = Memory
about-inference-pid = PID
about-inference-file = File
about-inference-size = Size
about-inference-learn-more = Learn more about the platform here.
about-inference-category-local-inference = Local Inference
about-inference-category-models-storage = Models Storage
about-inference-last-used = Last Used
about-inference-last-updated = Last Updated
about-inference-category-benchmark = Benchmark
about-inference-benchmark-title = Running Benchmark
about-inference-benchmark-description = Performs a full benchmark on the local inference platform. This will take a while.
about-inference-benchmark-button =
  .value = Run Benchmark
about-inference-prompt-title = Confirm
about-inference-prompt-message = Are you sure you want to delete this model?

about-inference-tab-inference-pad = Inference pad
about-inference-tab-text-generation = Text generation
about-inference-tab-devices = HWInference devices

## HWInference devices panel

about-inference-devices = HWInference devices
about-inference-devices-refresh = List devices
about-inference-devices-col-id = Id
about-inference-devices-col-description = Description
about-inference-devices-col-kind = Kind
about-inference-devices-col-backends = Backends

## Text generation panel

about-inference-text-generation = Text generation
about-inference-text-generation-hub = Model hub:
about-inference-text-generation-hub-default = Configured default
about-inference-text-generation-model = Model:
about-inference-text-generation-revision = Revision:
about-inference-text-generation-device = Device:
about-inference-text-generation-threads = Threads:
about-inference-text-generation-flash-attn = Flash attention:
about-inference-text-generation-kv-cache-dtype = KV cache type:
about-inference-text-generation-engines = Engines:
about-inference-text-generation-runs = Runs per engine:
about-inference-text-generation-max-tokens = Max tokens:
about-inference-text-generation-prompt = Prompt:
about-inference-text-generation-run = Generate
about-inference-text-generation-output = Output
about-inference-text-generation-device-cpu = CPU
about-inference-text-generation-device-gpu = GPU, CPU fallback
about-inference-text-generation-col-engine = Engine
about-inference-text-generation-col-run = Run
about-inference-text-generation-col-metric = Metric
about-inference-text-generation-col-min = Min
about-inference-text-generation-col-median = Median
about-inference-text-generation-col-max = Max
about-inference-text-generation-col-init = Engine init (ms)
about-inference-text-generation-col-total = Total (ms)
about-inference-text-generation-col-first-chunk = First chunk (ms)
about-inference-text-generation-col-prompt-tokens = Prompt tokens
about-inference-text-generation-col-prefill = Prefill (ms)
about-inference-text-generation-col-output-tokens = Output tokens
about-inference-text-generation-col-decode = Decode (ms)
about-inference-text-generation-col-tokens-per-second = Tokens/s

## Memories panel: an example run of the memories pipeline on this device.

about-inference-memories-title = Smart Window Memories
about-inference-memories-description = Memory generation, filtering, merging and classification, run on this device. This page uses fixture data and leaves your profile untouched.
about-inference-memories-setup = Setup
about-inference-memories-results = Results
about-inference-memories-inputs = Inputs and outputs
about-inference-memories-model = Model
about-inference-memories-model-loading = loading
about-inference-memories-model-ready = loaded
about-inference-memories-backend = Backend
about-inference-memories-device-label = Device
about-inference-memories-device-pending = GPU if available
# Variables:
#   $device (String) - Vendor name of the device that loaded the model.
#   $kind (String) - Kind of that device: "cpu", "gpu" or "npu".
about-inference-memories-device =
    { $device } { $kind ->
        [gpu] GPU
        [npu] NPU
       *[cpu] CPU
    }
about-inference-memories-cleanup =
  .label = Remove the model after the run
about-inference-memories-run =
  .label = Run
# Variables:
#   $sessions (Number) - Browsing sessions the pipeline reads.
#   $searches (Number) - Web searches across the sessions.
#   $titles (Number) - Visited page titles across the sessions.
#   $chats (Number) - Chat messages across the sessions.
about-inference-memories-sessions =
    .heading = Sessions · { $sessions } sessions · { $searches } searches · { $titles } titles · { $chats } chat turns
# Variables:
#   $count (Number) - Memories in the store.
about-inference-memories-store-summary =
    .heading = Memory store · { $count } memories
about-inference-memories-col-memory = memory
about-inference-memories-col-type = type
about-inference-memories-col-strength = strength
about-inference-memories-col-sources = sources

## Memories panel: headline figures of a run, then one row per pipeline step.

about-inference-memories-tile-inference = inference time
about-inference-memories-tile-load = model load
about-inference-memories-tile-download = model download
about-inference-memories-tile-prefill = prefill tok/s
about-inference-memories-tile-decode = decode tok/s
about-inference-memories-tile-memory = peak memory
about-inference-memories-col-in = in
about-inference-memories-col-out = out
about-inference-memories-col-time = time
about-inference-memories-generation = Generation
about-inference-memories-filter = Filter
about-inference-memories-merge = Merge
about-inference-memories-classification = Classification

## Memories panel: what each step turns into what. Counts not known yet are
## shown as a dash.

# Variables:
#   $sessions (Number) - Sessions the step reads.
#   $candidates (String) - Candidate memories it produced.
about-inference-memories-flow-generation = { $sessions } sessions → { $candidates } candidates
# Variables:
#   $candidates (String) - Candidate memories the step reads.
#   $kept (String) - Candidates it kept.
about-inference-memories-flow-filter = { $candidates } candidates → { $kept } kept
# Variables:
#   $before (String) - Memories in the store before the step.
#   $memories (String) - Memories in the store after the step.
#   $merged (String) - Memories the step formed by merging others.
about-inference-memories-flow-merge = { $before } memories → { $memories } memories, { $merged } merged
# Variables:
#   $fresh (String) - New memories the step classifies.
about-inference-memories-flow-classification =
    { $fresh ->
        [one] { $fresh } new memory
       *[other] { $fresh } new memories
    } → category, intent
# Variables:
#   $seconds (Number) - Duration in seconds.
about-inference-memories-seconds = { NUMBER($seconds, maximumFractionDigits: 1) } s
# Variables:
#   $minutes (Number) - Whole minutes of the duration.
#   $seconds (Number) - Seconds of the duration past the minutes.
about-inference-memories-minutes = { $minutes } min { $seconds } s

# Variables:
#   $stored (Number) - Memories in the store after the run.
#   $new (Number) - Memories the run generated.
#   $merged (Number) - Memories the run formed by merging others.
about-inference-memories-store-after =
    .heading = Memory store · { $stored } memories · { $new } new · { $merged } merged
about-inference-memories-new = new
about-inference-memories-merged = merged
