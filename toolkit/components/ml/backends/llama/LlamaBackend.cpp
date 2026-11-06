/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "LlamaBackend.h"

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include <functional>
#include "mozilla/HashTable.h"
#include "nsIFileStreams.h"
#include "nsTArray.h"

#include "mozilla/Logging.h"
#include "mozilla/dom/BindingUtils.h"
#include "nsFmtString.h"
#include "GeckoProfiler.h"
#include "mozilla/ml/MLProfilerMarkers.h"

mozilla::LazyLogModule gLlamaBackendLog("GeckoMLLlamaBackendNative");

namespace mozilla::llama {

void LlamaBackend::GgmlThreadpoolDeleter::operator()(
    struct ggml_threadpool* aTp) const {
  if (auto* lib = LlamaRuntimeLinker::Get()) {
    if (lib->ggml_threadpool_free) {
      lib->ggml_threadpool_free(aTp);
    }
  }
}

void LlamaBackend::LlamaModelDeleter::operator()(llama_model* aModel) const {
  if (auto* lib = LlamaRuntimeLinker::Get()) {
    if (lib->llama_model_free) {
      lib->llama_model_free(aModel);
    }
  }
}

void LlamaBackend::LlamaContextDeleter::operator()(llama_context* aCtx) const {
  if (auto* lib = LlamaRuntimeLinker::Get()) {
    if (lib->llama_free) {
      lib->llama_free(aCtx);
    }
  }
}

void LlamaBackend::LlamaSamplerDeleter::operator()(llama_sampler* aSmpl) const {
  if (auto* lib = LlamaRuntimeLinker::Get()) {
    if (lib->llama_sampler_free) {
      lib->llama_sampler_free(aSmpl);
    }
  }
}

#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gLlamaBackendLog, LogLevel::Debug, fmt, ##__VA_ARGS__)

#define LOGV(fmt, ...) \
  MOZ_LOG_FMT(gLlamaBackendLog, LogLevel::Verbose, fmt, ##__VA_ARGS__)

#define LOGW(fmt, ...) \
  MOZ_LOG_FMT(gLlamaBackendLog, LogLevel::Warning, fmt, ##__VA_ARGS__)

#define LOGE(fmt, ...) \
  MOZ_LOG_FMT(gLlamaBackendLog, LogLevel::Verbose, fmt, ##__VA_ARGS__)

ggml_type GgmlTypeFromKVCacheDtype(LlamaKVCacheDtype aDtype) {
  switch (aDtype) {
    case LlamaKVCacheDtype::F32:
      return GGML_TYPE_F32;
    case LlamaKVCacheDtype::F16:
      return GGML_TYPE_F16;
    case LlamaKVCacheDtype::Q8_0:
      return GGML_TYPE_Q8_0;
    case LlamaKVCacheDtype::Q5_1:
      return GGML_TYPE_Q5_1;
    case LlamaKVCacheDtype::Q5_0:
      return GGML_TYPE_Q5_0;
    case LlamaKVCacheDtype::Q4_1:
      return GGML_TYPE_Q4_1;
    case LlamaKVCacheDtype::Q4_0:
      return GGML_TYPE_Q4_0;
    default:
      MOZ_ASSERT_UNREACHABLE("Unhandled LlamaKVCacheQuantizationLevel");
      break;
  }
  return GGML_TYPE_F16;
}

static const char* NonNull(const char* aStr) { return aStr ? aStr : ""; }

static const char* DeviceId(const ggml_backend_dev_props& aProps) {
  return NonNull(aProps.device_id ? aProps.device_id : aProps.name);
}

static dom::HWInferenceDeviceKind DeviceKind(enum ggml_backend_dev_type aType) {
  switch (aType) {
    case GGML_BACKEND_DEVICE_TYPE_CPU:
      return dom::HWInferenceDeviceKind::Cpu;
    case GGML_BACKEND_DEVICE_TYPE_GPU:
    case GGML_BACKEND_DEVICE_TYPE_IGPU:
      return dom::HWInferenceDeviceKind::Gpu;
    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
      return dom::HWInferenceDeviceKind::Npu;
    case GGML_BACKEND_DEVICE_TYPE_META:
      break;
  }
  MOZ_CRASH("not a device");
}

// Initialises an accelerator for real and frees it; the CPU always passes.
static bool ProbeDevice(const LlamaLibWrapper* aLib,
                        ggml_backend_dev_t aDevice) {
  if (aLib->ggml_backend_dev_type(aDevice) == GGML_BACKEND_DEVICE_TYPE_CPU) {
    return true;
  }
  ggml_backend_t backend = aLib->ggml_backend_dev_init(aDevice, nullptr);
  if (!backend) {
    return false;
  }
  aLib->ggml_backend_free(backend);
  return true;
}

const nsTArray<Device>& Devices(const LlamaLibWrapper* aLib) {
  static const nsTArray<Device> sDevices = [aLib]() {
    nsTArray<Device> devices;
    for (size_t i = 0; i < aLib->ggml_backend_dev_count(); ++i) {
      ggml_backend_dev_t dev = aLib->ggml_backend_dev_get(i);
      ggml_backend_dev_props props;
      aLib->ggml_backend_dev_get_props(dev, &props);
      if (props.type == GGML_BACKEND_DEVICE_TYPE_META) {
        continue;
      }
      const TimeStamp probeStart = TimeStamp::Now();
      const bool usable = ProbeDevice(aLib, dev);
      PROFILER_MARKER(
          ML_TEXT_GENERATION_TRACK, ML_SETUP,
          MarkerTiming::IntervalUntilNowFrom(probeStart), MLDeviceProbeMarker,
          ProfilerString8View::WrapNullTerminatedString(props.name), usable);
      if (!usable) {
        LOGW("{}: {} failed to initialise, skipped", __PRETTY_FUNCTION__,
             props.name);
        continue;
      }
      Device* device = devices.AppendElement(Device{dev, DeviceId(props),
                                                    NonNull(props.description),
                                                    DeviceKind(props.type)});
      LOGD("{}: {}: {} ({})", __PRETTY_FUNCTION__, device->mId,
           device->mDescription, dom::GetEnumString(device->mKind));
    }
    return devices;
  }();
  return sDevices;
}

// The CPU device, found without probing the accelerators.
static const Device* CpuDevice(const LlamaLibWrapper* aLib) {
  static const Maybe<Device> sCpu = [aLib]() -> Maybe<Device> {
    for (size_t i = 0; i < aLib->ggml_backend_dev_count(); ++i) {
      ggml_backend_dev_t dev = aLib->ggml_backend_dev_get(i);
      ggml_backend_dev_props props;
      aLib->ggml_backend_dev_get_props(dev, &props);
      if (props.type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return Some(Device{dev, DeviceId(props), NonNull(props.description),
                           dom::HWInferenceDeviceKind::Cpu});
      }
    }
    return Nothing();
  }();
  return sCpu.ptrOr(nullptr);
}

// Devices in placement order: by id, or every device of each kind. An empty
// placement resolves to the CPU alone.
static nsTArray<const Device*> ResolvePlacement(
    const LlamaLibWrapper* aLib, const dom::HWInferencePlacement& aPlacement) {
  nsTArray<const Device*> candidates;
  if (aPlacement.mDevices.IsEmpty() && aPlacement.mKinds.IsEmpty()) {
    if (const Device* cpu = CpuDevice(aLib)) {
      candidates.AppendElement(cpu);
    }
    return candidates;
  }
  auto appendIf = [&](auto&& aMatches) {
    for (const Device& device : Devices(aLib)) {
      if (aMatches(device)) {
        candidates.AppendElement(&device);
      }
    }
  };
  if (!aPlacement.mDevices.IsEmpty()) {
    for (const nsCString& id : aPlacement.mDevices) {
      appendIf([&](const Device& aDevice) { return id.Equals(aDevice.mId); });
    }
  } else {
    for (dom::HWInferenceDeviceKind kind : aPlacement.mKinds) {
      appendIf([&](const Device& aDevice) { return aDevice.mKind == kind; });
    }
  }
  return candidates;
}

LlamaBackend::LlamaModelUPtr LlamaBackend::LoadOnDevice(
    FILE* aFp, const Device& aDevice, llama_model_params aParams) {
  constexpr int32_t kNoGpuLayers = 0;
  constexpr int32_t kAllGpuLayers = -1;
  const bool cpu = aDevice.mKind == dom::HWInferenceDeviceKind::Cpu;
  // An empty list keeps llama.cpp off every accelerator.
  ggml_backend_dev_t devices[2] = {cpu ? nullptr : aDevice.mHandle, nullptr};
  aParams.devices = devices;
  aParams.n_gpu_layers = cpu ? kNoGpuLayers : kAllGpuLayers;
  rewind(aFp);
  return LlamaModelUPtr(mLib->llama_model_load_from_file_handle(aFp, aParams));
}

ResultStatus LlamaBackend::LoadOnFirstDevice(const LlamaModelOptions& aOptions,
                                             FILE* aFp,
                                             llama_model_params aParams) {
  if (!aOptions.mPlacement.mDevices.IsEmpty() &&
      !aOptions.mPlacement.mKinds.IsEmpty()) {
    auto msg = nsFmtCString("{}: a placement takes devices or kinds, not both",
                            __PRETTY_FUNCTION__);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }
  nsTArray<const Device*> candidates =
      ResolvePlacement(mLib, aOptions.mPlacement);
  if (candidates.IsEmpty()) {
    auto msg = nsFmtCString("{}: no device matches the requested placement",
                            __PRETTY_FUNCTION__);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }
  for (const Device* device : candidates) {
    const TimeStamp loadStart = TimeStamp::Now();
    mModel = LoadOnDevice(aFp, *device, aParams);
    PROFILER_MARKER(
        ML_TEXT_GENERATION_TRACK, ML_SETUP,
        MarkerTiming::IntervalUntilNowFrom(loadStart), MLDeviceLoadMarker,
        ProfilerString8View::WrapNullTerminatedString(device->mId), !!mModel);
    if (mModel) {
      mLoadedDevice = device;
      mDevice = device->mId;
      mDeviceKind = device->mKind;
      return Ok();
    }
    LOGW("{}: {} could not load the model", __PRETTY_FUNCTION__,
         device->mDescription);
  }
  auto msg = nsFmtCString("{}: no device of the placement could load the model",
                          __PRETTY_FUNCTION__);
  LOGE("{}", msg);
  return mozilla::Err(Error{std::move(msg)});
}

LlamaBackend::~LlamaBackend() {
  LOGD("Entered {}", __PRETTY_FUNCTION__);
  // Note: mLib is not freed here because LlamaRuntimeLinker manages
  // the library lifetime as a singleton
}

ResultStatus LlamaBackend::LoadDraftModel(FILE* aFp,
                                          llama_model_params aParams) {
  mDraftModel = LoadOnDevice(aFp, *mLoadedDevice, aParams);
  if (!mDraftModel) {
    auto msg = nsFmtCString("{}: could not load the drafter on {}",
                            __PRETTY_FUNCTION__, mDevice);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }
  const int32_t nEmbd = mLib->llama_model_n_embd(mModel.get());
  const int32_t nEmbdDraft = mLib->llama_model_n_embd_out(mDraftModel.get());
  if (nEmbd != nEmbdDraft) {
    auto msg =
        nsFmtCString("{}: the drafter reads a hidden size of {}, {} has {}",
                     __PRETTY_FUNCTION__, nEmbdDraft, mModelGeneralName, nEmbd);
    LOGE("{}", msg);
    mDraftModel.reset();
    return mozilla::Err(Error{std::move(msg)});
  }
  return Ok();
}

ResultStatus LlamaBackend::Reinitialize(const LlamaModelOptions& aOptions,
                                        FILE* aFp, FILE* aDraftFp) {
  LOGV("Entered {}", __PRETTY_FUNCTION__);
  // The draft context reads the model's context: drop it first.
  mDraftCtx.reset();
  mDraftModel.reset();

  if (!mLib) {
    mLib = LlamaRuntimeLinker::Get();
  }

  if (!mLib) {
    auto msg = nsFmtCString("{}: Failed to get llama runtime linker",
                            __PRETTY_FUNCTION__);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  mModelOptions = aOptions;

  LOGV("{}: Initializing the model", __PRETTY_FUNCTION__);

  // initialize the model
  llama_model_params modelParams = mLib->llama_model_default_params();
  modelParams.use_mmap = aOptions.mUseMmap;
  modelParams.use_mlock = aOptions.mUseMlock;
  modelParams.check_tensors = aOptions.mCheckTensors;
  mDevice.Truncate();
  mDeviceKind = dom::HWInferenceDeviceKind::Cpu;

  auto loaded = LoadOnFirstDevice(aOptions, aFp, modelParams);
  if (loaded.isErr()) {
    return loaded;
  }

  // Preallocate buffer based on rough estimate of the size of the name
  mModelGeneralName.SetLength(256);
  auto numWritten = mLib->llama_model_meta_val_str(
      mModel.get(), "general.name", mModelGeneralName.BeginWriting(),
      mModelGeneralName.Length());
  if (numWritten >= 0) {
    // Reset the length to the number of actual bytes written
    mModelGeneralName.SetLength(numWritten);
    mModelGeneralName.Insert("Model: "_ns, 0);
  } else {
    LOGW(
        "Unable to extract the name of the model. Logging message will not "
        "include it.");
    mModelGeneralName = ""_ns;
  }

  LOGV("{}: Successfully Initialized {}", __PRETTY_FUNCTION__,
       mModelGeneralName);

  if (aDraftFp) {
    auto loaded = LoadDraftModel(aDraftFp, modelParams);
    if (loaded.isErr()) {
      return loaded;
    }
  }

  return ReinitializeContext(aOptions.mContext, aOptions.mContext.mNCtx);
}

ResultStatus LlamaBackend::ReinitializeContext(
    const LlamaContextOptions& aOptions, int aNumContext) {
  LOGV("Entered {}", __PRETTY_FUNCTION__);

  MOZ_ASSERT(mLib,
             "No shared library pointer in ReinitializeContext, fix this");

  llama_context_params ctxParams = mLib->llama_context_default_params();

  ctxParams.n_ctx = aNumContext;

  ctxParams.n_batch = aOptions.mNBatch;
  ctxParams.n_ubatch = aOptions.mNUbatch;
  ctxParams.n_seq_max = aOptions.mNSeqMax;
  ctxParams.n_threads = aOptions.mNThreads;
  ctxParams.n_threads_batch = aOptions.mNThreadsBatch;

  ctxParams.type_k = GgmlTypeFromKVCacheDtype(aOptions.mKCacheDtype);
  ctxParams.type_v = GgmlTypeFromKVCacheDtype(aOptions.mVCacheDtype);
  // The KV cache and host-side ops follow the model onto an accelerator.
  const bool onAccelerator = mDeviceKind != dom::HWInferenceDeviceKind::Cpu;
  ctxParams.offload_kqv = onAccelerator;
  ctxParams.flash_attn_type = aOptions.mFlashAttn
                                  ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                  : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  ctxParams.no_perf = aOptions.mNoPerf;
  // On the CPU, op_offload would still route big matmuls to a registered GPU.
  ctxParams.op_offload = onAccelerator;
  ctxParams.swa_full = aOptions.mSwaFull;

  mDraftCtx.reset();
  // llama_init_from_model does not take ownership of the model or context
  // parameters. We retain ownership of all objects passed in and must keep them
  // alive for the entire lifetime of mCtx.
  mCtx.reset(mLib->llama_init_from_model(mModel.get(), ctxParams));
  if (!mCtx) {
    auto msg = nsFmtCString("{}: failed to create the llama_context {}",
                            __PRETTY_FUNCTION__, mModelGeneralName);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  // ggml_threadpool_params_init/ggml_threadpool_new do not retain pointers
  // to the params structs. We own those stack-allocated structs, but they can
  // safely go out of scope after threadpool creation.
  ggml_threadpool_params tpp;
  mLib->ggml_threadpool_params_init(&tpp, ctxParams.n_threads);
  tpp.thread_create_callback = []() { PROFILER_REGISTER_THREAD("llama.cpp"); };
  tpp.thread_destroy_callback = []() { PROFILER_UNREGISTER_THREAD(); };

  ggml_threadpool_params tppBatch;
  mLib->ggml_threadpool_params_init(&tppBatch, ctxParams.n_threads_batch);
  tppBatch.thread_create_callback = []() {
    PROFILER_REGISTER_THREAD("llama.cpp");
  };
  tppBatch.thread_destroy_callback = []() { PROFILER_UNREGISTER_THREAD(); };

  mThreadpoolBatch.reset();
  if (!mLib->ggml_threadpool_params_match(&tpp, &tppBatch)) {
    mThreadpoolBatch.reset(mLib->ggml_threadpool_new(&tppBatch));
    if (!mThreadpoolBatch) {
      auto msg = nsFmtCString(
          "{}: Failed to create decoding threadpool: n_threads: {}  {}",
          __PRETTY_FUNCTION__, ctxParams.n_threads_batch, mModelGeneralName);
      LOGE("{}", msg);
      return mozilla::Err(Error{std::move(msg)});
    }
    // Start the non-batch threadpool in the paused state
    tpp.paused = true;
  }

  mThreadpool.reset(mLib->ggml_threadpool_new(&tpp));
  if (!mThreadpool) {
    auto msg = nsFmtCString("{}: Failed to create threadpool: n_threads: {} {}",
                            __PRETTY_FUNCTION__, ctxParams.n_threads,
                            mModelGeneralName);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  // llama_attach_threadpool does not take ownership of the threadpools.
  // We must keep mThreadpool and mThreadpoolBatch alive for as long as
  // mCtx is used, since they are accessed internally by the context.
  mLib->llama_attach_threadpool(mCtx.get(), mThreadpool.get(),
                                mThreadpoolBatch.get());

  if (mDraftModel) {
    llama_context_params draftParams = ctxParams;
    draftParams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    draftParams.ctx_other = mCtx.get();
    mDraftCtx.reset(
        mLib->llama_init_from_model(mDraftModel.get(), draftParams));
    if (!mDraftCtx) {
      auto msg = nsFmtCString("{}: failed to create the draft llama_context {}",
                              __PRETTY_FUNCTION__, mModelGeneralName);
      LOGE("{}", msg);
      return mozilla::Err(Error{std::move(msg)});
    }
    mLib->llama_attach_threadpool(mDraftCtx.get(), mThreadpool.get(),
                                  mThreadpoolBatch.get());
    // The drafter consumes the hidden state the model sampled from, and
    // feeds its own back to itself; both only for output rows.
    mLib->mozllama_set_embeddings_nextn(mCtx.get(), true, true);
    mLib->mozllama_set_embeddings_nextn(mDraftCtx.get(), true, true);
  }

  LOGV("{}: Successfully Initialized context for model {}", __PRETTY_FUNCTION__,
       mModelGeneralName);

  return mozilla::Ok();
}

ChatMessageResult LlamaBackend::FormatChat(
    const mozilla::dom::LlamaFormatChatOptions& aOptions) {
  LOGV("Entered {}", __PRETTY_FUNCTION__);

  MOZ_ASSERT(mLib, "No shared library pointer in FormatChat, fix this");

  if (!mModel) {
    auto msg = nsFmtCString("{}: Model not loaded when trying to format chat",
                            __PRETTY_FUNCTION__);
    LOGE("{}", msg);
    return ChatMessageResult(Error{std::move(msg)});
  }

  nsTArray<llama_chat_message> chatMessages;
  chatMessages.SetCapacity(aOptions.mMessages.Length());
  for (const auto& msg : aOptions.mMessages) {
    // llama_chat_message is a simple struct that stores raw `const char*`
    // pointers. It does not take ownership of the strings — the caller must
    // ensure that msg.mRole and msg.mContent outlive any use of chatMessages.
    // Here, we build it on the stack, so the pointers are valid during the
    // function call.
    chatMessages.AppendElement(
        llama_chat_message{msg.mRole.get(), msg.mContent.get()});
  }

  // This returns a pointer to a template string stored inside mModel.
  // Since we own mModel, the pointer remains valid throughout this function.
  const char* tmpl =
      mLib->llama_model_chat_template(mModel.get(), /* name */ nullptr);

  int32_t numCharInMessages = 0;
  for (const auto& msg : aOptions.mMessages) {
    numCharInMessages += msg.mRole.Length() + msg.mContent.Length();
  }
  numCharInMessages *= 2;
  LOGD("{}: Estimated number of chars {}, for the formatted message {}",
       numCharInMessages, __PRETTY_FUNCTION__, mModelGeneralName);

  // Preallocate buffer based on rough estimate
  nsCString formatted;
  formatted.SetLength(numCharInMessages);

  // llama_chat_apply_template does not retain any of the pointers passed to it.
  // All data must remain valid only during the call.
  int32_t chatTemplateLength = mLib->llama_chat_apply_template(
      tmpl, chatMessages.Elements(), chatMessages.Length(),
      aOptions.mAddAssistant, formatted.BeginWriting(), formatted.Length());

  if (chatTemplateLength < 0) {
    auto msg = nsFmtCString("{}: failed to apply the chat template",
                            __PRETTY_FUNCTION__);
    LOGE("{}", msg);
    return ChatMessageResult(Error{std::move(msg)});
  }

  // Retry if the estimated buffer size was too small
  if (mozilla::AssertedCast<size_t>(chatTemplateLength) > formatted.Length()) {
    LOGD(
        "{}: Wrong estimate for format chat. Retrying with the correct value "
        "{}, {}",
        __PRETTY_FUNCTION__, chatTemplateLength, mModelGeneralName);

    formatted.SetLength(chatTemplateLength);
    chatTemplateLength = mLib->llama_chat_apply_template(
        tmpl, chatMessages.Elements(), chatMessages.Length(),
        aOptions.mAddAssistant, formatted.BeginWriting(), formatted.Length());
  }

  // Trim to actual size returned by llama
  formatted.SetLength(chatTemplateLength);

  LOGD("{}: Successfully formatted chat", __PRETTY_FUNCTION__);

  return ChatMessageResult(std::move(formatted));
}

LlamaBackend::SamplerResult LlamaBackend::InitializeSampler(
    const mozilla::dom::Sequence<LlamaSamplerConfig>& aSamplers,
    const llama_vocab* vocab) {
  LOGV("Entered {}", __PRETTY_FUNCTION__);

  MOZ_ASSERT(mLib, "No shared library pointer in InitializeSampler, fix this");

  // initialize the sampler. We own the pointer returned by
  // llama_sampler_chain_init. sampler_element pointer are added to sampler by
  // llama_sampler_chain_add and we need to delete them

  LlamaSamplerUPtr sampler(mLib->llama_sampler_chain_init(
      mLib->llama_sampler_chain_default_params()));
  LlamaSamplerUPtr grammar;

  auto n_vocab = mLib->llama_vocab_n_tokens(vocab);
  bool selectsToken = false;

  for (const auto& samplerConfig : aSamplers) {
    llama_sampler* samplerElement = nullptr;

    switch (samplerConfig.mType) {
      case LlamaSamplerType::Temperature:
        samplerElement = mLib->llama_sampler_init_temp(samplerConfig.mTemp);
        break;

      case LlamaSamplerType::Dist: {
        auto seed = samplerConfig.mSeed.WasPassed()
                        ? samplerConfig.mSeed.Value()
                        : LLAMA_DEFAULT_SEED;
        samplerElement = mLib->llama_sampler_init_dist(seed);
        selectsToken = true;
        break;
      }

      case LlamaSamplerType::Grammar:
        grammar.reset(mLib->llama_sampler_init_grammar(
            vocab, samplerConfig.mGrammar.get(), "root"));
        if (!grammar) {
          auto msg = nsFmtCString("{}: Grammar rejected", __PRETTY_FUNCTION__);
          LOGE("{}", msg);
          return mozilla::Err(Error{std::move(msg)});
        }
        break;

      case LlamaSamplerType::Top_k:
        samplerElement = mLib->llama_sampler_init_top_k(samplerConfig.mTopK);
        break;

      case LlamaSamplerType::Top_p:
        samplerElement = mLib->llama_sampler_init_top_p(samplerConfig.mTopP,
                                                        samplerConfig.mMinKeep);
        break;

      case LlamaSamplerType::Logit_bias: {
        nsTArray<llama_logit_bias> logitBias;
        logitBias.SetCapacity(samplerConfig.mLogitBias.Length());
        for (const auto& val : samplerConfig.mLogitBias) {
          logitBias.AppendElement(llama_logit_bias{val.mToken, val.mBias});
        }

        samplerElement = mLib->llama_sampler_init_logit_bias(
            n_vocab, samplerConfig.mLogitBias.Length(), logitBias.Elements());
        break;
      }

      default:

        auto msg =
            nsFmtCString("{}: Unimplemented sampler type", __PRETTY_FUNCTION__);
        LOGE("{}", msg);
        return mozilla::Err(Error{std::move(msg)});
    }

    if (samplerElement) {
      mLib->llama_sampler_chain_add(sampler.get(), samplerElement);
    }
  }

  // llama_sampler_sample asserts when no sampler selected a token.
  if (!selectsToken) {
    mLib->llama_sampler_chain_add(sampler.get(),
                                  mLib->llama_sampler_init_greedy());
  }

  LOGV("{} Sampler successfully initialized.", __PRETTY_FUNCTION__);
  return SamplerResult(Samplers{std::move(sampler), std::move(grammar)});
}

llama_token LlamaBackend::SampleToken(const Samplers& aSamplers,
                                      int32_t aLogitIndex,
                                      nsTArray<llama_token_data>& aCandidates) {
  llama_sampler* chain = aSamplers.mChain.get();
  llama_sampler* grammar = aSamplers.mGrammar.get();
  if (!grammar) {
    return mLib->llama_sampler_sample(chain, mCtx.get(), aLogitIndex);
  }

  const float* logits = mLib->llama_get_logits_ith(mCtx.get(), aLogitIndex);
  const int32_t nVocab =
      mLib->llama_vocab_n_tokens(mLib->llama_model_get_vocab(mModel.get()));
  auto setCandidates = [&]() {
    aCandidates.SetLength(nVocab);
    for (int32_t i = 0; i < nVocab; i++) {
      aCandidates[i] = llama_token_data{i, logits[i], 0.0f};
    }
    return llama_token_data_array{aCandidates.Elements(), aCandidates.Length(),
                                  -1, false};
  };

  llama_token_data_array candidates = setCandidates();
  mLib->llama_sampler_apply(chain, &candidates);
  MOZ_RELEASE_ASSERT(candidates.selected >= 0);
  llama_token token = candidates.data[candidates.selected].id;

  llama_token_data single{token, 1.0f, 0.0f};
  llama_token_data_array probe{&single, 1, -1, false};
  mLib->llama_sampler_apply(grammar, &probe);
  if (single.logit == -INFINITY) {
    candidates = setCandidates();
    mLib->llama_sampler_apply(grammar, &candidates);
    mLib->llama_sampler_apply(chain, &candidates);
    MOZ_RELEASE_ASSERT(candidates.selected >= 0);
    token = candidates.data[candidates.selected].id;
  }

  mLib->llama_sampler_accept(grammar, token);
  mLib->llama_sampler_accept(chain, token);
  return token;
}

// Draft length and the drafter confidence below which a draft stops: the
// defaults of llama.cpp's draft-mtp speculation.
static constexpr int32_t kDraftMax = 8;
static constexpr float kDraftPMin = 0.75f;

nsTArray<llama_token> LlamaBackend::Draft(llama_token aLast, llama_pos aPos,
                                          const nsTArray<float>& aH,
                                          int32_t aMax) {
  nsTArray<llama_token> draft;
  if (aMax <= 0 || aH.IsEmpty()) {
    return draft;
  }
  const int32_t nVocab = mLib->llama_vocab_n_tokens(
      mLib->llama_model_get_vocab(mDraftModel.get()));
  // Every draft token sits at aPos: the drafter attends to the model's KV
  // cache and carries its own tokens through the hidden state only.
  nsTArray<float> h(aH.Clone());
  llama_token token = aLast;
  llama_pos pos = aPos;
  int32_t nSeq = 1;
  llama_seq_id seq = 0;
  llama_seq_id* seqIds = &seq;
  int8_t output = 1;
  while (draft.Length() < static_cast<size_t>(aMax)) {
    llama_batch batch{1, &token, h.Elements(), &pos, &nSeq, &seqIds, &output};
    if (mLib->llama_decode(mDraftCtx.get(), batch) != 0) {
      LOGW("{}: the drafter failed to decode", __PRETTY_FUNCTION__);
      break;
    }
    const float* logits = mLib->llama_get_logits_ith(mDraftCtx.get(), 0);
    int32_t best = 0;
    for (int32_t i = 1; i < nVocab; i++) {
      if (logits[i] > logits[best]) {
        best = i;
      }
    }
    double sum = 0.0;
    for (int32_t i = 0; i < nVocab; i++) {
      sum += std::exp(static_cast<double>(logits[i] - logits[best]));
    }
    if (1.0 / sum < kDraftPMin) {
      break;
    }
    draft.AppendElement(best);
    const float* next =
        mLib->mozllama_get_embeddings_nextn_ith(mDraftCtx.get(), 0);
    if (!next) {
      break;
    }
    token = best;
    memcpy(h.Elements(), next, h.Length() * sizeof(float));
  }
  return draft;
}

ResultStatus LlamaBackend::Generate(
    const LlamaChatOptions& aOptions,
    std::function<ResultStatus(const LlamaChatResponse&)> aTokenCallback,
    std::function<bool()> aCancelCallback) {
  LOGV("Entered {}", __PRETTY_FUNCTION__, mModelGeneralName);

  MOZ_ASSERT(mLib, "No shared library pointer in Generate, fix this");

  auto cleanup = mozilla::MakeScopeExit([&ctx = mCtx, lib = mLib] {
    // clear the memory used by this generation session to make it ready for the
    // next session
    bool clearDataBuffers = true;
    if (ctx) {
      lib->llama_memory_clear(lib->llama_get_memory(ctx.get()),
                              clearDataBuffers);
    }
  });

  if (!mModel) {
    auto msg = nsFmtCString("{}: error: Model not loaded", __PRETTY_FUNCTION__);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  // Just a non-owned pointer to existing data, so fast to get each time
  const llama_vocab* vocab = mLib->llama_model_get_vocab(mModel.get());

  if (!vocab) {
    auto msg = nsFmtCString("{}: error: Unable to get model vocabulary.",
                            __PRETTY_FUNCTION__);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  auto samplerResult = InitializeSampler(aOptions.mSamplers, vocab);

  if (samplerResult.isErr()) {
    LOGE("{}", samplerResult.inspectErr().mMessage);
    return mozilla::Err(samplerResult.inspectErr());
  }

  auto samplers = samplerResult.unwrap();
  nsTArray<llama_token_data> candidates;

  const size_t estimatedNumPromptTokens = aOptions.mPrompt.Length() + 1;
  LOGD("{} Estimated tokenization size is {} {}", __PRETTY_FUNCTION__,
       estimatedNumPromptTokens, mModelGeneralName);
  nsTArray<llama_token> promptTokens;
  promptTokens.SetLength(estimatedNumPromptTokens);
  // attempt to tokenize the prompt using the estimated number of prompts
  int32_t nPromptTokens = mLib->llama_tokenize(
      vocab, aOptions.mPrompt.get(), aOptions.mPrompt.Length(),
      promptTokens.Elements(), promptTokens.Length(),
      aOptions.mTokenizationOptions.mAddBosAndEos,
      aOptions.mTokenizationOptions.mParseSpecilControlTokens);

  if (nPromptTokens < 0) {
    auto msg = nsFmtCString("{}: failed to tokenize the prompt {}",
                            __PRETTY_FUNCTION__, mModelGeneralName);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  // If the estimate was wrong, retry with the correct number
  if (mozilla::AssertedCast<size_t>(nPromptTokens) > estimatedNumPromptTokens) {
    LOGD(
        "{} Wrong estimate for tokenization. Retrying with the correct "
        "value {} {}",
        __PRETTY_FUNCTION__, nPromptTokens, mModelGeneralName);
    promptTokens.SetLength(nPromptTokens);

    nPromptTokens = mLib->llama_tokenize(
        vocab, aOptions.mPrompt.get(), aOptions.mPrompt.Length(),
        promptTokens.Elements(), promptTokens.Length(),
        aOptions.mTokenizationOptions.mAddBosAndEos,
        aOptions.mTokenizationOptions.mParseSpecilControlTokens);
  }

  promptTokens.SetLength(nPromptTokens);

  int nCtx = mLib->llama_n_ctx(mCtx.get());

  // 0 generates until the context is full.
  int32_t seqLen = aOptions.mMaxGeneratedTokens;
  if (seqLen == 0) {
    seqLen = std::max(nCtx - nPromptTokens, 0);
  }

  int estimatedCtx = nPromptTokens + seqLen;

  if (estimatedCtx > nCtx) {
    LOGW(
        "{} We estimated that {} will be needed but the model context was "
        "initialized "
        "with {}. Re-initializing the model context {}",
        __PRETTY_FUNCTION__, estimatedCtx, nCtx, mModelGeneralName);

    auto initContextResult =
        ReinitializeContext(mModelOptions.mContext, estimatedCtx);
    if (initContextResult.isErr()) {
      LOGE("{}", initContextResult.inspectErr().mMessage);
      return initContextResult;
    }

    nCtx = mLib->llama_n_ctx(mCtx.get());

    mModelOptions.mContext.mNCtx = estimatedCtx;
  }

  // llama_decode asserts on a batch larger than the context batch size.
  int nBatch = mLib->llama_n_batch(mCtx.get());
  if (nPromptTokens > nBatch) {
    auto msg = nsFmtCString(
        "{}: prompt of {} tokens exceeds the batch of {} {}",
        __PRETTY_FUNCTION__, nPromptTokens, nBatch, mModelGeneralName);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  LOGD("{} Creating llama.cpp batch from prompt tokens for {}",
       __PRETTY_FUNCTION__, mModelGeneralName);
  llama_batch batch =
      mLib->llama_batch_get_one(promptTokens.Elements(), promptTokens.Length());

  int32_t numGeneratedTokens = 0;

  // The stop tokens are user inputs and could be very big.
  // We store them in a hash set to bound the time complexity of the membership
  // check. Typically, this is just empty.
  mozilla::HashSet<int32_t> stopTokens;
  for (const auto& v : aOptions.mStopTokens) {
    if (!stopTokens.put(v)) {
      auto msg = nsFmtCString("{}: Unable to create stopTokens {}",
                              __PRETTY_FUNCTION__, mModelGeneralName);
      LOGE("{}", msg);
      return mozilla::Err(Error{std::move(msg)});
    };
  }

  auto onToken = [aTokenCallback](nsCString piece, nsTArray<int32_t> tokens,
                                  LlamaChatPhase phase, bool completed) {
    LlamaChatResponse result;
    result.mPiece = std::move(piece);
    result.mTokens = std::move(tokens);
    result.mPhase = phase;
    result.mIsPhaseCompleted = completed;
    return aTokenCallback(result);
  };

  if (aCancelCallback()) {
    LOGD("{} Generation successfully cancelled. {}", __PRETTY_FUNCTION__,
         mModelGeneralName);
    return mozilla::Ok();
  }

  if (nPromptTokens > nCtx) {
    auto msg = nsFmtCString(
        "{}: context size exceeded. Size is: {} Needed: {} {}",
        __PRETTY_FUNCTION__, nCtx, nPromptTokens, mModelGeneralName);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  LOGV("{}: Decoding the prompt {}", __PRETTY_FUNCTION__, mModelGeneralName);
  if (mLib->llama_decode(mCtx.get(), batch) != 0) {
    auto msg = nsFmtCString("{}: failed to decode {}", __PRETTY_FUNCTION__,
                            mModelGeneralName);
    LOGE("{}", msg);
    return mozilla::Err(Error{std::move(msg)});
  }

  // Tokens the model committed to, in order: the one sampled after the batch
  // it just decoded, preceded by the drafts it verified.
  nsTArray<llama_token> accepted;
  accepted.AppendElement(SampleToken(samplers, -1, candidates));
  // Tokens in the KV cache; accepted.LastElement() is not among them yet.
  llama_pos nPast = nPromptTokens;
  // The model's hidden state for accepted.LastElement(): the drafter's input.
  nsTArray<float> pendingH;
  auto keepHiddenState = [&](int32_t aRow) {
    if (!mDraftCtx) {
      return;
    }
    const float* h = mLib->mozllama_get_embeddings_nextn_ith(mCtx.get(), aRow);
    const int32_t nEmbd = mLib->llama_model_n_embd(mModel.get());
    pendingH.ClearAndRetainStorage();
    if (h) {
      pendingH.AppendElements(h, nEmbd);
    }
  };
  keepHiddenState(nPromptTokens - 1);
  mDraftStats = DraftStats();

  // Sending the end of prompt once we get the first token to ensure
  // all operations are indeed completed.
  // Otherwise we would need a call to llama_synchronize(ctx)
  LOGD("{}: Prompt processing successfully completed {}", __PRETTY_FUNCTION__,
       mModelGeneralName);
  {
    // move the promptTokens as we don't need it anymore
    auto onTokenResult = onToken(aOptions.mPrompt, std::move(promptTokens),
                                 LlamaChatPhase::Prompt, true);
    if (onTokenResult.isErr()) {
      LOGE("{}", onTokenResult.inspectErr().mMessage);
      return onTokenResult;
    }
  }

  // The verification batch: the last accepted token and the drafts after it.
  nsTArray<llama_token> verifyTokens;
  nsTArray<llama_pos> verifyPos;
  nsTArray<int32_t> verifyNSeq;
  nsTArray<llama_seq_id*> verifySeq;
  nsTArray<int8_t> verifyLogits;
  llama_seq_id seqZero = 0;

  bool done = false;
  while (!done) {
    for (llama_token token : accepted) {
      // Check if the current token signals the end of generation:
      // - Case 1: The token is an End-of-Generation (EOG) token and stopping
      // on EOG is enabled.
      // - Case 2: The token matches any user-defined stop token.
      // If either condition is true, stop generation early.
      if ((mLib->llama_vocab_is_eog(vocab, token) &&
           aOptions.mStopOnEndOfGenerationTokens) ||
          stopTokens.has(token)) {
        LOGD("{}: Early stopping {}", __PRETTY_FUNCTION__, mModelGeneralName);
        done = true;
        break;
      }

      nsCString buffer;
      buffer.SetLength(aOptions.mDeTokenizationOptions.mMaxCharsPerToken);
      // write in the buffer from index 0
      int32_t lstrip = 0;
      auto n = mLib->llama_token_to_piece(
          vocab, token, buffer.BeginWriting(), buffer.Length(), lstrip,
          aOptions.mDeTokenizationOptions.mRenderSpecialTokens);
      if (n < 0) {
        auto msg =
            nsFmtCString("{}: failed to convert token to string piece {}",
                         __PRETTY_FUNCTION__, mModelGeneralName);
        LOGE("{}", msg);
        return mozilla::Err(Error{std::move(msg)});
      }
      buffer.SetLength(n);

      auto onTokenResult = onToken(std::move(buffer), {token},
                                   LlamaChatPhase::Generation, false);
      if (onTokenResult.isErr()) {
        LOGE("{}", onTokenResult.inspectErr().mMessage);
        return onTokenResult;
      }

      numGeneratedTokens++;
      if (numGeneratedTokens >= seqLen) {
        LOGD(
            "{} Maximum number of generation tokens reached. Stopping the "
            "generation",
            __PRETTY_FUNCTION__);
        done = true;
        break;
      }
    }
    if (done) {
      break;
    }

    if (aCancelCallback()) {
      LOGD("{} Generation successfully cancelled. {}", __PRETTY_FUNCTION__,
           mModelGeneralName);
      return mozilla::Ok();
    }

    if (nPast + 1 > nCtx) {
      LOGD("{}: Context full after {} generated tokens {}", __PRETTY_FUNCTION__,
           numGeneratedTokens, mModelGeneralName);
      break;
    }

    const llama_token idLast = accepted.LastElement();
    nsTArray<llama_token> draft;
    if (mDraftCtx) {
      const int32_t room = nCtx - nPast - 1;
      draft = Draft(idLast, nPast, pendingH, std::min(kDraftMax, room));
    }

    verifyTokens.ClearAndRetainStorage();
    verifyPos.ClearAndRetainStorage();
    verifyNSeq.ClearAndRetainStorage();
    verifySeq.ClearAndRetainStorage();
    verifyLogits.ClearAndRetainStorage();
    verifyTokens.AppendElement(idLast);
    verifyTokens.AppendElements(draft);
    for (size_t i = 0; i < verifyTokens.Length(); i++) {
      verifyPos.AppendElement(nPast + static_cast<llama_pos>(i));
      verifyNSeq.AppendElement(1);
      verifySeq.AppendElement(&seqZero);
      verifyLogits.AppendElement(1);
    }
    batch = llama_batch{static_cast<int32_t>(verifyTokens.Length()),
                        verifyTokens.Elements(),
                        nullptr,
                        verifyPos.Elements(),
                        verifyNSeq.Elements(),
                        verifySeq.Elements(),
                        verifyLogits.Elements()};

    LOGV("{}: Decoding {} tokens to generate next token probabilities {}",
         __PRETTY_FUNCTION__, batch.n_tokens, mModelGeneralName);
    if (mLib->llama_decode(mCtx.get(), batch) != 0) {
      auto msg = nsFmtCString("{}: failed to decode {}", __PRETTY_FUNCTION__,
                              mModelGeneralName);
      LOGE("{}", msg);
      return mozilla::Err(Error{std::move(msg)});
    }

    // Row i holds the distribution after token i of the batch. A draft is
    // accepted when the model samples it; the first miss ends the run, and
    // its sample is the next token either way.
    accepted.ClearAndRetainStorage();
    for (size_t i = 0; i <= draft.Length(); i++) {
      llama_token token =
          SampleToken(samplers, static_cast<int32_t>(i), candidates);
      accepted.AppendElement(token);
      if (i == draft.Length() || token != draft[i]) {
        break;
      }
    }
    const size_t nAccepted = accepted.Length() - 1;
    mDraftStats.mDrafted += draft.Length();
    mDraftStats.mAccepted += nAccepted;
    if (nAccepted < draft.Length()) {
      mLib->llama_memory_seq_rm(mLib->llama_get_memory(mCtx.get()), 0,
                                nPast + static_cast<llama_pos>(nAccepted) + 1,
                                -1);
    }
    keepHiddenState(static_cast<int32_t>(nAccepted));
    nPast += static_cast<llama_pos>(nAccepted) + 1;
  }

  LOGV("{}: Sending end of generation to callback {}", __PRETTY_FUNCTION__,
       mModelGeneralName);
  auto onTokenResult = onToken({}, {}, LlamaChatPhase::Generation, true);
  if (onTokenResult.isErr()) {
    LOGE("{}", onTokenResult.inspectErr().mMessage);
    return onTokenResult;
  }

  LOGD("{} LlamaBackend generation successfully complete {}",
       __PRETTY_FUNCTION__, mModelGeneralName);

  return mozilla::Ok();
}

}  // namespace mozilla::llama
