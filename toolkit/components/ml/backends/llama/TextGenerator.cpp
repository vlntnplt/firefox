/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "TextGenerator.h"

#include <algorithm>

#include "HWInferenceBrowserManagerParent.h"
#include "HWInferenceParent.h"
#include "ModelFileUtils.h"
#include "TextGenerationParent.h"
#include "mozilla/IntegerPrintfMacros.h"
#include "mozilla/dom/Blob.h"
#include "mozilla/dom/Promise.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "mozilla/ml/MLProfilerMarkers.h"
#include "nsIMLUtils.h"
#include "nsPrintfCString.h"
#include "nsServiceManagerUtils.h"

using namespace mozilla::hwinference;

namespace mozilla::dom {

NS_IMPL_CYCLE_COLLECTING_ADDREF(TextGenerator)
NS_IMPL_CYCLE_COLLECTING_RELEASE(TextGenerator)
NS_IMPL_CYCLE_COLLECTION_WRAPPERCACHE(TextGenerator, mGlobal)
NS_INTERFACE_MAP_BEGIN_CYCLE_COLLECTION(TextGenerator)
  NS_WRAPPERCACHE_INTERFACE_MAP_ENTRY
  NS_INTERFACE_MAP_ENTRY(nsISupports)
NS_INTERFACE_MAP_END

using dom::Promise;

static uint32_t ResolveNumThreads(uint32_t aRequested) {
  if (aRequested > 0) {
    return aRequested;
  }
  uint8_t cores = 1;
  nsCOMPtr<nsIMLUtils> utils = do_GetService("@mozilla.org/ml-utils;1");
  if (utils) {
    utils->GetOptimalCPUConcurrency(&cores);
  }
  return std::max<uint32_t>(1, cores);
}

// llama.cpp's LLAMA_DEFAULT_SEED.
static constexpr uint32_t kDefaultSeed = 0xFFFFFFFF;

static void MarkGeneratorCreate(TimeStamp aStart, double aBackendInitMs) {
  const double wallMs = (TimeStamp::Now() - aStart).ToMilliseconds();
  PROFILER_MARKER(AIR_TEXT_GENERATION_TRACK, ML_SETUP,
                  MarkerTiming::IntervalUntilNowFrom(aStart),
                  MLGeneratorCreateMarker, aBackendInitMs,
                  std::max(0.0, wallMs - aBackendInitMs));
}

static void MarkFailed(TimeStamp aStart, const nsCString& aPhase,
                       const nsCString& aReason) {
  PROFILER_MARKER(AIR_TEXT_GENERATION_TRACK, ML_SETUP,
                  MarkerTiming::IntervalUntilNowFrom(aStart), MLFailedMarker,
                  aPhase, aReason);
}

static void RejectFailed(Promise* aPromise, TimeStamp aStart,
                         const nsCString& aPhase, const nsCString& aMessage) {
  MarkFailed(aStart, aPhase, aMessage);
  aPromise->MaybeRejectWithOperationError(aMessage);
}

static TextGenerationOptions ToTextGenerationOptions(
    const dom::TextGeneratorCreateOptions& aOptions) {
  const uint32_t numThreads = ResolveNumThreads(aOptions.mNumThreads);
  const uint32_t numThreadsDecoding = aOptions.mNumThreadsDecoding > 0
                                          ? aOptions.mNumThreadsDecoding
                                          : numThreads;
  return TextGenerationOptions(aOptions.mContextSize, numThreads,
                               numThreadsDecoding, aOptions.mBatchSize,
                               aOptions.mUbatchSize, aOptions.mKvCacheDtype,
                               aOptions.mFlashAttn);
}

static Result<mozilla::ipc::FileDescriptor, const char*> BlobToModelFd(
    JSContext* aCx, dom::Blob& aModel) {
  JS::Rooted<JS::Value> blobValue(aCx);
  if (!GetOrCreateDOMReflector(aCx, &aModel, &blobValue)) {
    return Err("TextGenerator.create: failed to reflect model Blob");
  }
  mozilla::ipc::FileDescriptor modelFd;
  if (NS_FAILED(BlobJSObjectToFileDescriptor(aCx, blobValue, &modelFd))) {
    return Err("TextGenerator.create: model Blob is not file-backed");
  }
  return modelFd;
}

static GenerateRequest ToGenerateRequest(
    const dom::TextGenerationRequest& aRequest) {
  GenerateRequest request;
  for (const auto& message : aRequest.mMessages) {
    request.messages().AppendElement(
        ChatMessage(message.mRole, nsCString(message.mContent)));
  }
  request.maxTokens() = aRequest.mMaxTokens;
  request.bufferLength() = aRequest.mBufferLength;
  request.stopOnEndOfGenerationTokens() = aRequest.mStopOnEndOfGenerationTokens;
  for (int32_t stopToken : aRequest.mStopTokens) {
    request.stopTokens().AppendElement(stopToken);
  }
  for (const auto& sampler : aRequest.mSamplers) {
    CopyableTArray<LogitBias> logitBias;
    for (const auto& bias : sampler.mLogitBias) {
      logitBias.AppendElement(LogitBias(bias.mToken, bias.mBias));
    }
    request.samplers().AppendElement(Sampler(
        sampler.mType, sampler.mTopK, sampler.mTopP, sampler.mTemp,
        sampler.mSeed.WasPassed() ? sampler.mSeed.Value() : kDefaultSeed,
        std::move(logitBias)));
  }
  return request;
}

static dom::TextGenerationResult ToJSResult(const GenerateResult& aResult) {
  const hwinference::Usage& usage = aResult.usage();
  const ResourceUsage& resources = aResult.resources();
  dom::TextGenerationResult jsResult;
  jsResult.mContent = aResult.content();
  jsResult.mReason = aResult.reason();
  jsResult.mUsage.mPromptTokens = usage.promptTokens();
  jsResult.mUsage.mPromptCharacters = usage.promptCharacters();
  jsResult.mUsage.mGeneratedTokens = usage.generatedTokens();
  jsResult.mUsage.mTimings.mPrefillMs = usage.timings().prefillMs();
  jsResult.mUsage.mTimings.mDecodeMs = usage.timings().decodeMs();
  jsResult.mResources.mBefore.mCpuTimeMs = resources.before().cpuTimeMs();
  jsResult.mResources.mBefore.mMemoryBytes = resources.before().memoryBytes();
  jsResult.mResources.mAfter.mCpuTimeMs = resources.after().cpuTimeMs();
  jsResult.mResources.mAfter.mMemoryBytes = resources.after().memoryBytes();
  return jsResult;
}

// Main thread, outside any script-blocked section.
MOZ_CAN_RUN_SCRIPT_BOUNDARY
static void CallDeltaCallback(dom::TextGenerationDeltaCallback& aCallback,
                              const nsACString& aText) {
  aCallback.Call(aText);
}

// Feeds the run marker only.
struct DeltaStats {
  TimeStamp mFirstDeltaAt;
  double mDeliverMs = 0.0;
};

// Empty when the caller passed no callback, which is what RecvDelta tests.
static TextGenerationParent::DeltaHandler MakeDeltaHandler(
    const dom::Optional<OwningNonNull<dom::TextGenerationDeltaCallback>>&
        aOnDelta,
    const std::shared_ptr<DeltaStats>& aStats) {
  if (!aOnDelta.WasPassed()) {
    return nullptr;
  }
  RefPtr<dom::TextGenerationDeltaCallback> callback = &aOnDelta.Value();
  const bool profiling = profiler_thread_is_being_profiled_for_markers();
  // Capturing `this` would close a cycle the cycle collector cannot see.
  return [callback, aStats, profiling](const nsCString& aText) {
    TimeStamp deltaStart;
    if (profiling) {
      deltaStart = TimeStamp::Now();
      if (aStats->mFirstDeltaAt.IsNull()) {
        aStats->mFirstDeltaAt = deltaStart;
      }
    }
    CallDeltaCallback(MOZ_KnownLive(*callback), aText);
    if (profiling) {
      aStats->mDeliverMs += (TimeStamp::Now() - deltaStart).ToMilliseconds();
    }
  };
}

static void MarkGeneratorRun(TimeStamp aStart, const GenerateResult& aResult,
                             const DeltaStats& aStats, uint32_t aChunkTokens) {
  const hwinference::Usage& usage = aResult.usage();
  const double ttfcMs = aStats.mFirstDeltaAt.IsNull()
                            ? 0.0
                            : (aStats.mFirstDeltaAt - aStart).ToMilliseconds();
  const double computeMs =
      usage.timings().prefillMs() + usage.timings().decodeMs();
  const double wallMs = (TimeStamp::Now() - aStart).ToMilliseconds();
  PROFILER_MARKER(
      AIR_TEXT_GENERATION_TRACK, ML_INFERENCE,
      MarkerTiming::IntervalUntilNowFrom(aStart), MLGeneratorRunMarker,
      usage.promptTokens(), usage.generatedTokens(), ttfcMs, aChunkTokens,
      ml::TokensPerSecond(usage.promptTokens(), usage.timings().prefillMs()),
      ml::TokensPerSecond(usage.generatedTokens(), usage.timings().decodeMs()),
      computeMs, aStats.mDeliverMs,
      std::max(0.0, wallMs - computeMs - aStats.mDeliverMs),
      dom::GetEnumString(aResult.reason()));
}

TextGenerator::TextGenerator(nsIGlobalObject* aGlobal,
                             RefPtr<HWInferenceBrowserManagerParent> aManager,
                             RefPtr<TextGenerationParent> aActor)
    : mGlobal(aGlobal),
      mManager(std::move(aManager)),
      mActor(std::move(aActor)) {}

TextGenerator::~TextGenerator() { Terminate(); }

JSObject* TextGenerator::WrapObject(JSContext* aCx,
                                    JS::Handle<JSObject*> aGivenProto) {
  return dom::TextGenerator_Binding::Wrap(aCx, this, aGivenProto);
}

/* static */
already_AddRefed<Promise> TextGenerator::Create(
    const dom::GlobalObject& aGlobal, dom::Blob& aModel,
    const dom::TextGeneratorCreateOptions& aOptions) {
  nsCOMPtr<nsIGlobalObject> global = do_QueryInterface(aGlobal.GetAsSupports());
  RefPtr<Promise> promise = Promise::Create(global, IgnoreErrors());
  if (!promise) {
    return nullptr;
  }

  auto fd = BlobToModelFd(aGlobal.Context(), aModel);
  if (fd.isErr()) {
    promise->MaybeRejectWithOperationError(nsDependentCString(fd.unwrapErr()));
    return promise.forget();
  }
  mozilla::ipc::FileDescriptor modelFd = fd.unwrap();
  TextGenerationOptions options = ToTextGenerationOptions(aOptions);

  TimeStamp acquireStart = TimeStamp::Now();
  HWInferenceBrowserManagerParent::GetOrCreate()->Then(
      GetMainThreadSerialEventTarget(), __func__,
      [promise, global, modelFd, options,
       acquireStart](const RefPtr<HWInferenceBrowserManagerParent::Reservation>&
                         aReservation) {
        TimeStamp createStart = TimeStamp::Now();
        PROFILER_MARKER(AIR_TEXT_GENERATION_TRACK, ML_SETUP,
                        MarkerTiming::Interval(acquireStart, createStart),
                        MLProcessAcquireMarker, aReservation->ProcessReused());
        RefPtr<TextGenerationParent> actor =
            aReservation->CreateTextGeneration(modelFd, options);
        if (!actor) {
          RejectFailed(
              promise, createStart, "generator create"_ns,
              "TextGenerator.create: failed to construct the generator"_ns);
          return;
        }
        RefPtr<TextGenerationParent> readyActor = actor;
        RefPtr<TextGenerator> generator = new TextGenerator(
            global, aReservation->Manager(), std::move(actor));
        readyActor->WhenReady()->Then(
            GetMainThreadSerialEventTarget(), __func__,
            [promise, generator, createStart](double aBackendInitMs) {
              MarkGeneratorCreate(createStart, aBackendInitMs);
              promise->MaybeResolve(generator);
            },
            [promise, generator, createStart](const nsCString& aError) {
              generator->Terminate();
              RejectFailed(promise, createStart, "generator create"_ns, aError);
            });
      },
      [promise, acquireStart](nsresult aError) {
        MarkFailed(acquireStart, "process acquire"_ns,
                   "no inference process available"_ns);
        promise->MaybeRejectWithOperationError(nsPrintfCString(
            "TextGenerator.create: failed to start the inference process "
            "(0x%08" PRIx32 ")",
            static_cast<uint32_t>(aError)));
      });

  return promise.forget();
}

already_AddRefed<Promise> TextGenerator::Generate(
    const dom::TextGenerationRequest& aRequest,
    const dom::Optional<OwningNonNull<dom::TextGenerationDeltaCallback>>&
        aOnDelta,
    ErrorResult& aRv) {
  RefPtr<Promise> promise = Promise::Create(mGlobal, aRv);
  if (aRv.Failed()) {
    return nullptr;
  }
  if (!mActor || !mActor->CanSend()) {
    promise->MaybeRejectWithInvalidStateError(
        "TextGenerator.generate: generator is terminated");
    return promise.forget();
  }
  if (mGenerateInFlight) {
    promise->MaybeRejectWithInvalidStateError(
        "TextGenerator.generate: a generation is already in flight");
    return promise.forget();
  }

  GenerateRequest request = ToGenerateRequest(aRequest);

  TimeStamp generateStart = TimeStamp::Now();
  auto stats = std::make_shared<DeltaStats>();
  mActor->SetDeltaHandler(MakeDeltaHandler(aOnDelta, stats));

  mGenerateInFlight = true;
  RefPtr<TextGenerator> self = this;
  const uint32_t chunkTokens = request.bufferLength();
  mActor->SendGenerate(request)->Then(
      GetMainThreadSerialEventTarget(), __func__,
      [promise, self, generateStart, stats,
       chunkTokens](const GenerateResponse& aResponse) {
        self->OnGenerateSettled();
        if (aResponse.type() == GenerateResponse::TGenerateError) {
          // Inference-phase category; RejectFailed emits under ML_SETUP.
          PROFILER_MARKER(AIR_TEXT_GENERATION_TRACK, ML_INFERENCE,
                          MarkerTiming::IntervalUntilNowFrom(generateStart),
                          MLFailedMarker, "generate"_ns,
                          aResponse.get_GenerateError().message());
          promise->MaybeRejectWithOperationError(
              aResponse.get_GenerateError().message());
          return;
        }
        const GenerateResult& result = aResponse.get_GenerateResult();
        MarkGeneratorRun(generateStart, result, *stats, chunkTokens);
        promise->MaybeResolve(ToJSResult(result));
      },
      [promise, self, generateStart](mozilla::ipc::ResponseRejectReason) {
        self->OnGenerateSettled();
        if (self->mTeardownRequested) {
          PROFILER_MARKER(AIR_TEXT_GENERATION_TRACK, ML_INFERENCE,
                          MarkerTiming::IntervalUntilNowFrom(generateStart),
                          MLCancelMarker, "teardown"_ns);
          promise->MaybeRejectWithAbortError(
              "TextGenerator.generate: the generator was terminated");
          return;
        }
        PROFILER_MARKER(AIR_TEXT_GENERATION_TRACK, ML_INFERENCE,
                        MarkerTiming::IntervalUntilNowFrom(generateStart),
                        MLFailedMarker, "generate"_ns,
                        "the generator went away"_ns);
        promise->MaybeRejectWithAbortError(
            "TextGenerator.generate: the generator went away");
      });

  return promise.forget();
}

void TextGenerator::OnGenerateSettled() {
  mGenerateInFlight = false;
  if (mActor) {
    mActor->SetDeltaHandler(nullptr);
  }
}

void TextGenerator::Clear() {
  if (mActor && mActor->CanSend()) {
    (void)mActor->SendClear();
  }
}

void TextGenerator::Cancel() {
  if (mActor && mActor->CanSend()) {
    PROFILER_MARKER(AIR_TEXT_GENERATION_TRACK, ML_INFERENCE, {}, MLCancelMarker,
                    "requested"_ns);
    (void)mActor->SendCancel();
  }
}

void TextGenerator::Terminate() {
  mTeardownRequested = true;
  if (mActor && mActor->CanSend()) {
    (void)PTextGenerationParent::Send__delete__(mActor);
  }
  mActor = nullptr;
  mManager = nullptr;
}

}  // namespace mozilla::dom
