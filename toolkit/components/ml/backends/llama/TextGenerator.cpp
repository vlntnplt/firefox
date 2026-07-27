/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "TextGenerator.h"

#include <algorithm>

#include "HWInferenceBrowserManagerParent.h"
#include "HWInferenceParent.h"
#include "ModelFileUtils.h"
#include "TextGenerationMarkers.h"
#include "TextGenerationParent.h"
#include "mozilla/IntegerPrintfMacros.h"
#include "mozilla/dom/Blob.h"
#include "mozilla/dom/Promise.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "nsIMLUtils.h"
#include "nsPrintfCString.h"
#include "nsServiceManagerUtils.h"

namespace mozilla::hwinference {

NS_IMPL_CYCLE_COLLECTING_ADDREF(TextGenerator)
NS_IMPL_CYCLE_COLLECTING_RELEASE(TextGenerator)
NS_IMPL_CYCLE_COLLECTION_WRAPPERCACHE(TextGenerator, mGlobal)
NS_INTERFACE_MAP_BEGIN_CYCLE_COLLECTION(TextGenerator)
  NS_WRAPPERCACHE_INTERFACE_MAP_ENTRY
  NS_INTERFACE_MAP_ENTRY(nsISupports)
NS_INTERFACE_MAP_END

using dom::Promise;

// The one place the thread-count 0 (auto) defaults resolve, to the
// physical-core count. The wire then carries concrete counts, so the
// sandboxed child never samples core topology itself.
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

// The default seed llama.cpp uses when none is provided
// (LLAMA_DEFAULT_SEED); applied when chrome JS omits the optional seed.
static constexpr uint32_t kDefaultSeed = 0xFFFFFFFF;

// Deltas arrive from IPC on the main thread, outside any script-blocked
// section, so running the chrome JS callback here is safe.
MOZ_CAN_RUN_SCRIPT_BOUNDARY
static void CallDeltaCallback(dom::TextGenerationDeltaCallback& aCallback,
                              const nsACString& aText) {
  aCallback.Call(aText);
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

  ipc::FileDescriptor modelFd;
  JS::Rooted<JS::Value> blobValue(aGlobal.Context());
  if (!GetOrCreateDOMReflector(aGlobal.Context(), &aModel, &blobValue)) {
    promise->MaybeRejectWithOperationError(
        "TextGenerator.create: failed to reflect model Blob");
    return promise.forget();
  }
  nsresult rv =
      BlobJSObjectToFileDescriptor(aGlobal.Context(), blobValue, &modelFd);
  if (NS_FAILED(rv)) {
    promise->MaybeRejectWithOperationError(
        "TextGenerator.create: model Blob is not file-backed");
    return promise.forget();
  }

  const uint32_t numThreads = ResolveNumThreads(aOptions.mNumThreads);
  const uint32_t numThreadsDecoding = aOptions.mNumThreadsDecoding > 0
                                          ? aOptions.mNumThreadsDecoding
                                          : numThreads;
  TextGenerationOptions options(aOptions.mContextSize, numThreads,
                                numThreadsDecoding, aOptions.mBatchSize,
                                aOptions.mUbatchSize, aOptions.mKvCacheDtype,
                                aOptions.mFlashAttn);

  TimeStamp createStart = TimeStamp::Now();
  HWInferenceBrowserManagerParent::GetOrCreate()->Then(
      GetMainThreadSerialEventTarget(), __func__,
      [promise, global, modelFd, options,
       createStart](RefPtr<HWInferenceBrowserManagerParent> aManager) {
        RefPtr<TextGenerationParent> actor =
            aManager->CreateTextGeneration(modelFd, options);
        if (!actor) {
          PROFILER_MARKER("TextGenerator create", ML_SETUP,
                          MarkerTiming::IntervalUntilNowFrom(createStart),
                          TextGenerationCreateMarker,
                          "failed to construct the generator"_ns);
          promise->MaybeRejectWithOperationError(
              "TextGenerator.create: failed to construct the generator");
          return;
        }
        RefPtr<TextGenerationParent> readyActor = actor;
        RefPtr<TextGenerator> generator =
            new TextGenerator(global, std::move(aManager), std::move(actor));
        // Construction is acquisition: hand the generator out only once
        // its model loaded, and tear it down with the load error if not.
        readyActor->WhenReady()->Then(
            GetMainThreadSerialEventTarget(), __func__,
            [promise, generator, createStart](double) {
              PROFILER_MARKER("TextGenerator create", ML_SETUP,
                              MarkerTiming::IntervalUntilNowFrom(createStart),
                              TextGenerationCreateMarker, "ok"_ns);
              promise->MaybeResolve(generator);
            },
            [promise, generator, createStart](const nsCString& aError) {
              PROFILER_MARKER("TextGenerator create", ML_SETUP,
                              MarkerTiming::IntervalUntilNowFrom(createStart),
                              TextGenerationCreateMarker, aError);
              generator->Terminate();
              promise->MaybeRejectWithOperationError(aError);
            });
      },
      [promise, createStart](nsresult aError) {
        PROFILER_MARKER("TextGenerator create", ML_SETUP,
                        MarkerTiming::IntervalUntilNowFrom(createStart),
                        TextGenerationCreateMarker,
                        "failed to start the inference process"_ns);
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

  if (aOnDelta.WasPassed()) {
    RefPtr<dom::TextGenerationDeltaCallback> callback = &aOnDelta.Value();
    mActor->SetDeltaHandler([callback](const nsCString& aText) {
      CallDeltaCallback(MOZ_KnownLive(*callback), aText);
    });
  } else {
    mActor->SetDeltaHandler(nullptr);
  }

  // `self` keeps the wrapper (and through it the actor) alive until the
  // reply settles, so dropping the last JS reference mid-generation does
  // not tear the generation down at GC time.
  mGenerateInFlight = true;
  RefPtr<TextGenerator> self = this;
  TimeStamp generateStart = TimeStamp::Now();
  mActor->SendGenerate(request)->Then(
      GetMainThreadSerialEventTarget(), __func__,
      [promise, self, generateStart](const GenerateResponse& aResponse) {
        self->OnGenerateSettled();
        if (aResponse.type() == GenerateResponse::TGenerateError) {
          PROFILER_MARKER("TextGenerator generate", ML_INFERENCE,
                          MarkerTiming::IntervalUntilNowFrom(generateStart),
                          TextGenerationGenerateMarker, 0, 0,
                          aResponse.get_GenerateError().message());
          promise->MaybeRejectWithOperationError(
              aResponse.get_GenerateError().message());
          return;
        }
        const GenerateResult& result = aResponse.get_GenerateResult();
        PROFILER_MARKER("TextGenerator generate", ML_INFERENCE,
                        MarkerTiming::IntervalUntilNowFrom(generateStart),
                        TextGenerationGenerateMarker,
                        result.usage().promptTokens(),
                        result.usage().generatedTokens(),
                        dom::GetEnumString(result.reason()));
        dom::TextGenerationResult jsResult;
        jsResult.mContent = result.content();
        jsResult.mReason = result.reason();
        jsResult.mUsage.mPromptTokens = result.usage().promptTokens();
        jsResult.mUsage.mGeneratedTokens = result.usage().generatedTokens();
        jsResult.mUsage.mTimings.mPrefillMs =
            result.usage().timings().prefillMs();
        jsResult.mUsage.mTimings.mDecodeMs =
            result.usage().timings().decodeMs();
        promise->MaybeResolve(jsResult);
      },
      [promise, self, generateStart](ipc::ResponseRejectReason aReason) {
        self->OnGenerateSettled();
        PROFILER_MARKER("TextGenerator generate", ML_INFERENCE,
                        MarkerTiming::IntervalUntilNowFrom(generateStart),
                        TextGenerationGenerateMarker, 0, 0,
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
    (void)mActor->SendCancel();
  }
}

void TextGenerator::Terminate() {
  if (mActor && mActor->CanSend()) {
    (void)PTextGenerationParent::Send__delete__(mActor);
  }
  mActor = nullptr;
  mManager = nullptr;
}

}  // namespace mozilla::hwinference
