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

// Empty when the caller passed no callback, which is what RecvDelta tests.
static TextGenerationParent::DeltaHandler MakeDeltaHandler(
    const dom::Optional<OwningNonNull<dom::TextGenerationDeltaCallback>>&
        aOnDelta) {
  if (!aOnDelta.WasPassed()) {
    return nullptr;
  }
  RefPtr<dom::TextGenerationDeltaCallback> callback = &aOnDelta.Value();
  // Capturing `this` would close a cycle the cycle collector cannot see.
  return [callback](const nsCString& aText) {
    CallDeltaCallback(MOZ_KnownLive(*callback), aText);
  };
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

  HWInferenceBrowserManagerParent::GetOrCreate()->Then(
      GetMainThreadSerialEventTarget(), __func__,
      [promise, global, modelFd,
       options](const RefPtr<HWInferenceBrowserManagerParent::Reservation>&
                    aReservation) {
        RefPtr<TextGenerationParent> actor =
            aReservation->CreateTextGeneration(modelFd, options);
        if (!actor) {
          promise->MaybeRejectWithOperationError(
              "TextGenerator.create: failed to construct the generator");
          return;
        }
        RefPtr<TextGenerationParent> readyActor = actor;
        RefPtr<TextGenerator> generator = new TextGenerator(
            global, aReservation->Manager(), std::move(actor));
        readyActor->WhenReady()->Then(
            GetMainThreadSerialEventTarget(), __func__,
            [promise, generator](double) { promise->MaybeResolve(generator); },
            [promise, generator](const nsCString& aError) {
              generator->Terminate();
              promise->MaybeRejectWithOperationError(aError);
            });
      },
      [promise](nsresult aError) {
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
  mActor->SetDeltaHandler(MakeDeltaHandler(aOnDelta));

  mGenerateInFlight = true;
  RefPtr<TextGenerator> self = this;
  mActor->SendGenerate(request)->Then(
      GetMainThreadSerialEventTarget(), __func__,
      [promise, self](const GenerateResponse& aResponse) {
        self->OnGenerateSettled();
        if (aResponse.type() == GenerateResponse::TGenerateError) {
          promise->MaybeRejectWithOperationError(
              aResponse.get_GenerateError().message());
          return;
        }
        const GenerateResult& result = aResponse.get_GenerateResult();
        promise->MaybeResolve(ToJSResult(result));
      },
      [promise, self](mozilla::ipc::ResponseRejectReason) {
        self->OnGenerateSettled();
        if (self->mTeardownRequested) {
          promise->MaybeRejectWithAbortError(
              "TextGenerator.generate: the generator was terminated");
          return;
        }
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
  mTeardownRequested = true;
  if (mActor && mActor->CanSend()) {
    (void)PTextGenerationParent::Send__delete__(mActor);
  }
  mActor = nullptr;
  mManager = nullptr;
}

}  // namespace mozilla::dom
