/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "TextGenerationParent.h"

#include "mozilla/hwinference/HWInferenceLifecycleHelpers.h"
#include "mozilla/hwinference/HWInferenceLog.h"
#include "mozilla/hwinference/HWInferenceParent.h"
#include "mozilla/ipc/Endpoint.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "mozilla/ml/MLProfilerMarkers.h"

namespace mozilla::hwinference {

#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)

TextGenerationParent::TextGenerationParent(
    RefPtr<ipc::UtilityProcessKeepAlive> aKeepAlive, bool aProcessReused)
    : mKeepAlive(std::move(aKeepAlive)), mProcessReused(aProcessReused) {}

TextGenerationParent::~TextGenerationParent() = default;

/* static */
already_AddRefed<TextGenerationParent> TextGenerationParent::Create(
    const ipc::FileDescriptor& aModel, const TextGenerationOptions& aOptions) {
  AssertIsOnMainThread();
  const bool processReused = IsBrowserProcessUp();
  const TimeStamp spawnStart = TimeStamp::Now();
  RefPtr<ipc::UtilityProcessKeepAlive> keepAlive =
      ipc::UtilityProcessManager::GetSingleton()
          ->AcquireBrowserHWInferenceProcess();
  if (!keepAlive) {
    LOGD("{} - HW_INFERENCE_BROWSER failed to launch", __func__);
    PROFILER_MARKER(ML_HWINFERENCE_PROCESS_TRACK, ML_SETUP,
                    MarkerTiming::IntervalUntilNowFrom(spawnStart),
                    MLFailedMarker, "process spawn"_ns, "launch refused"_ns);
    return nullptr;
  }
  RefPtr<HWInferenceParent> process =
      HWInferenceParent::GetSingleton(kBrowserProcessKind);
  if (!processReused) {
    process->WhenReady()->Then(
        GetMainThreadSerialEventTarget(), __func__,
        [spawnStart]() {
          PROFILER_MARKER(ML_HWINFERENCE_PROCESS_TRACK, ML_SETUP,
                          MarkerTiming::IntervalUntilNowFrom(spawnStart),
                          MLProcessSpawnMarker,
                          (TimeStamp::Now() - spawnStart).ToMilliseconds());
        },
        [spawnStart]() {
          PROFILER_MARKER(ML_HWINFERENCE_PROCESS_TRACK, ML_SETUP,
                          MarkerTiming::IntervalUntilNowFrom(spawnStart),
                          MLFailedMarker, "process spawn"_ns,
                          "process never came up"_ns);
        });
  }

  ipc::Endpoint<PTextGenerationParent> parentEnd;
  ipc::Endpoint<PTextGenerationChild> childEnd;
  MOZ_ALWAYS_SUCCEEDS(PTextGeneration::CreateEndpoints(&parentEnd, &childEnd));
  RefPtr<TextGenerationParent> generator =
      new TextGenerationParent(std::move(keepAlive), processReused);
  MOZ_ALWAYS_TRUE(parentEnd.Bind(generator));
  process->StartTextGeneration(std::move(childEnd), aModel, aOptions);
  return generator.forget();
}

RefPtr<TextGenerationParent::ReadyPromise> TextGenerationParent::WhenReady() {
  if (mLoadMs) {
    return ReadyPromise::CreateAndResolve(*mLoadMs, __func__);
  }
  if (mLoadFailure) {
    return ReadyPromise::CreateAndReject(*mLoadFailure, __func__);
  }
  return mReadyPromise.Ensure(__func__);
}

ipc::IPCResult TextGenerationParent::RecvReady(const LoadResult& aResult) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  if (aResult.type() == LoadResult::TLoadSuccess) {
    mLoadMs = Some(aResult.get_LoadSuccess().loadMs());
    mReadyPromise.ResolveIfExists(*mLoadMs, __func__);
  } else {
    mLoadFailure = Some(LoadFailure{LoadFailure::Cause::Backend,
                                    aResult.get_LoadError().message()});
    mReadyPromise.RejectIfExists(*mLoadFailure, __func__);
  }
  return IPC_OK();
}

ipc::IPCResult TextGenerationParent::RecvDelta(const nsCString& aText) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  if (mDeltaHandler) {
    mDeltaHandler(aText);
  }
  return IPC_OK();
}

void TextGenerationParent::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  if (mLoadMs.isNothing() && mLoadFailure.isNothing()) {
    mLoadFailure = Some(LoadFailure{LoadFailure::Cause::ActorGone,
                                    "TextGenerator: actor destroyed"_ns});
    mReadyPromise.RejectIfExists(*mLoadFailure, __func__);
  }
  ReleaseBrowserProcessAfterGrace(std::move(mKeepAlive));
}

#undef LOGD

}  // namespace mozilla::hwinference
