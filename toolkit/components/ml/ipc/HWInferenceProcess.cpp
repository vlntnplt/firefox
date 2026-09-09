/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "HWInferenceProcess.h"

#include "HWInferenceParent.h"
#include "mozilla/ClearOnShutdown.h"
#include "HWInferenceLog.h"
#include "mozilla/StaticPrefs_browser.h"
#include "mozilla/StaticPtr.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "mozilla/ml/MLProfilerMarkers.h"
#include "nsThreadUtils.h"

namespace mozilla::hwinference {

#define LOGD(...) MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, __VA_ARGS__)

static StaticAutoPtr<HWInferenceProcess> sContent;
static StaticAutoPtr<HWInferenceProcess> sBrowser;

static HWInferenceProcess& Shared(StaticAutoPtr<HWInferenceProcess>& aSlot,
                                  HWInferenceProcess::Kind aKind) {
  AssertIsOnMainThread();
  if (!aSlot) {
    aSlot = new HWInferenceProcess(aKind);
    ClearOnShutdown(&aSlot);
  }
  return *aSlot;
}

/* static */
HWInferenceProcess& HWInferenceProcess::Content() {
  return Shared(sContent, Kind::Content);
}

/* static */
HWInferenceProcess& HWInferenceProcess::Browser() {
  return Shared(sBrowser, Kind::Browser);
}

Maybe<uint32_t> HWInferenceProcess::RestartBudget() const {
  switch (mKind) {
    case Kind::Content:
      return Some(StaticPrefs::browser_ml_hwinference_max_restarts());
    case Kind::Browser:
      return Nothing();
  }
  MOZ_ASSERT_UNREACHABLE("unknown HWInferenceProcess::Kind");
  return Nothing();
}

uint32_t HWInferenceProcess::GraceMs() const {
  switch (mKind) {
    case Kind::Content:
      return 0;
    case Kind::Browser:
      return StaticPrefs::
          browser_ml_hwinference_browser_idle_shutdown_grace_ms();
  }
  MOZ_ASSERT_UNREACHABLE("unknown HWInferenceProcess::Kind");
  return 0;
}

HWInferenceProcess::~HWInferenceProcess() {
  if (mActor) {
    RetireActor(mActor);
  }
}

RefPtr<HWInferenceParent> HWInferenceProcess::Actor() const {
  AssertIsOnMainThread();
  return mActor;
}

bool HWInferenceProcess::IsUp() const {
  AssertIsOnMainThread();
  RefPtr<ipc::UtilityProcessKeepAlive> keepAlive = mKeepAlive.get();
  return keepAlive && keepAlive->IsAlive() && mActor;
}

already_AddRefed<ipc::UtilityProcessKeepAlive> HWInferenceProcess::Acquire() {
  AssertIsOnMainThread();

  if (IsUp()) {
    return do_AddRef(mKeepAlive.get());
  }

  // The keep-alive died before the actor heard about it. A crash destroys the
  // actor first, so an actor that can still send is going through a clean
  // shutdown; an unbound one is a cancelled launch.
  if (mActor) {
    LOGD("{} - retiring the actor", __func__);
    if (mActor->CanSend()) {
      mRestarts = 0;
    }
    RetireActor(mActor);
  }

  // Only a relaunch is refused: a spent budget must not take a working process
  // away from its consumers.
  if (Maybe<uint32_t> budget = RestartBudget();
      budget && mRestarts >= *budget) {
    LOGD("{} - restart budget spent, not relaunching", __func__);
    return nullptr;
  }

  RefPtr<ipc::UtilityProcessManager> upm =
      ipc::UtilityProcessManager::GetSingleton();
  if (!upm) {
    return nullptr;
  }

  RefPtr<HWInferenceParent> actor = new HWInferenceParent();
  actor->mOwner = this;
  RefPtr<ipc::UtilityProcessKeepAlive> keepAlive =
      upm->LaunchIndependentHWInferenceProcess(actor);
  if (!keepAlive) {
    LOGD("{} - launch refused", __func__);
    actor->mOwner = nullptr;
    mRestarts++;
    return nullptr;
  }

  mActor = actor;
  mKeepAlive = keepAlive;
  return keepAlive.forget();
}

// Retired: this was the last keep-alive.
static void DropKeepAlive(RefPtr<ipc::UtilityProcessKeepAlive>&& aKeepAlive,
                          TimeStamp aReleased, uint32_t aGraceMs) {
  WeakPtr<ipc::UtilityProcessKeepAlive> weak = aKeepAlive.get();
  aKeepAlive = nullptr;
  const bool retired = !weak.get();
  LOGD("DropKeepAlive - keep-alive dropped, process {}",
       retired ? "retired" : "still in use");
  PROFILER_MARKER(ML_HWINFERENCE_PROCESS_TRACK, ML_SETUP,
                  MarkerTiming::IntervalUntilNowFrom(aReleased),
                  MLProcessReleaseMarker, aGraceMs, retired);
}

void HWInferenceProcess::Release(
    RefPtr<ipc::UtilityProcessKeepAlive>&& aKeepAlive) {
  AssertIsOnMainThread();

  const TimeStamp released = TimeStamp::Now();
  const uint32_t graceMs = GraceMs();
  if (!graceMs) {
    DropKeepAlive(std::move(aKeepAlive), released, graceMs);
    return;
  }

  LOGD("{} - dropping the keep-alive in {}ms", __func__, graceMs);
  // Not a timer: nothing would hold it, and the main thread drops the runnable
  // at shutdown.
  NS_DelayedDispatchToCurrentThread(
      NS_NewRunnableFunction(
          "hwinference::HWInferenceProcess::Release",
          [keepAlive = std::move(aKeepAlive), released, graceMs]() mutable {
            DropKeepAlive(std::move(keepAlive), released, graceMs);
          }),
      graceMs);
}

void HWInferenceProcess::OnActorDestroyed(
    HWInferenceParent* aActor, ipc::IProtocol::ActorDestroyReason aReason) {
  AssertIsOnMainThread();
  MOZ_ASSERT(aActor == mActor);

  // The reason is PUtilityProcess's: NormalShutdown is a clean shutdown.
  mRestarts = aReason == ipc::IProtocol::NormalShutdown ? 0 : mRestarts + 1;
  RetireActor(aActor);
}

void HWInferenceProcess::RetireActor(HWInferenceParent* aActor) {
  aActor->mOwner = nullptr;
  if (mActor == aActor) {
    mActor = nullptr;
  }
}

#undef LOGD

}  // namespace mozilla::hwinference
