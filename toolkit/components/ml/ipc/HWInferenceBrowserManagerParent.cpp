/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "HWInferenceBrowserManagerParent.h"

#include "HWInferenceLog.h"
#include "HWInferenceParent.h"
#include "mozilla/StaticPrefs_browser.h"
#include "mozilla/StaticPtr.h"
#include "mozilla/ipc/Endpoint.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "mozilla/ipc/UtilityProcessParent.h"

namespace mozilla::hwinference {

// Main thread only.
static StaticRefPtr<HWInferenceBrowserManagerParent> sManager;
static StaticRefPtr<HWInferenceBrowserManagerParent::CreatePromise>
    sPendingCreate;

#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Error, fmt, ##__VA_ARGS__)

HWInferenceBrowserManagerParent::Reservation::Reservation(
    HWInferenceBrowserManagerParent* aManager, bool aProcessReused)
    : mManager(aManager), mProcessReused(aProcessReused) {
  AssertIsOnMainThread();
  mManager->mReservations++;
}

HWInferenceBrowserManagerParent::Reservation::~Reservation() {
  AssertIsOnMainThread();
  MOZ_ASSERT(mManager->mReservations > 0);
  mManager->mReservations--;
  mManager->MaybeShutDownProcess();
}

/* static */
RefPtr<HWInferenceBrowserManagerParent::CreatePromise>
HWInferenceBrowserManagerParent::GetOrCreate() {
  AssertIsOnMainThread();
  if (sManager && sManager->CanSend()) {
    sManager->CancelIdleTimer();
    return CreatePromise::CreateAndResolve(
        MakeRefPtr<Reservation>(sManager.get(), /* aProcessReused */ true),
        __func__);
  }
  if (sPendingCreate) {
    return RefPtr<CreatePromise>(sPendingCreate.get());
  }
  RefPtr<CreatePromise> pending = Create();
  sPendingCreate = pending;
  pending->Then(
      GetMainThreadSerialEventTarget(), __func__,
      [](const RefPtr<Reservation>& aReservation) {
        sManager = aReservation->Manager();
        sPendingCreate = nullptr;
      },
      [](nsresult) { sPendingCreate = nullptr; });
  return pending;
}

/* static */
RefPtr<HWInferenceBrowserManagerParent::CreatePromise>
HWInferenceBrowserManagerParent::Create() {
  AssertIsOnMainThread();
#ifdef ANDROID
  // There is no HWInference process on Android; see UtilityProcessSandboxing.h.
  return CreatePromise::CreateAndReject(NS_ERROR_NOT_AVAILABLE, __func__);
#else
  RefPtr<ipc::UtilityProcessManager> manager =
      ipc::UtilityProcessManager::GetSingleton();
  if (!manager) {
    return CreatePromise::CreateAndReject(NS_ERROR_NOT_AVAILABLE, __func__);
  }

  return manager->StartHWInference(HWINFERENCE_BROWSER_INSTANCE_KEY)
      ->Then(
          GetMainThreadSerialEventTarget(), __func__,
          [manager](RefPtr<HWInferenceParent> aRoot) -> RefPtr<CreatePromise> {
            manager->AcquireHWInferenceKeepAlive(
                HWINFERENCE_BROWSER_INSTANCE_KEY);
            auto shutDown = [&manager] {
              manager->ReleaseHWInferenceKeepAlive(
                  HWINFERENCE_BROWSER_INSTANCE_KEY);
            };
            if (!aRoot || !aRoot->CanSend()) {
              LOGE("Create - ERROR: HWInference root actor lost");
              shutDown();
              return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
            }

            RefPtr<ipc::UtilityProcessParent> processParent =
                manager->GetProcessParent(ipc::SandboxingKind::HW_INFERENCE,
                                          HWINFERENCE_BROWSER_INSTANCE_KEY);
            if (!processParent) {
              LOGE("Create - ERROR: utility process parent lost");
              shutDown();
              return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
            }

            Endpoint<PHWInferenceBrowserManagerParent> parentEnd;
            Endpoint<PHWInferenceBrowserManagerChild> childEnd;
            MOZ_ALWAYS_SUCCEEDS(PHWInferenceBrowserManager::CreateEndpoints(
                ipc::EndpointProcInfo::Current(),
                processParent->OtherEndpointProcInfo(), &parentEnd, &childEnd));

            if (!aRoot->SendNewBrowserHWInferenceManager(std::move(childEnd))) {
              LOGE("Create - ERROR: SendNewBrowserHWInferenceManager failed");
              shutDown();
              return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
            }

            RefPtr<HWInferenceBrowserManagerParent> actor =
                new HWInferenceBrowserManagerParent();
            MOZ_ALWAYS_TRUE(parentEnd.Bind(actor));
            actor->mHoldsKeepAlive = true;
            LOGD("Create - browser inference manager bound");
            return CreatePromise::CreateAndResolve(
                MakeRefPtr<Reservation>(actor, /* aProcessReused */ false),
                __func__);
          },
          [](ipc::LaunchError aError) {
            LOGE("Create - ERROR: HWInference launch failed in {}",
                 aError.FunctionName().get());
            return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
          });
#endif  // ANDROID
}

bool HWInferenceBrowserManagerParent::IsIdle() {
  AssertIsOnMainThread();
  return mReservations == 0;
}

void HWInferenceBrowserManagerParent::MaybeShutDownProcess() {
  // Dispatched so the dying reservation or generator is gone before IsIdle().
  RefPtr<HWInferenceBrowserManagerParent> self = this;
  NS_DispatchToMainThread(NS_NewRunnableFunction(
      "HWInferenceBrowserManagerParent::MaybeShutDownProcess", [self] {
        if (!self->CanSend()) {
          self->RetireForIdle();
          return;
        }
        if (!self->IsIdle()) {
          return;
        }
        const uint32_t timeoutMs =
            StaticPrefs::browser_ml_hwInference_processTimeout();
        if (!timeoutMs) {
          LOGD("[{} - {}] idle, releasing the keep-alive", fmt::ptr(self.get()),
               "MaybeShutDownProcess");
          self->RetireForIdle();
          return;
        }
        LOGD("[{} - {}] idle, keeping the process for {}ms",
             fmt::ptr(self.get()), "MaybeShutDownProcess", timeoutMs);
        self->ArmIdleTimer(timeoutMs);
      }));
}

void HWInferenceBrowserManagerParent::ArmIdleTimer(uint32_t aTimeoutMs) {
  AssertIsOnMainThread();
  CancelIdleTimer();
  RefPtr<HWInferenceBrowserManagerParent> self = this;
  NS_NewTimerWithCallback(
      getter_AddRefs(mIdleTimer),
      [self](nsITimer*) {
        if (!self->CanSend() || self->IsIdle()) {
          LOGD("[{} - {}] idle timeout expired, releasing the keep-alive",
               fmt::ptr(self.get()), "ArmIdleTimer");
          self->RetireForIdle();
        }
      },
      aTimeoutMs, nsITimer::TYPE_ONE_SHOT,
      "HWInferenceBrowserManagerParent::mIdleTimer"_ns);
}

void HWInferenceBrowserManagerParent::CancelIdleTimer() {
  AssertIsOnMainThread();
  if (mIdleTimer) {
    mIdleTimer->Cancel();
    mIdleTimer = nullptr;
  }
}

void HWInferenceBrowserManagerParent::RetireForIdle() {
  AssertIsOnMainThread();
  CancelIdleTimer();
  // Cleared before ActorDestroy so a GetOrCreate in between relaunches.
  if (sManager == this) {
    sManager = nullptr;
  }
  ReleaseKeepAlive();
}

void HWInferenceBrowserManagerParent::ReleaseKeepAlive() {
  if (!mHoldsKeepAlive) {
    return;
  }
  mHoldsKeepAlive = false;
  // Null during XPCOM shutdown, where ClearOnShutdown drops the counts.
  if (RefPtr<ipc::UtilityProcessManager> manager =
          ipc::UtilityProcessManager::GetSingleton()) {
    manager->ReleaseHWInferenceKeepAlive(HWINFERENCE_BROWSER_INSTANCE_KEY);
  }
}

void HWInferenceBrowserManagerParent::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  CancelIdleTimer();
  if (sManager == this) {
    sManager = nullptr;
  }
  ReleaseKeepAlive();
}

#undef LOGD
#undef LOGE

}  // namespace mozilla::hwinference
