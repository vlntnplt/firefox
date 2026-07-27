/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "HWInferenceBrowserManagerParent.h"

#include "HWInferenceLog.h"
#include "HWInferenceParent.h"
#include "TextGenerationParent.h"
#include "mozilla/StaticPtr.h"
#include "mozilla/ipc/Endpoint.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "mozilla/ipc/UtilityProcessParent.h"

namespace mozilla::hwinference {

// Main thread only. sManager is the live shared manager (cleared by its
// ActorDestroy); sPendingCreate is the launch in progress that late
// GetOrCreate() callers join.
static StaticRefPtr<HWInferenceBrowserManagerParent> sManager;
static StaticRefPtr<HWInferenceBrowserManagerParent::CreatePromise>
    sPendingCreate;

#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Error, fmt, ##__VA_ARGS__)

/* static */
RefPtr<HWInferenceBrowserManagerParent::CreatePromise>
HWInferenceBrowserManagerParent::GetOrCreate() {
  AssertIsOnMainThread();
  if (sManager && sManager->CanSend()) {
    sManager->mPendingGenerators++;
    return CreatePromise::CreateAndResolve(
        RefPtr<HWInferenceBrowserManagerParent>(sManager.get()), __func__);
  }
  if (sPendingCreate) {
    return RefPtr<CreatePromise>(sPendingCreate.get());
  }
  RefPtr<CreatePromise> pending = Create();
  sPendingCreate = pending;
  pending->Then(
      GetMainThreadSerialEventTarget(), __func__,
      [](RefPtr<HWInferenceBrowserManagerParent> aManager) {
        sManager = aManager.get();
        sPendingCreate = nullptr;
      },
      [](nsresult) { sPendingCreate = nullptr; });
  return pending;
}

/* static */
RefPtr<HWInferenceBrowserManagerParent::CreatePromise>
HWInferenceBrowserManagerParent::Create() {
  AssertIsOnMainThread();
  RefPtr<ipc::UtilityProcessManager> manager =
      ipc::UtilityProcessManager::GetSingleton();
  if (!manager) {
    return CreatePromise::CreateAndReject(NS_ERROR_NOT_AVAILABLE, __func__);
  }

  return manager->StartHWInference(HWINFERENCE_BROWSER_INSTANCE_KEY)
      ->Then(
          GetMainThreadSerialEventTarget(), __func__,
          [manager](RefPtr<HWInferenceParent> aRoot) -> RefPtr<CreatePromise> {
            // The keep-alive is what keeps the process up; it moves to the
            // manager actor on success. A failure below releases it, which
            // shuts the process down unless another consumer holds one.
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
            return CreatePromise::CreateAndResolve(std::move(actor), __func__);
          },
          [](ipc::LaunchError aError) {
            LOGE("Create - ERROR: HWInference launch failed in {}",
                 aError.FunctionName().get());
            return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
          });
}

RefPtr<TextGenerationParent>
HWInferenceBrowserManagerParent::CreateTextGeneration(
    const ipc::FileDescriptor& aModel, const TextGenerationOptions& aOptions) {
  AssertIsOnMainThread();
  if (mPendingGenerators) {
    mPendingGenerators--;
  }
  if (!CanSend()) {
    return nullptr;
  }
  RefPtr<TextGenerationParent> generator = new TextGenerationParent();
  if (!SendPTextGenerationConstructor(generator, aModel, aOptions)) {
    MaybeShutDownProcess();
    return nullptr;
  }
  return generator;
}

void HWInferenceBrowserManagerParent::OnTextGenerationDestroyed() {
  MaybeShutDownProcess();
}

void HWInferenceBrowserManagerParent::MaybeShutDownProcess() {
  // A dying actor is still linked; decide after the managed set
  // settles.
  RefPtr<HWInferenceBrowserManagerParent> self = this;
  NS_DispatchToMainThread(NS_NewRunnableFunction(
      "HWInferenceBrowserManagerParent::MaybeShutDownProcess", [self] {
        if (!self->CanSend() || self->mPendingGenerators > 0 ||
            !self->ManagedPTextGenerationParent().IsEmpty()) {
          return;
        }
        LOGD("[{} - {}] last generator gone, releasing the process keep-alive",
             fmt::ptr(self.get()), "MaybeShutDownProcess");
        // Retire the cached singleton now, not at ActorDestroy: a
        // GetOrCreate between the two must launch afresh instead of
        // joining a manager whose process is going away.
        if (sManager == self.get()) {
          sManager = nullptr;
        }
        self->ReleaseKeepAlive();
      }));
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
  if (sManager == this) {
    sManager = nullptr;
  }
  // Covers process crash and any teardown that did not go through
  // MaybeShutDownProcess; without this the count never reaches zero and
  // the next process instance under this key would never shut down.
  ReleaseKeepAlive();
}

#undef LOGD
#undef LOGE

}  // namespace mozilla::hwinference
