/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "HWInferenceBrowserManagerParent.h"

#include "HWInferenceParent.h"
#include "mozilla/Logging.h"
#include "mozilla/ipc/Endpoint.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "mozilla/ipc/UtilityProcessParent.h"

namespace mozilla::hwinference {

extern LazyLogModule gHWInferenceLog;
#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Error, fmt, ##__VA_ARGS__)

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
            if (!aRoot || !aRoot->CanSend()) {
              LOGE("Create - ERROR: HWInference root actor lost");
              return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
            }

            RefPtr<ipc::UtilityProcessParent> processParent =
                manager->GetProcessParent(ipc::SandboxingKind::HW_INFERENCE,
                                          HWINFERENCE_BROWSER_INSTANCE_KEY);
            if (!processParent) {
              LOGE("Create - ERROR: utility process parent lost");
              return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
            }

            Endpoint<PHWInferenceBrowserManagerParent> parentEnd;
            Endpoint<PHWInferenceBrowserManagerChild> childEnd;
            MOZ_ALWAYS_SUCCEEDS(PHWInferenceBrowserManager::CreateEndpoints(
                ipc::EndpointProcInfo::Current(),
                processParent->OtherEndpointProcInfo(), &parentEnd, &childEnd));

            if (!aRoot->SendNewBrowserHWInferenceManager(std::move(childEnd))) {
              LOGE("Create - ERROR: SendNewBrowserHWInferenceManager failed");
              return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
            }

            RefPtr<HWInferenceBrowserManagerParent> actor =
                new HWInferenceBrowserManagerParent();
            MOZ_ALWAYS_TRUE(parentEnd.Bind(actor));
            LOGD("Create - browser inference manager bound");
            return CreatePromise::CreateAndResolve(std::move(actor), __func__);
          },
          [](ipc::LaunchError aError) {
            LOGE("Create - ERROR: HWInference launch failed in {}",
                 aError.FunctionName().get());
            return CreatePromise::CreateAndReject(NS_ERROR_FAILURE, __func__);
          });
}

void HWInferenceBrowserManagerParent::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
}

#undef LOGD
#undef LOGE

}  // namespace mozilla::hwinference
