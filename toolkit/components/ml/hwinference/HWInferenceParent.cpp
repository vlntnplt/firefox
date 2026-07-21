/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "mozilla/StaticPtr.h"
#include "HWInferenceParent.h"
#include "HWInferenceManagerParent.h"
#include "mozilla/dom/Blob.h"
#include "mozilla/dom/BlobBinding.h"
#include "mozilla/ipc/UtilityProcessParent.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "mozilla/dom/Promise.h"
#include "mozilla/dom/Promise-inl.h"
#include "mozilla/dom/FileBlobImpl.h"
#include "mozilla/dom/IPCBlobUtils.h"
#include "mozilla/dom/Blob.h"
#include "mozilla/dom/BlobBinding.h"
#include "mozilla/ErrorResult.h"
#include "nsString.h"
#include "mozilla/Logging.h"

namespace mozilla::hwinference {

extern LazyLogModule gHWInferenceLog;
#define LOGE(...) MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Error, __VA_ARGS__)
#define LOGD(...) MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, __VA_ARGS__)
#define LOGV(...) MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Verbose, __VA_ARGS__)

StaticAutoPtr<nsTHashMap<nsCStringHashKey, RefPtr<HWInferenceParent>>>
    HWInferenceParent::sInstances;

/* static */
RefPtr<HWInferenceParent> HWInferenceParent::GetSingleton(
    const nsACString& aInstanceKey) {
  AssertIsOnMainThread();
  if (!sInstances) {
    sInstances = new nsTHashMap<nsCStringHashKey, RefPtr<HWInferenceParent>>();
    ClearOnShutdown(&sInstances);
  }
  return sInstances->GetOrInsertNew(aInstanceKey, aInstanceKey);
}

void HWInferenceParent::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("{}", __func__);
  if (sInstances) {
    sInstances->Remove(mInstanceKey);
  }
}

nsresult HWInferenceParent::BindToUtilityProcess(
    const RefPtr<ipc::UtilityProcessParent>& aUtilityParent) {
  LOGD("{}", __func__);
  Endpoint<hwinference::PHWInferenceParent> parentEnd;
  Endpoint<hwinference::PHWInferenceChild> childEnd;
  MOZ_ALWAYS_SUCCEEDS(PHWInference::CreateEndpoints(
      ipc::EndpointProcInfo::Current(), aUtilityParent->OtherEndpointProcInfo(),
      &parentEnd, &childEnd));

  LOGD("Sending StartHWInferenceService to utility process");
  if (!aUtilityParent->SendStartHWInferenceService(std::move(childEnd))) {
    LOGE("Failed to send StartHWInferenceService");
    MOZ_ASSERT(false, "StartHWInference service failure");
    return NS_ERROR_FAILURE;
  }

  LOGD("StartHWInferenceService sent successfully, binding parent endpoint");
  MOZ_ALWAYS_TRUE(parentEnd.Bind(this));
  return NS_OK;
}

}  // namespace mozilla::hwinference

#undef LOGD
#undef LOGV
#undef LOGE
