/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "mozilla/StaticPrefs_browser.h"
#include "mozilla/StaticPtr.h"
#include "nsTHashSet.h"
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
#include "nsFmtString.h"
#include "mozilla/Logging.h"

namespace mozilla::hwinference {

extern LazyLogModule gHWInferenceLog;
#define LOGE(...) MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Error, __VA_ARGS__)
#define LOGD(...) MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, __VA_ARGS__)
#define LOGV(...) MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Verbose, __VA_ARGS__)

StaticAutoPtr<nsTHashMap<nsCStringHashKey, RefPtr<HWInferenceParent>>>
    HWInferenceParent::sInstances;

static StaticAutoPtr<nsTHashSet<nsCString>> sMockInstalledModels;

static nsCString MockModelKey(const nsACString& aModel,
                              const nsACString& aRevision,
                              const nsACString& aFilename) {
  nsCString key(aModel);
  key.Append('/');
  key.Append(aRevision);
  key.Append('/');
  key.Append(aFilename);
  return key;
}

// Expands aId to a concrete ModelHub artifact via the resolver registered for
// aTask (contract id "@mozilla.org/ml/model-resolver;1?task=<task>"). aId is
// only ever compared for equality against the resolver's own known ids, so
// the requesting process can select among already-approved artifacts but
// never name one itself. Returns false (logging) if no resolver is
// registered for aTask, or aId is unknown to it.
static bool ResolveModelId(const nsACString& aTask, const nsACString& aId,
                           nsCString& aEngine, nsCString& aModel,
                           nsCString& aRevision, nsCString& aFilename) {
  nsFmtCString contractId("@mozilla.org/ml/model-resolver;1?task={}", aTask);
  nsCOMPtr<nsIMLModelResolver> resolver = do_GetService(contractId.get());
  if (!resolver) {
    LOGE("ResolveModelId - no resolver registered for task {}", aTask);
    return false;
  }

  nsresult rv = resolver->Resolve(aId, aEngine, aModel, aRevision, aFilename);
  if (NS_FAILED(rv)) {
    LOGE("ResolveModelId - id {} not recognized for task {}", aId, aTask);
    return false;
  }
  return true;
}

/* static */
RefPtr<HWInferenceParent> HWInferenceParent::GetSingleton(
    const nsACString& aInstanceKey) {
  AssertIsOnMainThread();
  if (!sInstances) {
    sInstances = new nsTHashMap<nsCStringHashKey, RefPtr<HWInferenceParent>>();
    ClearOnShutdown(&sInstances);
  }

  // Evict an instance whose process is already gone. Its PHWInference channel
  // is separate from PUtilityProcess, so it keeps reporting CanSend() until
  // the peer actually dies and the channel errors, one main-thread dispatch
  // later. Handing it out in that window would make StartUtility take its
  // CanSend() fast path and resolve success on a doomed actor rather than
  // relaunching. Checked here rather than at teardown so it covers an
  // unexpected process death too, not just CleanShutdown.
  RefPtr<HWInferenceParent> existing = sInstances->Get(aInstanceKey);
  if (existing && existing->CanSend()) {
    RefPtr<ipc::UtilityProcessManager> upm =
        ipc::UtilityProcessManager::GetIfExists();
    if (!upm ||
        !upm->Process(ipc::SandboxingKind::HW_INFERENCE, aInstanceKey)) {
      LOGD("{} - evicting stale instance for {}", __func__, aInstanceKey);
      sInstances->Remove(aInstanceKey);
      // Synchronously runs ActorDestroy, so CanSend() is false on return.
      existing->Close();
    }
  }

  return sInstances->GetOrInsertNew(aInstanceKey, aInstanceKey);
}

void HWInferenceParent::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("{}", __func__);
  // Only remove ourselves: a late ActorDestroy from a superseded instance
  // must not evict the replacement created for the same key.
  if (sInstances) {
    if (auto entry = sInstances->Lookup(mInstanceKey);
        entry && entry.Data() == this) {
      entry.Remove();
    }
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

mozilla::ipc::IPCResult HWInferenceParent::RecvIsModelAvailable(
    nsCString&& aTask, nsCString&& aId, IsModelAvailableResolver&& aResolver) {
  LOGD("{}: task={} id={}", __func__, aTask, aId);

  nsCString engine, model, revision, filename;
  if (!ResolveModelId(aTask, aId, engine, model, revision, filename)) {
    aResolver(false);
    return IPC_OK();
  }

  if (StaticPrefs::browser_ml_modelHub_testing()) {
    bool available =
        sMockInstalledModels &&
        sMockInstalledModels->Contains(MockModelKey(model, revision, filename));
    LOGD("{} - testing mock: available={}", __func__, available);
    aResolver(available);
    return IPC_OK();
  }

  nsCOMPtr<nsIMLModelHub> modelHubService =
      do_GetService("@mozilla.org/ml-modelhub;1");

  if (!modelHubService) {
    LOGE("{} - Failed to get ModelHub XPCOM service", __func__);
    aResolver(false);
    return IPC_OK();
  }

  RefPtr<dom::Promise> promise;
  nsresult rv = modelHubService->IsModelAvailable(
      engine, model, revision, filename, getter_AddRefs(promise));

  if (NS_FAILED(rv) || !promise) {
    LOGE("{}  ERROR: ModelHub call failed with nsresult={:x}", __func__,
         static_cast<uint32_t>(rv));
    aResolver(false);
    return IPC_OK();
  }

  (void)promise->AddCallbacksWithCycleCollectedArgs(
      [aResolver](JSContext* aCx, JS::Handle<JS::Value> aArg,
                  ErrorResult& aRv) { aResolver(JS::ToBoolean(aArg)); },
      [aResolver](JSContext* aCx, JS::Handle<JS::Value> aArg,
                  ErrorResult& aRv) { aResolver(false); });

  return IPC_OK();
}

}  // namespace mozilla::hwinference

#undef LOGD
#undef LOGV
#undef LOGE
