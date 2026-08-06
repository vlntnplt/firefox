/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "ModelFileUtils.h"
#include "mozilla/StaticPrefs_browser.h"
#include "mozilla/StaticPtr.h"
#include "nsTHashSet.h"
#include "HWInferenceParent.h"
#include "HWInferenceManagerParent.h"
#include "mozilla/dom/Blob.h"
#include "mozilla/dom/BlobBinding.h"
#include "mozilla/ipc/FileDescriptor.h"
#include "mozilla/ipc/UtilityProcessParent.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "mozilla/dom/Promise.h"
#include "mozilla/dom/Promise-inl.h"
#include "nsIFileStreams.h"
#include "nsIInputStream.h"
#include "mozilla/ErrorResult.h"
#include "nsString.h"
#include "nsFmtString.h"
#include "mozilla/Logging.h"
#include "mozilla/Services.h"
#include "nsIMLModelDownloadGate.h"
#include "nsIMLModelHub.h"
#include "nsIMLModelResolver.h"
#include "nsIObserverService.h"
#include "nsServiceManagerUtils.h"
#include "prio.h"
#include "private/pprio.h"

#ifdef XP_WIN
#  include <windows.h>
#endif

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

class ModelDownloadProgressCallback final
    : public nsIMLModelDownloadProgressCallback {
 public:
  NS_DECL_ISUPPORTS

  ModelDownloadProgressCallback(const nsACString& aModel,
                                const nsAString& aProgressToken)
      : mModel(aModel), mProgressToken(aProgressToken) {}

  NS_IMETHOD OnProgress(int32_t aProgress, int64_t aCurrentLoaded,
                        int64_t aTotalLoaded, int64_t aTotal) override {
    LOGV("{} - model={} progress={}% current={} total loaded={} total={}",
         __func__, mModel.get(), aProgress, aCurrentLoaded, aTotalLoaded,
         aTotal);
    Notify(aProgress, aCurrentLoaded, aTotalLoaded, aTotal, false, true);
    return NS_OK;
  }

  void Notify(int32_t aProgress, int64_t aCurrentLoaded, int64_t aTotalLoaded,
              int64_t aTotal, bool aDone, bool aOk) {
    if (mProgressToken.IsEmpty()) {
      return;
    }

    nsCOMPtr<nsIObserverService> obs = services::GetObserverService();
    if (!obs) {
      return;
    }

    nsString data = nsFmtString(
        u"{{\"token\":\"{}\",\"progress\":{},\"currentLoaded\":{},"
        u"\"totalLoaded\":{},\"total\":{},\"done\":{},\"ok\":{}}}",
        mProgressToken, aProgress, aCurrentLoaded, aTotalLoaded, aTotal, aDone,
        aOk);
    obs->NotifyObservers(nullptr, "ml-model-download-progress", data.get());
  }

 private:
  ~ModelDownloadProgressCallback() = default;
  nsCString mModel;
  nsString mProgressToken;
};

NS_IMPL_ISUPPORTS(ModelDownloadProgressCallback,
                  nsIMLModelDownloadProgressCallback)

class ModelDownloadCompletionCallback final
    : public nsIMLModelDownloadCompletionCallback {
 public:
  NS_DECL_ISUPPORTS

  ModelDownloadCompletionCallback(
      HWInferenceParent::InstallModelResolver&& aResolver,
      ModelDownloadProgressCallback* aProgressCallback)
      : mResolver(std::move(aResolver)), mProgressCallback(aProgressCallback) {}

  NS_IMETHOD OnSuccess(const nsAString& aModel,
                       const nsAString& aRevision) override {
    LOGD("{} - model={} revision={}", __func__,
         NS_ConvertUTF16toUTF8(aModel).get(),
         NS_ConvertUTF16toUTF8(aRevision).get());
    mProgressCallback->Notify(100, 0, 0, 0, true, true);
    mResolver(true);
    return NS_OK;
  }

  NS_IMETHOD OnError(const nsAString& aError) override {
    LOGE("{} - Error when downloading {}", __func__,
         NS_ConvertUTF16toUTF8(aError).get());
    mProgressCallback->Notify(0, 0, 0, 0, true, false);
    mResolver(false);
    return NS_OK;
  }

 private:
  ~ModelDownloadCompletionCallback() = default;
  HWInferenceParent::InstallModelResolver mResolver;
  RefPtr<ModelDownloadProgressCallback> mProgressCallback;
};

NS_IMPL_ISUPPORTS(ModelDownloadCompletionCallback,
                  nsIMLModelDownloadCompletionCallback)

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
#ifndef ANDROID
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
#endif  // ANDROID

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

mozilla::ipc::IPCResult HWInferenceParent::RecvIsModelInstalled(
    nsCString&& aTask, nsCString&& aId, IsModelInstalledResolver&& aResolver) {
  LOGD("{}: task={} id={}", __func__, aTask, aId);

  nsCString engine, model, revision, filename;
  if (!ResolveModelId(aTask, aId, engine, model, revision, filename)) {
    aResolver(false);
    return IPC_OK();
  }

  if (StaticPrefs::browser_ml_modelHub_testing()) {
    bool installed =
        sMockInstalledModels &&
        sMockInstalledModels->Contains(MockModelKey(model, revision, filename));
    LOGD("{} - testing mock: installed={}", __func__, installed);
    aResolver(installed);
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
  nsresult rv = modelHubService->IsModelInstalled(
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

// Performs the actual model download (or testing-mock install) and resolves
// aResolver with the outcome. Honors the testing mock. Only reached after the
// gate (if any) has authorized the download.
static void PerformModelInstall(
    const nsCString& aTask, const nsCString& aModel, const nsCString& aRevision,
    const nsCString& aFilename, const nsString& aProgressToken,
    HWInferenceParent::InstallModelResolver&& aResolver) {
  if (StaticPrefs::browser_ml_modelHub_testing()) {
    if (!sMockInstalledModels) {
      sMockInstalledModels = new nsTHashSet<nsCString>();
      ClearOnShutdown(&sMockInstalledModels);
    }
    sMockInstalledModels->Insert(MockModelKey(aModel, aRevision, aFilename));
    LOGD("PerformModelInstall - testing mock: installed {}",
         MockModelKey(aModel, aRevision, aFilename).get());
    aResolver(true);
    return;
  }

  nsCOMPtr<nsIMLModelHub> modelHubService =
      do_GetService("@mozilla.org/ml-modelhub;1");

  if (!modelHubService) {
    LOGE("PerformModelInstall - Failed to get ModelHub XPCOM service");
    aResolver(false);
    return;
  }

  nsTArray<nsCString> files;
  files.AppendElement(aFilename);

  RefPtr<ModelDownloadProgressCallback> progressCallback =
      new ModelDownloadProgressCallback(aModel, aProgressToken);
  RefPtr<ModelDownloadCompletionCallback> completionCallback =
      new ModelDownloadCompletionCallback(std::move(aResolver),
                                          progressCallback);

  nsString downloadSessionId;
  nsresult rv = modelHubService->DownloadModel(
      aTask, aModel, aRevision, files, progressCallback, completionCallback,
      downloadSessionId);

  if (NS_FAILED(rv) || downloadSessionId.IsEmpty()) {
    LOGE(
        "PerformModelInstall - ERROR: ModelHub DownloadModel call failed "
        "with nsresult={:x}",
        static_cast<uint32_t>(rv));
    completionCallback->OnError(u"Failed to start download"_ns);
    return;
  }
  LOGD(
      "PerformModelInstall - download started successfully with session ID: {}",
      NS_ConvertUTF16toUTF8(downloadSessionId).get());
}

// Bridges nsIMLModelDownloadGate's decision back to the pending InstallModel:
// on allow it performs the download, on deny it resolves the install false.
// aProgressToken is an opaque id supplied by the caller (see PHWInference's
// InstallModel) and forwarded to PerformModelInstall, which uses it to tag
// the "ml-model-download-progress" notifications so the caller can match
// them to this request.
class ModelDownloadGateCallback final : public nsIMLModelDownloadGateCallback {
 public:
  NS_DECL_ISUPPORTS

  ModelDownloadGateCallback(const nsACString& aTask, const nsACString& aModel,
                            const nsACString& aRevision,
                            const nsACString& aFilename,
                            const nsAString& aProgressToken,
                            HWInferenceParent::InstallModelResolver&& aResolver)
      : mTask(aTask),
        mModel(aModel),
        mRevision(aRevision),
        mFilename(aFilename),
        mProgressToken(aProgressToken),
        mResolver(std::move(aResolver)) {}

  NS_IMETHOD Resolve(bool aAllow) override {
    if (!mResolver) {
      return NS_OK;
    }
    auto resolver = std::move(mResolver);
    mResolver = nullptr;
    if (!aAllow) {
      LOGD("ModelDownloadGateCallback - gate denied download of {}",
           mModel.get());
      resolver(false);
      return NS_OK;
    }
    PerformModelInstall(mTask, mModel, mRevision, mFilename, mProgressToken,
                        std::move(resolver));
    return NS_OK;
  }

 private:
  ~ModelDownloadGateCallback() {
    if (mResolver) {
      mResolver(false);
    }
  }

  nsCString mTask;
  nsCString mModel;
  nsCString mRevision;
  nsCString mFilename;
  nsString mProgressToken;
  HWInferenceParent::InstallModelResolver mResolver;
};

NS_IMPL_ISUPPORTS(ModelDownloadGateCallback, nsIMLModelDownloadGateCallback)

ipc::IPCResult HWInferenceParent::RecvInstallModel(
    nsCString&& aTask, nsCString&& aId, uint64_t aInnerWindowId,
    const dom::ContentParentId& aContentId, nsString&& aProgressToken,
    InstallModelResolver&& aResolver) {
  LOGD("{} task={} id={}", __func__, aTask, aId);

  nsCString engine, model, revision, filename;
  if (!ResolveModelId(aTask, aId, engine, model, revision, filename)) {
    aResolver(false);
    return IPC_OK();
  }

  // Installed short-circuit for the testing mock: an already-installed model
  // has nothing to download, so there is nothing to gate/consent to. The real
  // (non-testing) already-installed skip is handled by the gate, which knows
  // the engine id needed to query ModelHub. The mock lives only here, so the
  // install and availability paths agree on what has been "downloaded".
  if (StaticPrefs::browser_ml_modelHub_testing() && sMockInstalledModels &&
      sMockInstalledModels->Contains(MockModelKey(model, revision, filename))) {
    LOGD("{} - testing mock: already installed, skipping gate", __func__);
    aResolver(true);
    return IPC_OK();
  }

  // A task may register an authorization gate. If one exists the download only
  // happens when the gate allows it; otherwise the install is ungated and
  // downloads directly.
  nsFmtCString contractId("@mozilla.org/ml/model-download-gate;1?task={}",
                          aTask);
  nsCOMPtr<nsIMLModelDownloadGate> gate = do_GetService(contractId.get());
  if (!gate) {
    LOGD("{} - no gate for task {}, downloading ungated", __func__,
         aTask.get());
    PerformModelInstall(aTask, model, revision, filename, aProgressToken,
                        std::move(aResolver));
    return IPC_OK();
  }

  RefPtr<ModelDownloadGateCallback> callback = new ModelDownloadGateCallback(
      aTask, model, revision, filename, aProgressToken, std::move(aResolver));
  // The gate still runs for a privileged-chrome request; aInnerWindowId/
  // aContentId are passed through opaquely for it to validate. Both are 0
  // when the request originates from privileged chrome rather than content,
  // since there is no window/content id to validate for a trusted origin.
  gate->ShouldAllowDownload(aTask, model, revision, filename, aInnerWindowId,
                            uint64_t(aContentId), aProgressToken, callback);
  return IPC_OK();
}

ipc::IPCResult HWInferenceParent::RecvGetModelFile(
    nsCString&& aTask, nsCString&& aId, GetModelFileResolver&& aResolver) {
  LOGD("{} task={} id={}", __func__, aTask.get(), aId.get());

  nsCString engine, model, revision, filename;
  if (!ResolveModelId(aTask, aId, engine, model, revision, filename)) {
    GetModelError error;
    error.errorCode() = NS_ERROR_NOT_AVAILABLE;
    aResolver(GetModelFileResult(error));
    return IPC_OK();
  }

  nsCOMPtr<nsIMLModelHub> modelHubService =
      do_GetService("@mozilla.org/ml-modelhub;1");

  if (!modelHubService) {
    LOGE("{} - ERROR: Failed to get ModelHub XPCOM service", __func__);
    GetModelError error;
    error.errorCode() = NS_ERROR_FAILURE;
    aResolver(GetModelFileResult(error));
    return IPC_OK();
  }

  RefPtr<dom::Promise> promise;
  nsresult rv = modelHubService->GetModelBlob(
      engine, aTask, model, revision, filename, getter_AddRefs(promise));

  if (NS_FAILED(rv)) {
    LOGE("{} - ERROR: GetModelBlob call failed with rv={:x}", __func__,
         static_cast<uint32_t>(rv));
    GetModelError error;
    error.errorCode() = rv;
    aResolver(GetModelFileResult(error));
    return IPC_OK();
  }

  promise->AddCallbacksWithCycleCollectedArgs(
      [aResolver](JSContext* aCx, JS::Handle<JS::Value> aValue,
                  ErrorResult& aRv) {
        GetModelFileSuccess success;
        nsresult rv = BlobJSObjectToFileDescriptor(aCx, aValue, &success.fd());
        if (NS_FAILED(rv)) {
          aResolver(GetModelFileResult(GetModelError(rv)));
          return;
        }
        MOZ_ASSERT(success.fd().IsValid());
        aResolver(GetModelFileResult(std::move(success)));
      },
      [aResolver](JSContext* aCx, JS::Handle<JS::Value> aValue,
                  ErrorResult& aRv) {
        LOGE("RecvGetModelFile - ERROR: promise rejected in RecvGetModelFile");
        GetModelError error;
        error.errorCode() = NS_ERROR_FAILURE;
        aResolver(GetModelFileResult(error));
      });

  return IPC_OK();
}

}  // namespace mozilla::hwinference

#undef LOGD
#undef LOGV
#undef LOGE
