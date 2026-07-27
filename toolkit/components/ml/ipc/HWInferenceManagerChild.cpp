/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "HWInferenceManagerChild.h"
#include "mozilla/Logging.h"
#include "mozilla/dom/ContentChild.h"
#include "mozilla/ipc/Endpoint.h"
#include "mozilla/StaticPtr.h"
#include "nsThreadUtils.h"

namespace mozilla::hwinference {

extern LazyLogModule gHWInferenceLog;
#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Error, fmt, ##__VA_ARGS__)

StaticRefPtr<HWInferenceManagerChild> HWInferenceManagerChild::sSingleton;
StaticMutex HWInferenceManagerChild::sSingletonMutex;

/* static */
void HWInferenceManagerChild::ReleaseConnectionKeepAlive() {
  auto release = []() {
    // Null in the parent process, which holds its keep-alive directly.
    if (dom::ContentChild* contentChild = dom::ContentChild::GetSingleton()) {
      (void)contentChild->SendReleaseHWInferenceConnection();
    }
  };

  if (NS_IsMainThread()) {
    release();
  } else {
    NS_DispatchToMainThread(NS_NewRunnableFunction(
        "HWInferenceManagerChild::ReleaseConnectionKeepAlive", release));
  }
}

/* static */
void HWInferenceManagerChild::OpenForProcess(
    Endpoint<PHWInferenceManagerChild>&& aEndpoint) {
  LOGD("{} - Opening connection to utility process", __func__);

  {
    StaticMutexAutoLock lock(sSingletonMutex);

    if (sSingleton && sSingleton->CanSend()) {
      LOGD("{} - Already have active singleton, reusing", __func__);
      // Fall through: this endpoint is dropped, so release its keep-alive.
    } else {
      sSingleton = nullptr;

      if (!aEndpoint.IsValid()) {
        LOGE("{} - ERROR: Invalid endpoint received", __func__);
      } else {
        LOGD("Creating new manager and binding endpoint");
        RefPtr<HWInferenceManagerChild> manager = new HWInferenceManagerChild();
        if (aEndpoint.Bind(manager)) {
          sSingleton = manager;
          LOGD("Successfully bound endpoint, connection ready", __func__);
          // Kept until ActorDestroy.
          return;
        }
        LOGE("{} - ERROR: Failed to bind endpoint", __func__);
      }
    }
  }

  ReleaseConnectionKeepAlive();
}

/* static */
RefPtr<HWInferenceManagerChild> HWInferenceManagerChild::GetSingleton() {
  StaticMutexAutoLock lock(sSingletonMutex);
  return sSingleton;
}

void HWInferenceManagerChild::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("{} reason={}, clearing singleton", __func__, static_cast<int>(aReason));

  {
    StaticMutexAutoLock lock(sSingletonMutex);
    if (sSingleton == this) {
      sSingleton = nullptr;
    }
  }

  ReleaseConnectionKeepAlive();
}

}  // namespace mozilla::hwinference

#undef LOGD
#undef LOGE
