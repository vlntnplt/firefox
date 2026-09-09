/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "HWInferenceLifecycleHelpers.h"

#include "HWInferenceLog.h"
#include "mozilla/StaticPrefs_browser.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "nsThreadUtils.h"

namespace mozilla::hwinference {

#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)

bool IsBrowserProcessUp() {
  AssertIsOnMainThread();
  RefPtr<ipc::UtilityProcessManager> upm =
      ipc::UtilityProcessManager::GetIfExists();
  return upm && upm->GetProcessParent(kBrowserProcessKind);
}

void ReleaseBrowserProcessAfterGrace(
    RefPtr<ipc::UtilityProcessKeepAlive>&& aKeepAlive) {
  AssertIsOnMainThread();
  const uint32_t graceMs =
      StaticPrefs::browser_ml_hwinference_browser_idle_shutdown_grace_ms();
  if (!graceMs) {
    LOGD("{} - dropping the keep-alive at once", __func__);
    aKeepAlive = nullptr;
    return;
  }
  LOGD("{} - dropping the keep-alive in {}ms", __func__, graceMs);
  // A DelayedRunnable rather than a bare timer: nothing else keeps the timer
  // alive, and the main thread drops the runnable at shutdown.
  NS_DelayedDispatchToCurrentThread(
      NS_NewRunnableFunction("hwinference::ReleaseBrowserProcessAfterGrace",
                             [keepAlive = std::move(aKeepAlive)]() mutable {
                               LOGD(
                                   "ReleaseBrowserProcessAfterGrace - grace "
                                   "expired, dropping the keep-alive");
                               keepAlive = nullptr;
                             }),
      graceMs);
}

#undef LOGD

}  // namespace mozilla::hwinference
