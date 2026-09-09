/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_HWInferenceLifecycleHelpers_h
#define mozilla_hwinference_HWInferenceLifecycleHelpers_h

#include "mozilla/RefPtr.h"
#include "mozilla/ipc/UtilityProcessSandboxing.h"

namespace mozilla::ipc {
class UtilityProcessKeepAlive;
}  // namespace mozilla::ipc

// The keep-alive policy of HW_INFERENCE_BROWSER. A consumer acquires the
// process with UtilityProcessManager::AcquireBrowserHWInferenceProcess and
// holds the keep-alive for as long as it uses the process; it releases it
// through ReleaseBrowserProcessAfterGrace, which defers the drop by the idle
// grace so the next consumer skips the launch. Main thread only.
namespace mozilla::hwinference {

constexpr ipc::SandboxingKind kBrowserProcessKind =
    ipc::SandboxingKind::HW_INFERENCE_BROWSER;

// Whether the process is up, so acquiring it now skips the launch. False
// while a launch is still pending: the caller waits for it all the same.
bool IsBrowserProcessUp();

// Drops aKeepAlive after browser.ml.hwinference.browser_idle_shutdown_grace_ms;
// a zero grace drops it at once. Every consumer defers its own drop, so the
// process goes away after the grace following the last release.
void ReleaseBrowserProcessAfterGrace(
    RefPtr<ipc::UtilityProcessKeepAlive>&& aKeepAlive);

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_HWInferenceLifecycleHelpers_h
