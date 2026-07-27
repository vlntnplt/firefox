/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "TextGenerationParent.h"

#include "HWInferenceBrowserManagerParent.h"
#include "HWInferenceLog.h"

namespace mozilla::hwinference {

#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)

TextGenerationParent::TextGenerationParent() {
  // Materialize the promise now so a Ready that beats the first
  // WhenReady() caller still has something to resolve.
  (void)mReadyPromise.Ensure(__func__);
}

ipc::IPCResult TextGenerationParent::RecvReady(const LoadResult& aResult) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  if (aResult.type() == LoadResult::TLoadSuccess) {
    mReadyPromise.ResolveIfExists(aResult.get_LoadSuccess().loadMs(), __func__);
  } else {
    mReadyPromise.RejectIfExists(aResult.get_LoadError().message(), __func__);
  }
  return IPC_OK();
}

ipc::IPCResult TextGenerationParent::RecvDelta(const nsCString& aText) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  if (mDeltaHandler) {
    mDeltaHandler(aText);
  }
  return IPC_OK();
}

void TextGenerationParent::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  mReadyPromise.RejectIfExists("TextGenerator: actor destroyed"_ns, __func__);
  if (auto* manager =
          static_cast<HWInferenceBrowserManagerParent*>(Manager())) {
    manager->OnTextGenerationDestroyed();
  }
}

#undef LOGD

}  // namespace mozilla::hwinference
