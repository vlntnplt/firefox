/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "HWInferenceBrowserManagerChild.h"

#include "mozilla/Logging.h"

namespace mozilla::hwinference {

extern LazyLogModule gHWInferenceLog;
#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)

/* static */
bool HWInferenceBrowserManagerChild::CreateForBrowser(
    Endpoint<PHWInferenceBrowserManagerChild>&& aEndpoint) {
  RefPtr<HWInferenceBrowserManagerChild> actor =
      new HWInferenceBrowserManagerChild();
  if (!aEndpoint.Bind(actor)) {
    return false;
  }
  LOGD("[{} - {}] browser inference manager bound", fmt::ptr(actor.get()),
       __func__);
  return true;
}

void HWInferenceBrowserManagerChild::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
}

#undef LOGD

}  // namespace mozilla::hwinference
