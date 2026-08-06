/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_HWInferenceBrowserManagerChild_h
#define mozilla_hwinference_HWInferenceBrowserManagerChild_h

#include "mozilla/hwinference/PHWInferenceBrowserManagerChild.h"
#include "mozilla/ipc/Endpoint.h"
#include "mozilla/ipc/FileDescriptor.h"

namespace mozilla::hwinference {

// Utility-process side of the browser inference manager.
class HWInferenceBrowserManagerChild final
    : public PHWInferenceBrowserManagerChild {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(HWInferenceBrowserManagerChild,
                                        override);

  static bool CreateForBrowser(
      Endpoint<PHWInferenceBrowserManagerChild>&& aEndpoint);

#ifndef ANDROID
  already_AddRefed<PTextGenerationChild> AllocPTextGenerationChild(
      const ipc::FileDescriptor& aModel, const TextGenerationOptions& aOptions);

  mozilla::ipc::IPCResult RecvPTextGenerationConstructor(
      PTextGenerationChild* aActor, const ipc::FileDescriptor& aModel,
      const TextGenerationOptions& aOptions) override;
#endif

  void ActorDestroy(ActorDestroyReason aReason) override;

 private:
  friend PHWInferenceBrowserManagerChild;
  HWInferenceBrowserManagerChild() = default;
  ~HWInferenceBrowserManagerChild() = default;
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_HWInferenceBrowserManagerChild_h
