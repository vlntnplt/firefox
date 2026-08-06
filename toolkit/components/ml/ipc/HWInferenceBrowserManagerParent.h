/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_HWInferenceBrowserManagerParent_h
#define mozilla_hwinference_HWInferenceBrowserManagerParent_h

#include "mozilla/MozPromise.h"
#include "mozilla/hwinference/PHWInferenceBrowserManagerParent.h"
#include "mozilla/ipc/FileDescriptor.h"

namespace mozilla::hwinference {

// Main-process side of the browser inference manager. Create() launches
// (or reuses) the browser-keyed HWInference process, hands the child
// endpoint over PHWInference, and resolves with the bound manager.
class HWInferenceBrowserManagerParent final
    : public PHWInferenceBrowserManagerParent {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(HWInferenceBrowserManagerParent,
                                        override);

  using CreatePromise =
      MozPromise<RefPtr<HWInferenceBrowserManagerParent>, nsresult, true>;

  // Main thread only.
  static RefPtr<CreatePromise> Create();

  void ActorDestroy(ActorDestroyReason aReason) override;

 private:
  friend PHWInferenceBrowserManagerParent;
  HWInferenceBrowserManagerParent() = default;
  ~HWInferenceBrowserManagerParent() = default;
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_HWInferenceBrowserManagerParent_h
