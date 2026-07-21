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

class TextGenerationParent;

// Main-process side of the browser inference manager: one manager over
// the one browser-keyed HWInference process. GetOrCreate() launches the
// process and binds the manager on first use, and every generator in the
// browser shares them, so the manager's managed set is the process-wide
// generator count its shutdown decision consults. Text generators are
// constructed with an already-open model fd; the process never requests
// files by name.
class HWInferenceBrowserManagerParent final
    : public PHWInferenceBrowserManagerParent {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(HWInferenceBrowserManagerParent,
                                        override);

  using CreatePromise =
      MozPromise<RefPtr<HWInferenceBrowserManagerParent>, nsresult, false>;

  // Main thread only. Resolves with the live shared manager, joining an
  // in-flight launch if one is already under way.
  static RefPtr<CreatePromise> GetOrCreate();

  // Constructs a PTextGeneration generator; the model is loaded as part
  // of construction and Ready reports the outcome. Returns nullptr if
  // the actor can no longer send.
  RefPtr<TextGenerationParent> CreateTextGeneration(
      const ipc::FileDescriptor& aModel, const TextGenerationOptions& aOptions);

  // Called from TextGenerationParent::ActorDestroy. The manager holds the
  // process keep-alive: when its last generator dies, the keep-alive is
  // released and UtilityProcessManager shuts the browser-keyed process
  // down unless another consumer still holds one.
  void OnTextGenerationDestroyed();

  void ActorDestroy(ActorDestroyReason aReason) override;

 private:
  friend PHWInferenceBrowserManagerParent;
  HWInferenceBrowserManagerParent() = default;
  ~HWInferenceBrowserManagerParent() = default;

  static RefPtr<CreatePromise> Create();

  // Releases the keep-alive once the managed set is empty and no
  // GetOrCreate caller is about to construct a generator; the decision
  // is dispatched so a dying actor has left the set first.
  void MaybeShutDownProcess();

  // Releases mHoldsKeepAlive exactly once; safe to call from both the
  // shutdown decision and ActorDestroy.
  void ReleaseKeepAlive();

  // Main thread only. Number of GetOrCreate callers handed this manager
  // whose CreateTextGeneration has not run yet; keeps the deferred
  // shutdown from firing between the two.
  uint32_t mPendingGenerators = 0;

  // Whether this actor still holds the UtilityProcessManager keep-alive
  // acquired for it in Create().
  bool mHoldsKeepAlive = false;
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_HWInferenceBrowserManagerParent_h
