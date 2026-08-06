/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_HWInferenceBrowserManagerParent_h
#define mozilla_hwinference_HWInferenceBrowserManagerParent_h

#include "mozilla/MozPromise.h"
#include "mozilla/hwinference/PHWInferenceBrowserManagerParent.h"
#include "mozilla/ipc/FileDescriptor.h"
#include "nsCOMPtr.h"
#include "nsITimer.h"

namespace mozilla::hwinference {

#ifndef ANDROID
class TextGenerationParent;
#endif

// Main-process side of the browser inference manager, one per browser.
class HWInferenceBrowserManagerParent final
    : public PHWInferenceBrowserManagerParent {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(HWInferenceBrowserManagerParent,
                                        override);

  class Reservation;

  using CreatePromise = MozPromise<RefPtr<Reservation>, nsresult, false>;

  // Main thread only.
  static RefPtr<CreatePromise> GetOrCreate();

#ifndef ANDROID
  RefPtr<TextGenerationParent> CreateTextGeneration(
      const ipc::FileDescriptor& aModel, const TextGenerationOptions& aOptions);

  void OnTextGenerationDestroyed();
#endif

  void ActorDestroy(ActorDestroyReason aReason) override;

 private:
  friend PHWInferenceBrowserManagerParent;
  HWInferenceBrowserManagerParent() = default;
  ~HWInferenceBrowserManagerParent() = default;

  static RefPtr<CreatePromise> Create();

  void MaybeShutDownProcess();

  bool IsIdle();

  void RetireForIdle();

  void ArmIdleTimer(uint32_t aTimeoutMs);
  void CancelIdleTimer();

  // Idempotent; runs from both the shutdown decision and ActorDestroy.
  void ReleaseKeepAlive();

  // Main thread only.
  uint32_t mReservations = 0;
  nsCOMPtr<nsITimer> mIdleTimer;
  bool mHoldsKeepAlive = false;
};

// While a reservation is outstanding the manager is not retired for idle.
class HWInferenceBrowserManagerParent::Reservation final {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(Reservation);

  Reservation(HWInferenceBrowserManagerParent* aManager, bool aProcessReused);

  HWInferenceBrowserManagerParent* Manager() const { return mManager; }

  bool ProcessReused() const { return mProcessReused; }

#ifndef ANDROID
  RefPtr<TextGenerationParent> CreateTextGeneration(
      const ipc::FileDescriptor& aModel, const TextGenerationOptions& aOptions);
#endif

 private:
  ~Reservation();

  const RefPtr<HWInferenceBrowserManagerParent> mManager;
  const bool mProcessReused;
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_HWInferenceBrowserManagerParent_h
