/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_TextGenerationParent_h
#define mozilla_hwinference_TextGenerationParent_h

#include <functional>

#include "mozilla/MozPromise.h"
#include "mozilla/hwinference/PTextGenerationParent.h"

namespace mozilla::hwinference {

// Main-process side of one generator. Held by whoever created
// the generator through HWInferenceBrowserManagerParent::CreateTextGeneration;
// the generation API is the generated Send methods (SendGenerate returns the
// reply promise) plus a delta handler for the streamed text. WhenReady
// resolves with the load time once the construction-time model load
// succeeds, and rejects with the child's message if it fails or the actor
// dies first; owners wait on it before exposing the generator.
class TextGenerationParent final : public PTextGenerationParent {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(TextGenerationParent, override);

  using DeltaHandler = std::function<void(const nsCString&)>;
  using ReadyPromise = MozPromise<double, nsCString, true>;

  TextGenerationParent();

  RefPtr<ReadyPromise> WhenReady() { return mReadyPromise.Ensure(__func__); }

  // Called on the actor's thread for every Delta. Deltas belong to the
  // single pending Generate and always arrive before its reply resolves.
  void SetDeltaHandler(DeltaHandler aHandler) {
    mDeltaHandler = std::move(aHandler);
  }

  mozilla::ipc::IPCResult RecvReady(const LoadResult& aResult);
  mozilla::ipc::IPCResult RecvDelta(const nsCString& aText);

  void ActorDestroy(ActorDestroyReason aReason) override;

 private:
  friend PTextGenerationParent;
  ~TextGenerationParent() = default;

  DeltaHandler mDeltaHandler;
  MozPromiseHolder<ReadyPromise> mReadyPromise;
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_TextGenerationParent_h
