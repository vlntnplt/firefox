/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_TextGenerationParent_h
#define mozilla_hwinference_TextGenerationParent_h

#include <functional>

#include "mozilla/Maybe.h"
#include "mozilla/MozPromise.h"
#include "mozilla/hwinference/PTextGenerationParent.h"

namespace mozilla::hwinference {

// Main-process side of one generator.
class TextGenerationParent final : public PTextGenerationParent {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(TextGenerationParent, override);

  using DeltaHandler = std::function<void(const nsCString&)>;
  using ReadyPromise = MozPromise<double, nsCString, true>;

  // Resolves with the model load time in milliseconds.
  RefPtr<ReadyPromise> WhenReady();

  // The handler runs on the actor thread, before the Generate reply.
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
  Maybe<LoadResult> mReadyResult;
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_TextGenerationParent_h
