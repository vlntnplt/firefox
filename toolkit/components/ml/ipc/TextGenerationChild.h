/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_TextGenerationChild_h
#define mozilla_hwinference_TextGenerationChild_h

#include <functional>

#include "mozilla/Atomics.h"
#include "mozilla/TaskQueue.h"
#include "mozilla/hwinference/PTextGenerationChild.h"
#include "mozilla/ipc/FileDescriptor.h"

namespace mozilla::llama {
class LlamaBackend;
}

namespace mozilla::hwinference {

// Utility-process side of one text generator. Owns a LlamaBackend and
// drives it on a per-generator TaskQueue. Construction is acquisition:
// Initialize() dispatches the model load onto the queue and reports the
// outcome with the one-shot Ready message; Generates queue behind the
// load and behind each other, so no ready check exists. Messages are
// handled on the actor thread; every Delta and each Generate reply are
// issued from the actor thread, reply last (see PTextGeneration.ipdl).
//
// The generator accumulates ChatMessages across Generates (the design's
// append semantics); Clear empties the history. The backend itself is
// driven statelessly with the full formatted history each time, which
// preserves the current engine's identical-runs-give-identical-text
// behavior.
class TextGenerationChild final : public PTextGenerationChild {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(TextGenerationChild, override);

  TextGenerationChild(const ipc::FileDescriptor& aModel,
                      const TextGenerationOptions& aOptions);

  // Completes construction once the actor is bound: acquires the model
  // on the TaskQueue and reports the outcome with Ready.
  void Initialize();

  mozilla::ipc::IPCResult RecvGenerate(const GenerateRequest& aRequest,
                                       GenerateResolver&& aResolve);
  mozilla::ipc::IPCResult RecvClear();
  mozilla::ipc::IPCResult RecvCancel();

  void ActorDestroy(ActorDestroyReason aReason) override;

 private:
  friend PTextGenerationChild;
  class Generation;

  ~TextGenerationChild();

  // Runs on mTaskQueue; fills mBackend or mLoadError.
  LoadResult LoadOnQueue();

  // Runs aFn on the actor thread iff the actor can still send.
  void RunOnActorThread(const char* aName, std::function<void()>&& aFn);

  ipc::FileDescriptor mModel;
  const TextGenerationOptions mOptions;

  const RefPtr<TaskQueue> mTaskQueue;
  const nsCOMPtr<nsISerialEventTarget> mActorThread;

  // TaskQueue only. mBackend is null iff the load failed (or has not run
  // yet); mLoadError then carries the message Generates answer with.
  RefPtr<llama::LlamaBackend> mBackend;
  nsCString mLoadError;

  // Actor thread only. mCurrentGeneration is the pending Generate that
  // Cancel targets; the parent surface guarantees at most one.
  CopyableTArray<ChatMessage> mHistory;
  RefPtr<Generation> mCurrentGeneration;

  // Set on the actor thread by ActorDestroy, read by the compute loop's
  // cancel callback so queued work self-cancels after the actor dies.
  Atomic<bool> mShutdown{false};
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_TextGenerationChild_h
