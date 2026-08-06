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

// Utility-process side of one text generator, driving a LlamaBackend on
// its own TaskQueue.
class TextGenerationChild final : public PTextGenerationChild {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(TextGenerationChild, override);

  TextGenerationChild(const ipc::FileDescriptor& aModel,
                      const TextGenerationOptions& aOptions);

  // Call once the actor is bound; reports the load outcome with Ready.
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

  // Runs on mTaskQueue.
  LoadResult LoadOnQueue();

  // Skipped if the actor can no longer send.
  void RunOnActorThread(const char* aName, std::function<void()>&& aFn);

  ipc::FileDescriptor mModel;
  const TextGenerationOptions mOptions;

  const RefPtr<TaskQueue> mTaskQueue;
  const nsCOMPtr<nsISerialEventTarget> mActorThread;

  // TaskQueue only.
  RefPtr<llama::LlamaBackend> mBackend;
  nsCString mLoadError;

  // Actor thread only.
  CopyableTArray<ChatMessage> mHistory;
  RefPtr<Generation> mCurrentGeneration;

  Atomic<bool> mShutdown{false};
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_TextGenerationChild_h
