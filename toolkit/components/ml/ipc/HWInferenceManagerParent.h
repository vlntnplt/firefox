/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TOOLKIT_COMPONENTS_ML_IPC_HWINFERENCEMANAGERPARENT_H_
#define TOOLKIT_COMPONENTS_ML_IPC_HWINFERENCEMANAGERPARENT_H_

#include "mozilla/hwinference/PHWInferenceManagerParent.h"
#include "mozilla/dom/ipc/IdType.h"
#include "nsRefPtrHashtable.h"

namespace mozilla::hwinference {

class HWInferenceManagerParent final : public PHWInferenceManagerParent {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(HWInferenceManagerParent, override);

  static bool CreateForContent(Endpoint<PHWInferenceManagerParent>&& aEndpoint,
                               dom::ContentParentId aContentId);

  void ActorDestroy(ActorDestroyReason aReason) override;

  // Trusted id of the content process this manager was created for, assigned
  // by the parent (never content-supplied). Used to attribute install
  // requests to the requesting process so ownership of the requesting window
  // can be verified. For a manager created on behalf of the parent process
  // itself (a browser feature, not a content document), this is 0.
  dom::ContentParentId ContentId() const { return mContentId; }

 private:
  explicit HWInferenceManagerParent(dom::ContentParentId aContentId);
  ~HWInferenceManagerParent() = default;

  const dom::ContentParentId mContentId;
};

}  // namespace mozilla::hwinference

#endif  // TOOLKIT_COMPONENTS_ML_IPC_HWINFERENCEMANAGERPARENT_H_
