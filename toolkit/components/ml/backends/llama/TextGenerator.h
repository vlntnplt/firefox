/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_TextGenerator_h
#define mozilla_hwinference_TextGenerator_h

#include "js/TypeDecls.h"
#include "mozilla/dom/BindingDeclarations.h"
#include "mozilla/dom/TextGeneratorBinding.h"
#include "nsCycleCollectionParticipant.h"
#include "nsIGlobalObject.h"
#include "nsWrapperCache.h"

namespace mozilla::dom {
class Blob;
class Promise;
}  // namespace mozilla::dom

namespace mozilla::hwinference {

class HWInferenceBrowserManagerParent;
class TextGenerationParent;

// Chrome-JS-facing wrapper over one PTextGeneration actor in the
// browser-keyed HWInference process (see TextGenerator.webidl). Main
// thread only; the actor does its own thread hops.
class TextGenerator final : public nsISupports, public nsWrapperCache {
 public:
  NS_DECL_CYCLE_COLLECTING_ISUPPORTS
  NS_DECL_CYCLE_COLLECTION_WRAPPERCACHE_CLASS(TextGenerator)

  static already_AddRefed<dom::Promise> Create(
      const dom::GlobalObject& aGlobal, dom::Blob& aModel,
      const dom::TextGeneratorCreateOptions& aOptions);

  already_AddRefed<dom::Promise> Generate(
      const dom::TextGenerationRequest& aRequest,
      const dom::Optional<OwningNonNull<dom::TextGenerationDeltaCallback>>&
          aOnDelta,
      ErrorResult& aRv);

  void Clear();
  void Cancel();
  void Terminate();

  nsIGlobalObject* GetParentObject() const { return mGlobal; }
  JSObject* WrapObject(JSContext* aCx,
                       JS::Handle<JSObject*> aGivenProto) override;

 private:
  TextGenerator(nsIGlobalObject* aGlobal,
                RefPtr<HWInferenceBrowserManagerParent> aManager,
                RefPtr<TextGenerationParent> aActor);
  ~TextGenerator();

  void OnGenerateSettled();

  nsCOMPtr<nsIGlobalObject> mGlobal;
  RefPtr<HWInferenceBrowserManagerParent> mManager;
  RefPtr<TextGenerationParent> mActor;
  bool mGenerateInFlight = false;
};

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_TextGenerator_h
