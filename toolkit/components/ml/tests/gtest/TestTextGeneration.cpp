/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "gtest/gtest.h"
#include "mozilla/dom/TextGeneratorBinding.h"
#include "mozilla/gtest/WaitFor.h"
#include "mozilla/gtest/ipc/TestUtilityProcess.h"
#include "mozilla/hwinference/HWInferenceBrowserManagerParent.h"
#include "mozilla/hwinference/HWInferenceParent.h"
#include "mozilla/hwinference/TextGenerationParent.h"
#include "mozilla/ipc/FileDescriptor.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "nsThreadUtils.h"

using namespace mozilla;
using namespace mozilla::hwinference;

// Round-trips the PTextGeneration surface through a real browser-keyed
// HWInference process against the echo child: no model, no llama. What
// it pins is the plumbing the rest of the stack relies on: the manager
// bootstrap over PHWInference, generator construction with an fd and
// options, the Generate reply, and the ordering contract that a
// request's deltas arrive before its reply resolves.
class TextGenerationTest : public mozilla::gtest::ipc::TestUtilityProcess {};

TEST_F(TextGenerationTest, EchoRoundTrip) {
  auto manager = ipc::UtilityProcessManager::GetSingleton();
  ASSERT_TRUE(manager);

  auto createResult = WaitFor(HWInferenceBrowserManagerParent::GetOrCreate());
  ASSERT_TRUE(createResult.isOk());
  RefPtr<HWInferenceBrowserManagerParent> browserManager =
      createResult.unwrap();
  ASSERT_TRUE(browserManager->CanSend());

  // The echo child never touches the model; an invalid fd keeps this test
  // model-free until the llama commit.
  RefPtr<TextGenerationParent> generator = browserManager->CreateTextGeneration(
      ipc::FileDescriptor(),
      TextGenerationOptions(512, 0, 0, 2048, 512,
                            dom::TextGenerationKVCacheDtype::F16, false));
  ASSERT_TRUE(generator);

  auto ready = WaitFor(generator->WhenReady());
  ASSERT_TRUE(ready.isOk())
  << "construction should report Ready";

  nsCString streamed;
  bool sawDeltaBeforeReply = false;
  generator->SetDeltaHandler([&](const nsCString& aText) {
    streamed.Append(aText);
    sawDeltaBeforeReply = true;
  });

  GenerateRequest request;
  request.messages().AppendElement(
      ChatMessage(dom::TextGenerationRole::System, "hello "_ns));
  request.messages().AppendElement(
      ChatMessage(dom::TextGenerationRole::User, "world"_ns));
  request.maxTokens() = 8;
  request.bufferLength() = 4;
  request.stopOnEndOfGenerationTokens() = true;

  auto generateResult = WaitFor(generator->SendGenerate(request));
  ASSERT_TRUE(generateResult.isOk());
  const GenerateResponse response = generateResult.unwrap();
  ASSERT_EQ(response.type(), GenerateResponse::TGenerateResult);
  const GenerateResult& result = response.get_GenerateResult();

  EXPECT_EQ(result.content(), "hello world"_ns);
  EXPECT_EQ(result.reason(), dom::TextGenerationFinishReason::Eos);
  EXPECT_TRUE(sawDeltaBeforeReply);
  EXPECT_EQ(streamed, result.content());

  (void)PTextGenerationParent::Send__delete__(generator);

  manager->CleanShutdown(ipc::SandboxingKind::HW_INFERENCE,
                         HWINFERENCE_BROWSER_INSTANCE_KEY);
  NS_ProcessPendingEvents(nullptr);
}
