/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// On macOS these need MOZ_DISABLE_UTILITY_SANDBOX=1 to run locally: mach gtest
// points MOZ_XRE_DIR at <objdir>/dist/bin, so LlamaRuntimeLinker loads
// libmozinference from outside the bundle appPath that the HWInference sandbox
// policy grants file-map-executable on. CI runs with --xre-path inside the
// bundle and is unaffected.

#include "gtest/gtest.h"
#include "mozilla/Preferences.h"
#include "mozilla/SpinEventLoopUntil.h"
#include "mozilla/TimeStamp.h"
#include "mozilla/dom/TextGeneratorBinding.h"
#include "mozilla/gtest/WaitFor.h"
#include "mozilla/gtest/ipc/TestUtilityProcess.h"
#include "mozilla/hwinference/TextGenerationParent.h"
#include "mozilla/ipc/FileDescriptor.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "nsIProcessToolsService.h"
#include "nsServiceManagerUtils.h"
#include "nsThreadUtils.h"

using namespace mozilla;
using namespace mozilla::hwinference;

namespace {

constexpr auto kKind = ipc::SandboxingKind::HW_INFERENCE_BROWSER;

constexpr char kGracePref[] =
    "browser.ml.hwinference.browser_idle_shutdown_grace_ms";

// Generous for a debug CI machine; a healthy run needs a fraction of it.
constexpr uint32_t kStepTimeoutSeconds = 120;

template <typename Condition>
bool SpinUntil(const char* aName, uint32_t aTimeoutSeconds,
               Condition&& aCondition) {
  TimeStamp deadline =
      TimeStamp::Now() + TimeDuration::FromSeconds(aTimeoutSeconds);
  bool held = false;
  SpinEventLoopUntil<ProcessFailureBehavior::IgnoreAndContinue>(
      nsDependentCString(aName), [&] {
        held = aCondition();
        return held || TimeStamp::Now() > deadline;
      });
  return held;
}

Maybe<base::ProcessId> ProcessPid() {
  auto manager = ipc::UtilityProcessManager::GetSingleton();
  return manager ? manager->ProcessPid(kKind) : Nothing();
}

bool ProcessIsGone() { return ProcessPid().isNothing(); }

// Kills the process the way a crash would, and waits for the manager to
// notice.
void Crash(base::ProcessId aPid) {
  nsCOMPtr<nsIProcessToolsService> tools =
      do_GetService("@mozilla.org/processtools-service;1");
  ASSERT_TRUE(tools);
  ASSERT_TRUE(NS_SUCCEEDED(tools->Kill(aPid)));
  ASSERT_TRUE(SpinUntil("Crash noticed", kStepTimeoutSeconds, ProcessIsGone));
}

already_AddRefed<TextGenerationParent> CreateGenerator() {
  return TextGenerationParent::Create(
      ipc::FileDescriptor(),
      TextGenerationOptions(512, 2, 2, 2048, 512,
                            dom::TextGenerationKVCacheDtype::F16, false));
}

GenerateRequest EchoRequest() {
  GenerateRequest request;
  request.messages().AppendElement(
      ChatMessage(dom::TextGenerationRole::System, "hello "_ns));
  request.messages().AppendElement(
      ChatMessage(dom::TextGenerationRole::User, "world"_ns));
  request.maxTokens() = 8;
  request.bufferLength() = 4;
  request.stopOnEndOfGenerationTokens() = true;
  return request;
}

void ExpectProcessGone(const char* aWhy) {
  EXPECT_TRUE(SpinUntil("Process exit", kStepTimeoutSeconds, ProcessIsGone))
      << aWhy;
}

}  // namespace

// Round-trips the PTextGeneration surface through a real HW_INFERENCE_BROWSER
// process against the echo child: the bootstrap over PHWInference, generator
// construction, the Generate reply, deltas arriving before the reply, and how
// a generator's lifetime follows the process and the process the generators.
class TextGenerationTest : public mozilla::gtest::ipc::TestUtilityProcess {
 protected:
  // Zero grace: the tests assert the process exits with its last generator.
  void SetUp() override { Preferences::SetUint(kGracePref, 0); }
};

TEST_F(TextGenerationTest, EchoRoundTrip) {
  RefPtr<TextGenerationParent> generator = CreateGenerator();
  ASSERT_TRUE(generator);
  EXPECT_FALSE(generator->ProcessReused());

  auto ready = WaitFor(generator->WhenReady());
  ASSERT_TRUE(ready.isOk())
  << "the echo child reports a successful load";

  nsCString streamed;
  bool sawDeltaBeforeReply = false;
  generator->SetDeltaHandler([&](const nsCString& aText) {
    streamed.Append(aText);
    sawDeltaBeforeReply = true;
  });

  auto generateResult = WaitFor(generator->SendGenerate(EchoRequest()));
  ASSERT_TRUE(generateResult.isOk());
  const GenerateResponse response = generateResult.unwrap();
  ASSERT_EQ(response.type(), GenerateResponse::TGenerateResult);
  const GenerateResult& result = response.get_GenerateResult();

  EXPECT_EQ(result.content(), "hello world"_ns);
  EXPECT_EQ(result.reason(), dom::TextGenerationFinishReason::Eos);
  EXPECT_TRUE(sawDeltaBeforeReply);
  EXPECT_EQ(streamed, result.content());

  generator->Close();
  ExpectProcessGone("last generator gone but the process survived");
}

TEST_F(TextGenerationTest, SiblingCloseKeepsTheProcess) {
  RefPtr<TextGenerationParent> first = CreateGenerator();
  ASSERT_TRUE(first);
  ASSERT_TRUE(WaitFor(first->WhenReady()).isOk());
  Maybe<base::ProcessId> pid = ProcessPid();
  ASSERT_TRUE(pid.isSome());

  RefPtr<TextGenerationParent> second = CreateGenerator();
  ASSERT_TRUE(second);
  EXPECT_TRUE(second->ProcessReused());
  ASSERT_TRUE(WaitFor(second->WhenReady()).isOk());
  EXPECT_EQ(ProcessPid(), pid) << "generators should share the process";

  first->Close();
  first = nullptr;
  EXPECT_EQ(ProcessPid(), pid) << "a live generator must keep the process";

  auto generateResult = WaitFor(second->SendGenerate(EchoRequest()));
  ASSERT_TRUE(generateResult.isOk())
  << "sibling teardown must not end this generator";

  second->Close();
  ExpectProcessGone("last generator gone but the process survived");
}

TEST_F(TextGenerationTest, CloseBeforeReadyRetiresTheProcess) {
  RefPtr<TextGenerationParent> generator = CreateGenerator();
  ASSERT_TRUE(generator);

  generator->Close();
  auto ready = WaitFor(generator->WhenReady());
  ASSERT_TRUE(ready.isErr());
  EXPECT_EQ(ready.inspectErr().cause,
            TextGenerationParent::LoadFailure::Cause::ActorGone);
  ExpectProcessGone("a generator closed before ready pinned the process");
}

TEST_F(TextGenerationTest, LaunchFailureEndsTheGenerator) {
  auto manager = ipc::UtilityProcessManager::GetSingleton();
  ASSERT_TRUE(manager);
  // An unrelated kind keeps the manager singleton alive across the failed
  // launch, as it goes away with its last process.
  ASSERT_TRUE(
      WaitFor(manager->LaunchProcess(ipc::SandboxingKind::GENERIC_UTILITY))
          .isOk());

  RefPtr<TextGenerationParent> generator = CreateGenerator();
  ASSERT_TRUE(generator);
  // Nothing spins the event loop in between, so the launch is still pending
  // and this fails it.
  manager->CleanShutdown(kKind);

  auto ready = WaitFor(generator->WhenReady());
  ASSERT_TRUE(ready.isErr())
  << "a generator on a failed launch must report it";
  EXPECT_EQ(ready.inspectErr().cause,
            TextGenerationParent::LoadFailure::Cause::ActorGone);
  EXPECT_FALSE(generator->CanSend());
  generator = nullptr;

  RefPtr<TextGenerationParent> next = CreateGenerator();
  ASSERT_TRUE(next);
  EXPECT_FALSE(next->ProcessReused());
  ASSERT_TRUE(WaitFor(next->WhenReady()).isOk())
  << "a fresh launch must follow a failed one";
  next->Close();
  ExpectProcessGone("last generator gone but the process survived");

  manager->CleanShutdown(ipc::SandboxingKind::GENERIC_UTILITY);
  NS_ProcessPendingEvents(nullptr);
}

TEST_F(TextGenerationTest, ProcessDeathEndsTheGenerators) {
  RefPtr<TextGenerationParent> generator = CreateGenerator();
  ASSERT_TRUE(generator);
  ASSERT_TRUE(WaitFor(generator->WhenReady()).isOk());
  Maybe<base::ProcessId> pid = ProcessPid();
  ASSERT_TRUE(pid.isSome());

  Crash(*pid);
  EXPECT_TRUE(SpinUntil("Generator destroyed", kStepTimeoutSeconds, [&] {
    return !generator->CanSend();
  })) << "the generator must be destroyed with its process";

  RefPtr<TextGenerationParent> next = CreateGenerator();
  ASSERT_TRUE(next);
  EXPECT_FALSE(next->ProcessReused());
  ASSERT_TRUE(WaitFor(next->WhenReady()).isOk());
  Maybe<base::ProcessId> newPid = ProcessPid();
  ASSERT_TRUE(newPid.isSome());
  EXPECT_NE(*newPid, *pid);

  generator = nullptr;
  EXPECT_EQ(ProcessPid(), newPid)
      << "a generator of the dead process must not touch its replacement";
  next->Close();
  ExpectProcessGone("last generator gone but the process survived");
}
