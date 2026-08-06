/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "gtest/gtest.h"
#include "mozilla/Preferences.h"
#include "mozilla/Result.h"
#include "mozilla/TimeStamp.h"
#include "mozilla/dom/TextGeneratorBinding.h"
#include "mozilla/gtest/WaitFor.h"
#include "mozilla/gtest/ipc/TestUtilityProcess.h"
#include "mozilla/hwinference/HWInferenceBrowserManagerParent.h"
#include "mozilla/hwinference/HWInferenceParent.h"
#include "mozilla/hwinference/TextGenerationParent.h"
#include "mozilla/ipc/FileDescriptor.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "nsThreadUtils.h"
#include "prio.h"
#include "private/pprio.h"

using namespace mozilla;
using namespace mozilla::hwinference;

namespace {

constexpr char kModelFile[] = "TinyStories-656K.Q8_0.gguf";

// Generous for a debug CI machine; a healthy run needs a fraction of it.
constexpr uint32_t kStepTimeoutSeconds = 120;

ipc::FileDescriptor OpenModelFd(const char* aPath) {
  PRFileDesc* fd = PR_Open(aPath, PR_RDONLY, 0);
  if (!fd) {
    return ipc::FileDescriptor();
  }
  ipc::FileDescriptor result(
      ipc::FileDescriptor::PlatformHandleType(PR_FileDesc2NativeHandle(fd)));
  PR_Close(fd);
  return result;
}

template <typename R, typename E, bool Excl>
Maybe<Result<R, E>> WaitForOrTimeout(RefPtr<MozPromise<R, E, Excl>> aPromise,
                                     uint32_t aTimeoutSeconds) {
  auto result = MakeRefPtr<media::Refcountable<Maybe<Result<R, E>>>>();
  aPromise->Then(
      GetCurrentSerialEventTarget(), __func__,
      [result](R aResolve) {
        *result = Some(Result<R, E>(std::move(aResolve)));
      },
      [result](E aReject) {
        *result = Some(Result<R, E>(Err(std::move(aReject))));
      });
  TimeStamp deadline =
      TimeStamp::Now() + TimeDuration::FromSeconds(aTimeoutSeconds);
  SpinEventLoopUntil<ProcessFailureBehavior::IgnoreAndContinue>(
      "TestTextGeneration WaitForOrTimeout"_ns,
      [&] { return result->isSome() || TimeStamp::Now() > deadline; });
  return std::move(*result);
}

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

// The shutdown decision is one main-thread dispatch from the generator's
// ActorDestroy, so a runnable queued after it runs after it.
void FlushShutdownDecision() {
  bool ran = false;
  MOZ_ALWAYS_SUCCEEDS(NS_DispatchToMainThread(
      NS_NewRunnableFunction("FlushShutdownDecision", [&] { ran = true; })));
  SpinEventLoopUntil<ProcessFailureBehavior::IgnoreAndContinue>(
      "FlushShutdownDecision"_ns, [&] { return ran; });
}

bool ProcessIsGone() {
  auto manager = ipc::UtilityProcessManager::GetSingleton();
  return !manager ||
         !manager->GetProcessParent(ipc::SandboxingKind::HW_INFERENCE,
                                    HWINFERENCE_BROWSER_INSTANCE_KEY);
}

// top-k:1 is greedy, but llama.cpp needs the chain to end with a dist sampler.
CopyableTArray<Sampler> GreedySamplers() {
  CopyableTArray<Sampler> samplers;
  // WebIDL defaults for the fields each sampler ignores.
  Sampler topK(dom::TextGenerationSamplerType::Top_k, /* topK */ 1,
               /* topP */ 0.95f, /* temp */ 0.80f,
               /* seed */ 0xFFFFFFFF, CopyableTArray<LogitBias>());
  Sampler dist = topK;
  dist.type() = dom::TextGenerationSamplerType::Dist;
  samplers.AppendElement(topK);
  samplers.AppendElement(dist);
  return samplers;
}

GenerateRequest StoryRequest(uint32_t aMaxTokens = 16,
                             uint32_t aBufferLength = 4) {
  GenerateRequest request;
  request.messages().AppendElement(ChatMessage(
      dom::TextGenerationRole::System, "You are a friendly storyteller."_ns));
  request.messages().AppendElement(
      ChatMessage(dom::TextGenerationRole::User,
                  "Once upon a time there was a small mouse who"_ns));
  request.maxTokens() = aMaxTokens;
  request.bufferLength() = aBufferLength;
  request.samplers() = GreedySamplers();
  request.stopOnEndOfGenerationTokens() = true;
  return request;
}

struct Session {
  RefPtr<HWInferenceBrowserManagerParent> mManager;
  RefPtr<TextGenerationParent> mGenerator;
  nsCString mStreamed;
};

Session CreateGenerator(const ipc::FileDescriptor& aModel) {
  Session session;
  auto createResult = WaitForOrTimeout(
      HWInferenceBrowserManagerParent::GetOrCreate(), kStepTimeoutSeconds);
  if (createResult.isNothing() || createResult->isErr()) {
    printf_stderr("CreateGenerator: GetOrCreate() timed out or rejected\n");
    return session;
  }
  RefPtr<HWInferenceBrowserManagerParent::Reservation> reservation =
      createResult->unwrap();
  session.mManager = reservation->Manager();
  session.mGenerator = reservation->CreateTextGeneration(
      aModel,
      TextGenerationOptions(512, 2, 2, 2048, 512,
                            dom::TextGenerationKVCacheDtype::F16, false));
  if (!session.mGenerator) {
    printf_stderr("CreateGenerator: CreateTextGeneration returned null\n");
  }
  return session;
}

// Some() carries the load failure message; Nothing() means the model loaded.
Maybe<nsCString> WaitForReady(Session& aSession) {
  auto ready =
      WaitForOrTimeout(aSession.mGenerator->WhenReady(), kStepTimeoutSeconds);
  EXPECT_TRUE(ready.isSome()) << "Ready timed out";
  if (ready.isNothing()) {
    return Some("Ready timed out"_ns);
  }
  if (ready->isOk()) {
    return Nothing();
  }
  return Some(ready->unwrapErr());
}

auto StartGenerate(Session& aSession, const GenerateRequest& aRequest) {
  aSession.mGenerator->SetDeltaHandler(
      [session = &aSession](const nsCString& aText) {
        session->mStreamed.Append(aText);
      });
  return aSession.mGenerator->SendGenerate(aRequest);
}

// aErrorExpected only suppresses the diagnostic printf.
GenerateResponse Generate(Session& aSession, const GenerateRequest& aRequest,
                          bool aErrorExpected = false) {
  auto result =
      WaitForOrTimeout(StartGenerate(aSession, aRequest), kStepTimeoutSeconds);
  EXPECT_TRUE(result.isSome()) << "Generate timed out";
  if (result.isNothing()) {
    return GenerateResponse();
  }
  EXPECT_TRUE(result->isOk());
  GenerateResponse response =
      result->isOk() ? result->unwrap() : GenerateResponse();
  if (!aErrorExpected && response.type() == GenerateResponse::TGenerateError) {
    printf_stderr("GenerateError: %s\n",
                  response.get_GenerateError().message().get());
  }
  return response;
}

void Teardown(Session& aSession) {
  if (aSession.mGenerator) {
    (void)PTextGenerationParent::Send__delete__(aSession.mGenerator);
  }
  EXPECT_TRUE(
      SpinUntil("Teardown process exit", kStepTimeoutSeconds, ProcessIsGone))
      << "last generator gone but the HWInference process survived";
}

}  // namespace

class TextGenerationTest : public mozilla::gtest::ipc::TestUtilityProcess {
 protected:
  // Zero grace period: Teardown() asserts the process exits immediately.
  void SetUp() override {
    Preferences::SetUint("browser.ml.hwInference.processTimeout", 0);
  }
};

TEST_F(TextGenerationTest, GenerateFromModel) {
  ipc::FileDescriptor model = OpenModelFd(kModelFile);
  ASSERT_TRUE(model.IsValid())
  << "missing " << kModelFile;

  Session session = CreateGenerator(model);
  ASSERT_TRUE(session.mGenerator);
  EXPECT_TRUE(WaitForReady(session).isNothing())
      << "construction should load the model";

  GenerateResponse response = Generate(session, StoryRequest());
  ASSERT_EQ(response.type(), GenerateResponse::TGenerateResult);
  const GenerateResult& result = response.get_GenerateResult();

  EXPECT_FALSE(result.content().IsEmpty());
  EXPECT_EQ(session.mStreamed, result.content());
  EXPECT_GT(result.usage().promptTokens(), 0u);
  EXPECT_GT(result.usage().generatedTokens(), 0u);
  EXPECT_GT(result.usage().promptCharacters(), result.usage().promptTokens());
  EXPECT_TRUE(result.reason() == dom::TextGenerationFinishReason::Length ||
              result.reason() == dom::TextGenerationFinishReason::Eos);

  EXPECT_GE(result.resources().after().cpuTimeMs(),
            result.resources().before().cpuTimeMs());
  EXPECT_GT(result.resources().after().memoryBytes(), 0u);

  Teardown(session);
}

TEST_F(TextGenerationTest, ClearResetsHistory) {
  ipc::FileDescriptor model = OpenModelFd(kModelFile);
  ASSERT_TRUE(model.IsValid())
  << "missing " << kModelFile;

  Session session = CreateGenerator(model);
  ASSERT_TRUE(session.mGenerator);
  EXPECT_TRUE(WaitForReady(session).isNothing())
      << "construction should load the model";

  GenerateResponse first = Generate(session, StoryRequest());
  ASSERT_EQ(first.type(), GenerateResponse::TGenerateResult);

  session.mStreamed.Truncate();
  GenerateResponse appended = Generate(session, StoryRequest());
  ASSERT_EQ(appended.type(), GenerateResponse::TGenerateResult);
  EXPECT_GT(appended.get_GenerateResult().usage().promptTokens(),
            first.get_GenerateResult().usage().promptTokens())
      << "second Generate should prefill the accumulated history";

  // Fire-and-forget; in-order delivery lands it before the next Generate.
  EXPECT_TRUE(session.mGenerator->SendClear());
  session.mStreamed.Truncate();
  GenerateResponse cleared = Generate(session, StoryRequest());
  ASSERT_EQ(cleared.type(), GenerateResponse::TGenerateResult);
#if defined(XP_MACOSX) && defined(__x86_64__)
  // Bug 2047025: greedy decode is not byte-stable on mac x64.
  EXPECT_FALSE(cleared.get_GenerateResult().content().IsEmpty());
#else
  EXPECT_EQ(cleared.get_GenerateResult().content(),
            first.get_GenerateResult().content())
      << "greedy generation after Clear should reproduce the first run";
#endif

  Teardown(session);
}

TEST_F(TextGenerationTest, SharedProcessSurvivesSiblingTermination) {
  // One fd per generator: sending a FileDescriptor consumes its handle.
  ipc::FileDescriptor model = OpenModelFd(kModelFile);
  ASSERT_TRUE(model.IsValid())
  << "missing " << kModelFile;

  Session first = CreateGenerator(model);
  ASSERT_TRUE(first.mGenerator);
  Session second = CreateGenerator(OpenModelFd(kModelFile));
  ASSERT_TRUE(second.mGenerator);
  EXPECT_EQ(first.mManager.get(), second.mManager.get())
      << "independent creates should share the browser manager";
  EXPECT_TRUE(WaitForReady(first).isNothing());
  EXPECT_TRUE(WaitForReady(second).isNothing());

  (void)PTextGenerationParent::Send__delete__(first.mGenerator);
  first.mGenerator = nullptr;

  GenerateResponse response = Generate(second, StoryRequest());
  ASSERT_EQ(response.type(), GenerateResponse::TGenerateResult)
      << "sibling termination must not abort this generator";
  EXPECT_FALSE(response.get_GenerateResult().content().IsEmpty());

  Teardown(second);
}

TEST_F(TextGenerationTest, IdleTimeoutKeepsProcessForReuse) {
  Preferences::SetUint("browser.ml.hwInference.processTimeout", 120000);

  ipc::FileDescriptor model = OpenModelFd(kModelFile);
  ASSERT_TRUE(model.IsValid())
  << "missing " << kModelFile;

  Session first = CreateGenerator(model);
  ASSERT_TRUE(first.mGenerator);
  ASSERT_TRUE(WaitForReady(first).isNothing());
  RefPtr<HWInferenceBrowserManagerParent> idleManager = first.mManager;

  (void)PTextGenerationParent::Send__delete__(first.mGenerator);
  first.mGenerator = nullptr;

  FlushShutdownDecision();
  EXPECT_FALSE(ProcessIsGone())
      << "the idle timeout should have kept the process alive";

  Session second = CreateGenerator(OpenModelFd(kModelFile));
  ASSERT_TRUE(second.mGenerator);
  EXPECT_EQ(second.mManager.get(), idleManager.get())
      << "a generator inside the idle window should reuse the live manager";
  EXPECT_TRUE(WaitForReady(second).isNothing());

  // Teardown() asserts the process exits, so restore eager teardown first.
  Preferences::SetUint("browser.ml.hwInference.processTimeout", 0);
  Teardown(second);
}

TEST_F(TextGenerationTest, CancelResolvesWithPartialResult) {
  ipc::FileDescriptor model = OpenModelFd(kModelFile);
  ASSERT_TRUE(model.IsValid())
  << "missing " << kModelFile;

  Session session = CreateGenerator(model);
  ASSERT_TRUE(session.mGenerator);
  EXPECT_TRUE(WaitForReady(session).isNothing());

  auto pending = StartGenerate(
      session, StoryRequest(/* aMaxTokens */ 512, /* aBufferLength */ 1));
  ASSERT_TRUE(SpinUntil("first delta", kStepTimeoutSeconds,
                        [&] { return !session.mStreamed.IsEmpty(); }));
  EXPECT_TRUE(session.mGenerator->SendCancel());

  auto result = WaitForOrTimeout(std::move(pending), kStepTimeoutSeconds);
  ASSERT_TRUE(result.isSome())
  << "cancelled Generate never resolved";
  ASSERT_TRUE(result->isOk());
  GenerateResponse response = result->unwrap();
  ASSERT_EQ(response.type(), GenerateResponse::TGenerateResult);
  const GenerateResult& generated = response.get_GenerateResult();
  EXPECT_EQ(generated.reason(), dom::TextGenerationFinishReason::Cancelled);
  EXPECT_EQ(session.mStreamed, generated.content());
  EXPECT_LT(generated.usage().generatedTokens(), 512u)
      << "cancel should have stopped generation early";

  Teardown(session);
}

TEST_F(TextGenerationTest, CancelDoesNotLeakIntoNextGenerate) {
  ipc::FileDescriptor model = OpenModelFd(kModelFile);
  ASSERT_TRUE(model.IsValid())
  << "missing " << kModelFile;

  Session session = CreateGenerator(model);
  ASSERT_TRUE(session.mGenerator);
  EXPECT_TRUE(WaitForReady(session).isNothing());

  auto firstPending = StartGenerate(session, StoryRequest(/* aMaxTokens */ 512,
                                                          /* aBufferLength */
                                                          1));
  EXPECT_TRUE(session.mGenerator->SendCancel());
  auto first = WaitForOrTimeout(std::move(firstPending), kStepTimeoutSeconds);
  ASSERT_TRUE(first.isSome() && first->isOk());
  GenerateResponse firstResponse = first->unwrap();
  ASSERT_EQ(firstResponse.type(), GenerateResponse::TGenerateResult);
  EXPECT_EQ(firstResponse.get_GenerateResult().reason(),
            dom::TextGenerationFinishReason::Cancelled);

  session.mStreamed.Truncate();
  GenerateResponse secondResponse = Generate(session, StoryRequest());
  ASSERT_EQ(secondResponse.type(), GenerateResponse::TGenerateResult);
  const GenerateResult& second = secondResponse.get_GenerateResult();
  EXPECT_NE(second.reason(), dom::TextGenerationFinishReason::Cancelled)
      << "the earlier Cancel must not apply to this Generate";
  EXPECT_FALSE(second.content().IsEmpty());
  EXPECT_GT(second.usage().generatedTokens(), 0u);

  Teardown(session);
}

TEST_F(TextGenerationTest, WhenReadyAnswersAfterTheLoadReported) {
  ipc::FileDescriptor model = OpenModelFd(kModelFile);
  ASSERT_TRUE(model.IsValid())
  << "missing " << kModelFile;

  Session session = CreateGenerator(model);
  ASSERT_TRUE(session.mGenerator);

  ASSERT_TRUE(WaitForReady(session).isNothing())
  << "construction should load the model";

  ASSERT_TRUE(WaitForReady(session).isNothing())
  << "a late WhenReady() must still see the successful load";

  Teardown(session);
}

TEST_F(TextGenerationTest, InvalidFdFailsAtConstruction) {
  Session session = CreateGenerator(ipc::FileDescriptor());
  ASSERT_TRUE(session.mGenerator);

  Maybe<nsCString> loadError = WaitForReady(session);
  ASSERT_TRUE(loadError.isSome());
  EXPECT_FALSE(loadError->IsEmpty());

  GenerateResponse response =
      Generate(session, StoryRequest(), /* aErrorExpected */ true);
  ASSERT_EQ(response.type(), GenerateResponse::TGenerateError);
  EXPECT_FALSE(response.get_GenerateError().message().IsEmpty());

  Teardown(session);
}
