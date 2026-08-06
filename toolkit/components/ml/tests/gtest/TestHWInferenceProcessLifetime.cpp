/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "gtest/gtest.h"
#include "mozilla/Preferences.h"
#include "mozilla/Result.h"
#include "mozilla/TimeStamp.h"
#include "mozilla/gtest/WaitFor.h"
#include "mozilla/gtest/ipc/TestUtilityProcess.h"
#include "mozilla/hwinference/HWInferenceBrowserManagerParent.h"
#include "mozilla/hwinference/HWInferenceParent.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "nsThreadUtils.h"

using namespace mozilla;
using namespace mozilla::hwinference;

namespace {

// Generous for a debug CI machine; a healthy run needs a fraction of it.
constexpr uint32_t kStepTimeoutSeconds = 120;

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
      "TestHWInferenceProcessLifetime WaitForOrTimeout"_ns,
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

// The shutdown decision is one main-thread dispatch from the reservation
// destructor, so a runnable queued after the drop runs after it.
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

using Reservation = HWInferenceBrowserManagerParent::Reservation;

RefPtr<Reservation> Reserve() {
  auto result = WaitForOrTimeout(HWInferenceBrowserManagerParent::GetOrCreate(),
                                 kStepTimeoutSeconds);
  if (result.isNothing() || result->isErr()) {
    printf_stderr("Reserve: GetOrCreate() timed out or rejected\n");
    return nullptr;
  }
  return result->unwrap();
}

}  // namespace

class HWInferenceProcessLifetimeTest
    : public mozilla::gtest::ipc::TestUtilityProcess {
 protected:
  // Zero grace period: most cases assert the process exits immediately.
  void SetUp() override {
    Preferences::SetUint("browser.ml.hwInference.processTimeout", 0);
  }
};

TEST_F(HWInferenceProcessLifetimeTest, DiesWithLastReservation) {
  RefPtr<Reservation> reservation = Reserve();
  ASSERT_TRUE(reservation);
  ASSERT_TRUE(reservation->Manager()->CanSend());
  EXPECT_FALSE(reservation->ProcessReused());
  EXPECT_FALSE(ProcessIsGone());

  reservation = nullptr;
  EXPECT_TRUE(
      SpinUntil("Reservation process exit", kStepTimeoutSeconds, ProcessIsGone))
      << "last reservation gone but the HWInference process survived";
}

TEST_F(HWInferenceProcessLifetimeTest, ConcurrentCreatesShareOneLaunch) {
  auto firstPending = HWInferenceBrowserManagerParent::GetOrCreate();
  auto secondPending = HWInferenceBrowserManagerParent::GetOrCreate();

  auto first = WaitForOrTimeout(std::move(firstPending), kStepTimeoutSeconds);
  auto second = WaitForOrTimeout(std::move(secondPending), kStepTimeoutSeconds);
  ASSERT_TRUE(first.isSome() && first->isOk());
  ASSERT_TRUE(second.isSome() && second->isOk());

  RefPtr<Reservation> a = first->unwrap();
  RefPtr<Reservation> b = second->unwrap();
  EXPECT_EQ(a->Manager(), b->Manager())
      << "racing creates should share the browser manager";

  a = nullptr;
  EXPECT_FALSE(ProcessIsGone())
      << "a surviving reservation holder must keep the process";
  b = nullptr;
  EXPECT_TRUE(
      SpinUntil("Reservation process exit", kStepTimeoutSeconds, ProcessIsGone))
      << "last reservation gone but the HWInference process survived";
}

TEST_F(HWInferenceProcessLifetimeTest, IdleWindowKeepsProcessForReuse) {
  Preferences::SetUint("browser.ml.hwInference.processTimeout", 120000);

  RefPtr<Reservation> first = Reserve();
  ASSERT_TRUE(first);
  RefPtr<HWInferenceBrowserManagerParent> idleManager = first->Manager();
  EXPECT_FALSE(first->ProcessReused());
  first = nullptr;

  FlushShutdownDecision();
  EXPECT_FALSE(ProcessIsGone())
      << "the idle timeout should have kept the process alive";

  RefPtr<Reservation> second = Reserve();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->Manager(), idleManager.get())
      << "a caller inside the idle window should reuse the live manager";
  EXPECT_TRUE(second->ProcessReused());

  // Back to eager teardown so the last drop shuts the process down.
  Preferences::SetUint("browser.ml.hwInference.processTimeout", 0);
  second = nullptr;
  EXPECT_TRUE(
      SpinUntil("Reservation process exit", kStepTimeoutSeconds, ProcessIsGone))
      << "last reservation gone but the HWInference process survived";
}

TEST_F(HWInferenceProcessLifetimeTest, IdleWindowExpiryRetiresProcess) {
  constexpr uint32_t kIdleTimeoutMs = 50;
  Preferences::SetUint("browser.ml.hwInference.processTimeout", kIdleTimeoutMs);

  RefPtr<Reservation> reservation = Reserve();
  ASSERT_TRUE(reservation);
  EXPECT_FALSE(ProcessIsGone());

  TimeStamp dropped = TimeStamp::Now();
  reservation = nullptr;
  EXPECT_TRUE(
      SpinUntil("Idle window expiry", kStepTimeoutSeconds, ProcessIsGone))
      << "the idle timer never retired the process";
  // Only a lower bound: a loaded machine makes the timer fire late, never
  // early, so this catches a timer armed too short without being flaky.
  EXPECT_GE((TimeStamp::Now() - dropped).ToMilliseconds(), kIdleTimeoutMs * 0.8)
      << "the process went away before the idle window could expire";
}
