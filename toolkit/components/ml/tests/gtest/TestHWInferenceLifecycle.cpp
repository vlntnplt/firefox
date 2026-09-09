/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "gtest/gtest.h"
#include "mozilla/Preferences.h"
#include "mozilla/SpinEventLoopUntil.h"
#include "mozilla/TimeStamp.h"
#include "mozilla/gtest/WaitFor.h"
#include "mozilla/gtest/ipc/TestUtilityProcess.h"
#include "mozilla/hwinference/HWInferenceLifecycleHelpers.h"
#include "mozilla/hwinference/HWInferenceParent.h"
#include "mozilla/ipc/UtilityProcessManager.h"
#include "nsIProcessToolsService.h"
#include "nsServiceManagerUtils.h"
#include "nsThreadUtils.h"

using namespace mozilla;
using namespace mozilla::hwinference;

namespace {

using ipc::UtilityProcessKeepAlive;

constexpr auto kKind = kBrowserProcessKind;

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

bool ProcessIsUp() { return ProcessPid().isSome(); }
bool ProcessIsGone() { return ProcessPid().isNothing(); }

already_AddRefed<UtilityProcessKeepAlive> Acquire() {
  return ipc::UtilityProcessManager::GetSingleton()
      ->AcquireBrowserHWInferenceProcess();
}

void Release(RefPtr<UtilityProcessKeepAlive>& aKeepAlive) {
  ReleaseBrowserProcessAfterGrace(std::move(aKeepAlive));
  aKeepAlive = nullptr;
}

// The process is up once its actor is bound; an error means it never came up.
Result<bool, nsresult> WaitForReady() {
  return WaitFor(HWInferenceParent::GetSingleton(kKind)->WhenReady());
}

// Kills the process the way a crash would, and waits for the manager to
// notice.
void Crash(base::ProcessId aPid) {
  nsCOMPtr<nsIProcessToolsService> tools =
      do_GetService("@mozilla.org/processtools-service;1");
  ASSERT_TRUE(tools);
  ASSERT_TRUE(NS_SUCCEEDED(tools->Kill(aPid)));
  ASSERT_TRUE(SpinUntil("Crash noticed", kStepTimeoutSeconds, ProcessIsGone));
}

}  // namespace

class HWInferenceLifecycleTest
    : public mozilla::gtest::ipc::TestUtilityProcess {
 protected:
  // Zero grace: most cases assert the process exits with its last release.
  void SetUp() override { Preferences::SetUint(kGracePref, 0); }
};

TEST_F(HWInferenceLifecycleTest, DiesWithLastRelease) {
  EXPECT_FALSE(IsBrowserProcessUp());
  RefPtr<UtilityProcessKeepAlive> keepAlive = Acquire();
  ASSERT_TRUE(keepAlive);
  ASSERT_TRUE(WaitForReady().isOk());
  EXPECT_TRUE(ProcessIsUp());
  EXPECT_TRUE(IsBrowserProcessUp());

  Release(keepAlive);
  EXPECT_TRUE(SpinUntil("Process exit", kStepTimeoutSeconds, ProcessIsGone))
      << "last keep-alive released but the process survived";
}

TEST_F(HWInferenceLifecycleTest, IdleGraceKeepsProcessForReuse) {
  // Long enough to reacquire inside, short enough to wait out at the end: a
  // deferred drop cannot be recalled by lowering the pref afterwards.
  constexpr uint32_t kGraceMs = 2000;
  Preferences::SetUint(kGracePref, kGraceMs);

  RefPtr<UtilityProcessKeepAlive> first = Acquire();
  ASSERT_TRUE(first);
  ASSERT_TRUE(WaitForReady().isOk());
  Maybe<base::ProcessId> pid = ProcessPid();
  ASSERT_TRUE(pid.isSome());

  Release(first);
  EXPECT_EQ(ProcessPid(), pid)
      << "the idle grace should have kept the process alive";

  EXPECT_TRUE(IsBrowserProcessUp());
  RefPtr<UtilityProcessKeepAlive> second = Acquire();
  ASSERT_TRUE(second);
  EXPECT_EQ(ProcessPid(), pid)
      << "an acquire inside the idle grace should reuse the live process";

  TimeStamp dropped = TimeStamp::Now();
  Release(second);
  EXPECT_TRUE(SpinUntil("Process exit", kStepTimeoutSeconds, ProcessIsGone))
      << "last keep-alive released but the process survived";
  EXPECT_GE((TimeStamp::Now() - dropped).ToMilliseconds(), kGraceMs * 0.8)
      << "the process went away before the second grace could expire";
}

TEST_F(HWInferenceLifecycleTest, IdleGraceExpiryRetiresProcess) {
  constexpr uint32_t kGraceMs = 50;
  Preferences::SetUint(kGracePref, kGraceMs);

  RefPtr<UtilityProcessKeepAlive> keepAlive = Acquire();
  ASSERT_TRUE(keepAlive);
  ASSERT_TRUE(WaitForReady().isOk());

  TimeStamp dropped = TimeStamp::Now();
  Release(keepAlive);
  EXPECT_TRUE(
      SpinUntil("Idle grace expiry", kStepTimeoutSeconds, ProcessIsGone))
      << "the deferred drop never retired the process";
  // Only a lower bound: a loaded machine makes the timer fire late, never
  // early, so this catches a timer armed too short without being flaky.
  EXPECT_GE((TimeStamp::Now() - dropped).ToMilliseconds(), kGraceMs * 0.8)
      << "the process went away before the idle grace could expire";
}

TEST_F(HWInferenceLifecycleTest, LaunchFailureLaunchesAfresh) {
  auto manager = ipc::UtilityProcessManager::GetSingleton();
  ASSERT_TRUE(manager);
  // An unrelated kind keeps the manager singleton alive across the failed
  // launch, as it goes away with its last process.
  ASSERT_TRUE(
      WaitFor(manager->LaunchProcess(ipc::SandboxingKind::GENERIC_UTILITY))
          .isOk());

  RefPtr<UtilityProcessKeepAlive> keepAlive = Acquire();
  ASSERT_TRUE(keepAlive);
  EXPECT_FALSE(IsBrowserProcessUp())
      << "a pending launch is not a live process";
  // Nothing spins the event loop in between, so the launch is still pending
  // and this fails it.
  manager->CleanShutdown(kKind);
  ASSERT_TRUE(WaitForReady().isErr())
  << "the actor of a failed launch must report it";

  // The next acquire launches afresh even while the failed keep-alive lives.
  EXPECT_FALSE(IsBrowserProcessUp());
  RefPtr<UtilityProcessKeepAlive> second = Acquire();
  ASSERT_TRUE(second);
  EXPECT_NE(second, keepAlive);
  ASSERT_TRUE(WaitForReady().isOk());
  Maybe<base::ProcessId> pid = ProcessPid();
  ASSERT_TRUE(pid.isSome());

  Release(keepAlive);
  EXPECT_EQ(ProcessPid(), pid)
      << "the keep-alive of a failed launch must not touch its replacement";
  Release(second);
  EXPECT_TRUE(SpinUntil("Process exit", kStepTimeoutSeconds, ProcessIsGone))
      << "last keep-alive released but the process survived";

  manager->CleanShutdown(ipc::SandboxingKind::GENERIC_UTILITY);
  NS_ProcessPendingEvents(nullptr);
}

TEST_F(HWInferenceLifecycleTest, ProcessDeathLaunchesAfresh) {
  RefPtr<UtilityProcessKeepAlive> keepAlive = Acquire();
  ASSERT_TRUE(keepAlive);
  ASSERT_TRUE(WaitForReady().isOk());
  Maybe<base::ProcessId> pid = ProcessPid();
  ASSERT_TRUE(pid.isSome());

  Crash(*pid);

  // The next acquire launches afresh even while the dead keep-alive lives.
  EXPECT_FALSE(IsBrowserProcessUp());
  RefPtr<UtilityProcessKeepAlive> second = Acquire();
  ASSERT_TRUE(second);
  EXPECT_NE(second, keepAlive);
  ASSERT_TRUE(WaitForReady().isOk());
  Maybe<base::ProcessId> newPid = ProcessPid();
  ASSERT_TRUE(newPid.isSome());
  EXPECT_NE(*newPid, *pid);

  Release(keepAlive);
  EXPECT_EQ(ProcessPid(), newPid)
      << "the keep-alive of a dead process must not touch its replacement";
  Release(second);
  EXPECT_TRUE(SpinUntil("Process exit", kStepTimeoutSeconds, ProcessIsGone))
      << "last keep-alive released but the process survived";
}
