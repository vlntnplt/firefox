# Working guidelines for this branch

How work happens on `hwinference-rollout`, so that every commit is reviewable,
explainable, and reproducible. These rules operationalize the principles the
rollout leans on: tests assert behavior visible from outside, patterns are
grounded in tree precedent, tests cannot be gamed by the implementation, and
the author is the first reviewer — which means every commit must give that
reviewer what they need to actually review.

## Branch and stack shape

- Base is Paul's Bug 2051970 stack, pulled from Phabricator at pinned diffs
  on top of upstream main, excluding his WIP smoke-test prototype (D313227),
  which never enters our base. Re-pulling the stack or moving to a newer
  main is its own operation, never mixed with content changes.
- Fixed stack order: upstream cherry-picks and provisional fixes first, then
  the working documents (`ipc_design.md`, `rollout.md`, this file), then
  fixes to existing code the work uncovered, then the new-work commits in
  bug order. Doc updates that record decisions are folded into the documents
  commit on the next history rewrite; the documents describe the stack as it
  stands, not its history.
- Provisional commits are titled `PROVISIONAL - ...`. Each is a fix or
  refactor our stack needs in the unlanded base, sits directly above the
  base, gets proposed upstream on the revision it patches, and is dropped when
  that revision takes it. They never land from here.
- Landable commits are titled `Bug NN - ...` with NN a placeholder (01, 02,
  ...) to be substituted with Bugzilla bug IDs at submission time.
- One commit per bug, with a message that summarizes every change in it.
  A standalone product fix the work uncovers gets its own commit in the
  fixes-on-existing-code group, never buried inside a feature commit.
- A commit is born formatted: `./mach lint --fix` runs on its own files
  before it is committed, and no commit carries formatting churn on files it
  does not otherwise change. Formatting drift in mirrored provisional files
  is flagged upstream, not fixed here.

## Commit messages: what and why, never a paraphrase

A commit message gives the reviewer quick context; the diff itself is
open next to it. Three short parts:

1. **What** the commit does, in a sentence or two of observable behavior —
   never a file-by-file narration of the diff.
2. **Why** it sits where it does, plus only what the diff cannot show: the
   contract being established, the in-tree precedent followed (file and
   line), and deliberate departures from it. If a claim in the rollout doc
   turned out wrong during implementation, say so and follow up on the doc.
No verification footers: CI and the try run are the verification
record, a prose claim of green is not reviewable, and `artifacts/`
paths mean nothing in tree history. The discipline itself is a working
rule of this branch, not message content: run the gates before every
commit and keep the logs in `artifacts/`; if something cannot run
locally (missing model, missing hardware), record that in `artifacts/`
instead of implying green; never claim a test passes without having run
it.

## Tests

- A test asserts behavior observable from outside: generated text, telemetry
  presence and shape, perf numbers, a rendered preview. Never internals of the
  code under migration.
- Never weaken or edit a test to make an implementation pass — if an
  existing expectation looks wrong, stop and surface it; changing one needs
  evidence and sign-off, in its own commit with the reason.
- A test added at the base of the stack must run unchanged at the tip. That
  comparability is the entire point of the Bug 03 tests: no assertion may
  depend on which backend or process serves the request, and no option may
  be one the new engine will not honor.
- Golden determinism is a property to protect: greedy sampling, pinned run
  options, pinned models. Any change to pinned outputs is its own commit with
  the reason in the message.
- Slow commands (`./mach test`, mochitest, perftest) redirect output to
  `artifacts/` and the log is read selectively; never pipe them through
  filters and never rerun a slow command to extract something the log already
  has.

## Hygiene

- Precedent before pattern: find and read the in-tree example before writing
  new IPDL, actors, bindings, or telemetry; name it in the commit message. If
  the precedent contradicts the plan, stop and surface it rather than
  improvising.
- `./mach lint --fix` and `./mach format` on touched files before every
  commit; front-end-test-only changes need no build.
- Comments in code stay minimal per tree style; the narrative lives in the
  commit message.
- No emoji anywhere. No try pushes, no submissions to Phabricator; the user
  triggers CI and review.
- Environment for anything that builds or runs:
  `MOZCONFIG=/home/leaf/work/firefox/mozconfig-release`.
