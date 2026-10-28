# TESTING.md — what earns a test in this repo

`CLAUDE.md` says what must stay true. This file says what a test must do to
justify the seconds it costs and the attention it takes when it fails.

Measured 2026-10-28: 2,256 tests, 12m42s. The slowest thirty account for about
3,400 CPU-seconds; the long tail is nearly free. **Cost is concentrated, so
cleanup is about the expensive few and the pointless many, not about counting
tests.**

## The one question

**What defect would this test have caught, and could that defect have reached
production?**

If you cannot name the defect, the test is decoration. If the defect could not
reach production — because the type system, the database schema or the language
already forbids it — the test is decoration with a runtime cost.

## What earns its place

1. **A defect that actually happened.** The best tests here are incident
   records: the retrain/trader parity check had been red for a day before
   anyone looked; the funnel check caught a refusal reason going invisible on
   the dashboard; `test_cancel_order_path.py` pins a URL that was wrong for
   this system's entire history. Write the measurement into the docstring so
   the next reader knows what it is defending.

2. **An invariant from `CLAUDE.md`.** Point-in-time rules, the fee schedule,
   one `decide()` for backtest and live, the window-level CV split. These are
   the contract; a test is how the contract is enforced.

3. **A negative control.** A null that produces no skill, a canary that fails
   when the target is shifted, a duplicate-rows control. `RESEARCH_PROTOCOL.md`
   makes these mandatory, and **a stubbed control detects nothing** — these are
   allowed to be slow.

4. **A boundary where two things must agree.** Live against backtest, our
   arithmetic against the venue's, training against serving. Drift between two
   correct-looking halves is the defect class this repo keeps hitting.

5. **A silent failure mode.** NaN that substitutes a default, a 404 read as
   "normal", a flag declared and never read, a column that is always null.
   Nothing raises, so only a test complains.

## What does not earn its place

* **Testing the language or the library.** That a dataclass stores what was
  passed, that `sorted()` sorts, that pandas concatenates.
* **Testing a constant equals itself.** `assert TIMEOUT == 60` when the next
  line of production code reads `TIMEOUT`. It restates, it does not check.
* **Restating the implementation.** If the assertion is the function body
  rewritten, any change breaks the test for no reason. Assert the OUTCOME.
* **A happy path already covered by a richer test**, unless it is the fast
  smoke for something slow.
* **Mocking so much that only the mock is exercised.** If every collaborator
  is stubbed, the test asserts the stub's behaviour.
* **Snapshot assertions on text nobody reads** — log wording, a help string,
  an error message that carries no contract.

## Trivial SUBJECT is not trivial TEST

Be careful here, and prefer keeping to deleting when unsure.

A one-line assertion on a boring field can be the only thing standing between
a production defect and silence. Before deleting, check:

* Does the docstring name a measurement, a date or an incident? **Keep it.**
* Does it assert a cross-component agreement (two configs, two code paths,
  a schema and its reader)? **Keep it.**
* Would it still pass if you deleted the production line it covers?
  **Then it is not testing that line** — that is the real signal for deletion.

The last one is the decisive check and it is cheap: delete the production
line, run the test, see whether it fails.

## Cost rules

* **A test that trains a real model must justify it.** Structural assertions —
  a column exists, folds do not overlap, a histogram sums — do not need a
  fitted model. Use a tiny fit or a prebuilt artifact.
* **Mark the genuinely slow ones `slow`**, so `-m "not slow"` is a real fast
  loop rather than a rounding error.
* **Share expensive fixtures at module or session scope**, and remember that
  `pytest-xdist` caches per WORKER: with `--dist load` the same module fixture
  is rebuilt on every worker that gets one of its tests. Use `--dist loadfile`.
* **`-n 0` when iterating on one file.** `-n auto` costs ~4.2s of worker
  startup before a single assertion runs.

## Writing one

Name the test as the claim it makes, in a sentence:
`test_a_resumed_streak_does_not_immediately_re_halt`, not `test_halt_2`.
The docstring carries what the name cannot: the measurement, the incident, the
reason the number is what it is. Follow `CLAUDE.md` on comments — short
sentences, active voice, no history that git already holds, except where the
history IS the reason the test exists.
