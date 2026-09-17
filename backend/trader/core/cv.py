"""Cross-validation at the window level, because the row is not the unit.

Four decision offsets share one settlement. They are four rows with different
features and the *same* label, so a split that puts offset 3 in train and
offset 12 in test has leaked the answer — not subtly, but completely: the two
rows describe the same fifteen minutes and one of them is nine minutes closer
to knowing. Every split here is therefore on `window_open`, and the row-level
frames are selected by membership rather than sliced.

**The embargo is a day, and it is not about the label.** A fifteen-minute label
needs a fifteen-minute purge. What needs twenty-four hours is the *features*: a
training row immediately after a test block computes `log_rv_1440` from bars
inside that block. Purging for the label and forgetting the feature lookback is
the standard version of this mistake, and it leaks in the direction that
inflates measured skill.

**Folds are expanding, not rolling.** Each fold trains on everything before its
test block. That matches how the thing would actually be deployed — you never
throw away history you have — and it means train is always entirely before
test, so only the gap immediately preceding the test block needs purging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LeakageError(AssertionError):
    """A fold's train and test sets are not separated as claimed."""


@dataclass(frozen=True)
class WindowFold:
    """One expanding-window fold, addressed by window-open timestamps."""

    index: int
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    embargo_minutes: int

    @property
    def train_end(self) -> Optional[pd.Timestamp]:
        return self.train.max() if len(self.train) else None

    @property
    def test_start(self) -> Optional[pd.Timestamp]:
        return self.test.min() if len(self.test) else None

    @property
    def test_end(self) -> Optional[pd.Timestamp]:
        return self.test.max() if len(self.test) else None

    @property
    def gap_minutes(self) -> float:
        if self.train_end is None or self.test_start is None:
            return float('nan')
        return (self.test_start - self.train_end).total_seconds() / 60.0

    def label(self) -> str:
        if self.test_start is None:
            return f'fold {self.index}: empty'
        return (f'fold {self.index}: train {len(self.train):,}w -> '
                f'test {len(self.test):,}w '
                f'[{self.test_start:%Y-%m-%d} .. {self.test_end:%Y-%m-%d}]')


def purged_walk_forward(
    window_index: pd.DatetimeIndex,
    *,
    n_folds: int = 6,
    embargo_minutes: int = 1440,
    min_train_windows: int = 500,
    min_test_windows: int = 200,
    scheme: str = 'calendar',
    fold_block_days: float = 21.0,
    train_days: Optional[float] = None,
) -> list[WindowFold]:
    """Split distinct window opens into expanding folds with a purged gap.

    The timeline is cut into `n_folds + 1` blocks; fold *i* trains on blocks
    0..i and tests on block i+1, with any training window inside the embargo of
    the test start removed. The first block is never a test block, which is what
    makes the first fold's training set non-trivial.

    **`scheme` decides what "equal" means, and it matters more than it sounds.**

    `'count'` cuts into equal numbers of windows — the original behaviour, and
    what every ledger entry before 2026-09-03 used. With data density varying 8x
    across the span (30 to 244 windows/day, as `--complete-cases` intersects
    three sources whose coverage improved through 2026), equal counts give
    calendar spans of 21 to 78 days. So the folds are not the same experiment:
    one fitted model tested across 78 days of drift is averaged with one tested
    across 21. Worse, appending data re-cuts EVERY boundary, so a fold's skill
    moves between runs when only the splitting changed — which happens on every
    run once collection is continuous.

    `'calendar'` cuts into equal spans of `fold_block_days`, ANCHORED to the
    first window, so earlier boundaries genuinely stay put as data arrives and
    fold-level results are comparable run to run. That was always the stated
    reason for this scheme and until 2026-09-16 the implementation did not
    deliver it — it divided the span between the first and LAST window, so every
    boundary slid on every append. Measured
    on 12,909 complete-case windows it is also simply a better estimator:

        equal-count      +0.00276 +/- 0.00088  t=+3.15  6/6
        equal-calendar   +0.00312 +/- 0.00075  t=+4.18  6/6

    The trade is unavoidable: with density varying you can balance time or
    count, never both. Equal-calendar put 491 windows in one fold against 5,671
    in another, so folds carry equal weight at unequal precision — inefficient
    but unbiased, and the price of a boundary that does not move.
    """
    index = pd.DatetimeIndex(sorted(pd.DatetimeIndex(window_index).unique()))
    if len(index) < (n_folds + 1) * 2:
        raise ValueError(
            f'{len(index)} windows cannot support {n_folds} folds — need at least '
            f'{(n_folds + 1) * 2}'
        )
    if scheme not in ('calendar', 'count'):
        raise ValueError(
            f"scheme={scheme!r}; expected 'calendar' or 'count'")
    if scheme == 'count':
        cuts = [index[e] if e < len(index) else index[-1] + pd.Timedelta(1)
                for e in np.linspace(0, len(index), n_folds + 2, dtype=int)]
    else:
        # **Equal spans of TIME, ANCHORED to the first window.**
        #
        # This used to be `pd.date_range(index.min(), index.max(),
        # periods=n_folds + 2)`, which divides the span between the first and
        # LAST window — so every interior boundary slides whenever data is
        # appended. Measured: adding 3 days to a 245-day span moved the cuts by
        # +10h, +21h, +31h, +41h, +51h, +62h, +72h. Fold 5's test block shifted
        # by more than two days.
        #
        # That contradicted this function's own documented reason for existing,
        # which said "earlier boundaries then stay put as data arrives, so
        # fold-level results are comparable run to run". They did not, and the
        # consequence showed up the moment promotion became weekly: the SAME
        # configuration on 3 extra days took `max_drawdown` 0.182 -> 0.388 and
        # `calibration_vs_market` -0.014 -> +0.009, failing three gates it had
        # passed. Almost none of that was the model.
        #
        # Anchoring needs a block length that does not depend on how much data
        # exists yet — that dependency IS the bug — so it is a stated parameter
        # rather than an emergent one. New data extends the final block and,
        # when it overflows, starts a new one; existing boundaries never move.
        block = pd.Timedelta(days=float(fold_block_days))
        origin = index.min()
        span = index.max() - origin
        # **COMPLETE blocks only.** The trailing stub — however much data has
        # arrived since the last boundary — is not a test block. Testing on it
        # would hand back a fold whose width changes every run, which is the
        # instability being removed, dressed up as coverage. The cost is real
        # and bounded: up to `fold_block_days` of the newest data sits outside
        # the test set until its block completes.
        n_blocks = max(int(span // block), 1)
        edges = [origin + k * block for k in range(n_blocks + 1)]
        # **EVERY complete block, not the most recent `n_folds + 1`.**
        #
        # Keeping only the newest blocks looked like tracking the market the
        # model will trade. It halved the evaluation: measured 2026-09-16 on a
        # 245-day span, `windows_evaluated` fell 21,307 -> 10,488 and failed its
        # own 20,000 gate, because six 21-day test blocks cover 126 days where
        # the count scheme tested six sevenths of everything. Stability is worth
        # paying for; half the sample is too high a price, and it was not a
        # trade anyone chose.
        #
        # Using all of them costs nothing in stability — the boundaries are
        # anchored either way — and folds now ACCUMULATE rather than roll off,
        # so a new block adds a fold instead of evicting the oldest. The fold
        # count therefore grows with history, which is why the caller treats
        # `n_folds` as a cap rather than a promise and why gates counting folds
        # must be proportions.
        cuts = edges
        if len(cuts) < 3:
            # Fewer than two blocks: there is no anchored grid to speak of, and
            # refusing outright would break every short-span research run
            # (`--end` experiments, a store only weeks old). Fall back to
            # subdividing what exists, and SAY SO — the boundaries are then
            # span-dependent again, which is exactly the property the anchoring
            # exists to provide, so a caller comparing runs must know it is
            # absent.
            logger.warning(
                'span %s is shorter than two %.0f-day blocks, so folds are '
                'subdivided proportionally and their boundaries WILL move as '
                'data arrives. Lower fold_block_days to anchor them.',
                span, float(fold_block_days))
            cuts = list(pd.date_range(index.min(), index.max(),
                                      periods=n_folds + 2))
            cuts[-1] = cuts[-1] + pd.Timedelta(minutes=1)
    embargo = pd.Timedelta(minutes=embargo_minutes)
    folds: list[WindowFold] = []
    # **How many folds there are is a property of the DATA under an anchored
    # grid, and of `n_folds` under a proportional one.**
    #
    # The count scheme divides the index into exactly `n_folds + 2` cuts, so it
    # always yields `n_folds`. An anchored calendar grid yields however many
    # complete blocks the span contains — fewer on a young store, more as
    # history accumulates — and capping that at `n_folds` is what halved the
    # evaluation: `windows_evaluated` fell 21,307 -> 10,488 and failed its own
    # 20,000 bar, because six 21-day blocks cover 126 days of a 245-day span.
    #
    # So the calendar scheme tests EVERY complete block after the first. Folds
    # accumulate instead of rolling off, the boundaries still never move, and
    # `n_folds` is a cap that only binds when someone asks for fewer than the
    # data supports. Gates that count folds must therefore be proportions —
    # "5 of 6" is a strong claim and "5 of 12" is a weak one.
    #
    # `n_folds` is NOT a cap here. See the note at the loop below.
    available = max(len(cuts) - 2, 0)
    # **`n_folds` does NOT cap the calendar scheme, and that is deliberate.**
    # The comment above once claimed it "only binds when someone asks for fewer
    # than the data supports"; that cannot be implemented while `Config.n_folds`
    # defaults to 6, because an explicit request for 6 is indistinguishable
    # from the default, and capping there re-halves coverage -- measured,
    # `windows_evaluated` 21,307 -> 10,488, failing its own 20,000 bar. Capping
    # was tried on 2026-09-16 and reverted for exactly that. Under `calendar`,
    # `n_folds` reaches only the minimum-windows guard and the short-span
    # fallback; to test fewer blocks, widen `fold_block_days`.
    wanted = available if scheme == 'calendar' else min(int(n_folds), available)
    first = max(available - wanted, 0)
    for i in range(first, available):
        test = index[(index >= cuts[i + 1]) & (index < cuts[i + 2])]
        if len(test) == 0:
            continue
        # **A block thin on DATA is not a fold, even when it is full on TIME.**
        # Anchored blocks are equal spans, and `--complete-cases` coverage
        # varies 8x across the history (30 to 244 windows/day), so an early
        # block can be a real 21 days holding a few hundred windows. Scoring it
        # as a peer of a 5,000-window block makes "the worst fold" a lottery —
        # which is the objection that kept gating on counts, and it is answered
        # here directly rather than by giving up equal spans.
        if len(test) < min_test_windows:
            logger.warning(
                'fold %d: %d test windows in [%s .. %s) is under the %d '
                'minimum, skipped — the block is full on time and thin on '
                'data', i, len(test), cuts[i + 1], cuts[i + 2],
                min_test_windows)
            continue
        train_pool = index[index < cuts[i + 1]]
        train = train_pool[train_pool < test[0] - embargo]
        # **A rolling training window, when asked for.** Cut from the embargo's
        # far edge rather than from the test start, so the window is `train_days`
        # of USABLE history and does not silently shrink by the embargo — a day
        # out of 35 is 3% of the sample, and a parameter that means 34 when it
        # says 35 is the kind of drift this file exists to prevent.
        #
        # Expanding stays the default because it is what the Sunday retrain
        # deploys. Setting this without matching it in the retrain reintroduces
        # the defect the quote-source work just closed: an evaluation measuring
        # one system while another trades.
        if train_days is not None and len(train):
            floor = (test[0] - embargo) - pd.Timedelta(days=float(train_days))
            train = train[train >= floor]
        if len(train) < min_train_windows:
            logger.warning(
                'fold %d: %d training windows is under the %d minimum, skipped',
                i, len(train), min_train_windows,
            )
            continue
        folds.append(WindowFold(index=i, train=train, test=test,
                                embargo_minutes=embargo_minutes))
    if not folds:
        raise ValueError('no fold had enough training windows')
    return folds


def folds_for_config(window_index, config) -> list[WindowFold]:
    """`purged_walk_forward` with EVERY geometry field read off the Config.

    **The single place folds are built from a Config, because the alternative
    was measured and it diverged.** `purged_walk_forward` carries its own
    defaults, and `scripts/train.py`, `scripts/baseline.py` and
    `scripts/ablate.py` each called it passing only `n_folds` and
    `embargo_minutes`. So `--fold-scheme count` was parsed, echoed in the
    header, recorded in provenance -- and ignored, while `core/backtest.py`
    forwarded all seven fields and cut a different experiment.

    That went from latent to live the moment `Config.fold_block_days` became
    7.0 while this module's default stayed 21.0: the gated numbers were cut
    into 7-day blocks and the ablation deciding which feature groups survive
    was cut into 21-day ones. Two different experiments, same run, no warning.

    Adding an argument to three call sites would fix today's instance and leave
    the fourth caller to be written wrong later. A function that takes the
    Config cannot be called with half of it.
    """
    return purged_walk_forward(
        window_index,
        n_folds=config.n_folds,
        embargo_minutes=config.embargo_minutes,
        scheme=getattr(config, 'fold_scheme', 'calendar'),
        fold_block_days=getattr(config, 'fold_block_days', 21.0),
        min_test_windows=getattr(config, 'min_test_windows', 200),
        train_days=getattr(config, 'fold_train_days', None))


def assert_no_leakage(fold: WindowFold) -> None:
    """Refuse a fold whose sets overlap or whose embargo is not honoured."""
    overlap = fold.train.intersection(fold.test)
    if len(overlap):
        raise LeakageError(
            f'fold {fold.index}: {len(overlap)} window opens are in both train and test'
        )
    if fold.train_end is None or fold.test_start is None:
        return
    if fold.train_end >= fold.test_start:
        raise LeakageError(
            f'fold {fold.index}: training ends {fold.train_end} at or after the '
            f'test start {fold.test_start}'
        )
    if fold.gap_minutes < fold.embargo_minutes:
        raise LeakageError(
            f'fold {fold.index}: gap of {fold.gap_minutes:.0f} minutes is under the '
            f'{fold.embargo_minutes}-minute embargo — a training row this close '
            f'computes its 1440-minute features from test-period bars'
        )


def rows_for(table: pd.DataFrame, window_opens: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask selecting every row belonging to these windows.

    Membership, not a timestamp comparison. A `>=`/`<` slice on `decision_time`
    would split a window across the boundary — the offset-3 row inside train and
    the offset-12 row inside test — which is the exact leak this module exists
    to prevent.
    """
    return table['window_open'].isin(window_opens)


def effective_observations(table: pd.DataFrame) -> int:
    """Distinct windows, not rows.

    Four offsets per window means a row count overstates the sample fourfold,
    and a standard error computed from it is half what it should be. Every
    reported error bar in this system divides by this.
    """
    if table.empty:
        return 0
    return int(table.drop_duplicates(['symbol', 'window_open']).shape[0])


def recency_weights(
    window_opens: pd.Series,
    half_life_days: Optional[float],
) -> Optional[np.ndarray]:
    """Exponential decay by age, or None when disabled.

    Off by default here, unlike the previous incarnation of this project, where
    a 50-day half-life meant five years of history bought one effective
    observation over one year. At 15-minute windows the sample is large enough
    that decay costs more than the non-stationarity it buys — but it is exposed
    so a run can disagree, and `scripts/evaluate.py` reports the sweep.
    """
    if not half_life_days:
        return None
    times = pd.DatetimeIndex(window_opens)
    age_days = (times.max() - times).total_seconds() / 86400.0
    return np.power(0.5, age_days / float(half_life_days))
