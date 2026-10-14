"""The classifier, which predicts a *correction* to the baseline.

The architecture is one decision and it makes everything else honest: the
baseline's logit enters LightGBM as an `init_score`, so the model fits the
residual. An untrained model reproduces the baseline exactly, every tree it
grows is incremental skill by construction, and "did it beat the baseline"
stops being a comparison of two numbers computed by different code paths and
becomes a property of a single fitted object.

Three things follow from that choice, and they are the reason for it:

* **The objective is the right one.** Cross-entropy on a binary label, not a
  regression on a return that is then thresholded. The previous incarnation of
  this project regressed net return and took its sign, which counted every flat
  bar as a miss and optimised magnitude accuracy nobody was paid for. Here the
  quantity being fitted is exactly the quantity being traded: a probability.

* **Abstention is native.** A calibrated probability compares directly against
  the price, so "no trade" is not a separate model or a tuned threshold on a
  score of unknown units — it is `edge below the gate`, in probability points,
  against a break-even the fee schedule defines.

* **Overconfidence is measurable in one number.** `residual_scale` is a single
  coefficient fitted on held-out training rows: how much of the correction the
  model claims actually survives out of sample. The last version of this repo
  discovered its predictions were thirty-four times too confident only by
  regressing realised on predicted after the fact. Here it is `alpha`, it is
  fitted before anything is traded, and a value near zero says the model found
  nothing however good its in-sample loss looked.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from core.baseline import BarrierBaseline, clip_prob, expit, log_loss, logit
from core.config import Config, DEFAULT_CONFIG
from core.features import CONTROL_GROUPS, feature_columns, mirror_rows

logger = logging.getLogger(__name__)

BASELINE_LOGIT = 'baseline_probability_logit'
MARKET_LOGIT = 'market_probability_logit'

# Which forecaster the correction is fitted on top of. `Config.init_score_source`
# selects it; this maps the name to the column.
#
# **The baseline is the weaker of the two, measurably.** Over 1,109 live-recorded
# rows and 285 symbol-windows, log loss was 0.331 for the market's de-spread mid,
# 0.428 for `F(x/sigma)` and 0.430 for the model — the same sign on all three
# symbols and all four offsets, and on the 108 rows actually traded the model came
# in *worse* than its own baseline. So a model initialised on the baseline spends
# its capacity correcting a forecaster that is already 0.10 nats behind the price
# it has to trade against, and "beat the baseline" stops implying anything about
# whether the trade pays.
#
# Initialised on the market instead, three things change, and the second is the
# reason to want it:
#
# * The residual being fitted is `logit(truth) - logit(price)` — how the price is
#   wrong, which is exactly the quantity the money depends on.
# * **The null inverts, in the right direction.** An untrained baseline-init model
#   reproduces `F(x/sigma)`, which disagrees with the price by 5.79pp on average —
#   a large apparent edge that is mostly noise, and it is *live by default*. An
#   untrained market-init model reproduces the price, so the edge is identically
#   zero and nothing trades. The default state becomes "the price is right",
#   which is the honest prior.
# * `decide()`'s edge becomes the model's own output rather than a difference
#   between two independently-fitted things.
#
# It is not trainable yet, and the gap is not close: 285 symbol-windows of
# recorded quotes exist against a `windows_evaluated >= 20,000` gate — roughly
# seventy days of recording at ~285 a day. The mechanism is here, refusing
# clearly, so that the day the data exists this is a config change and not a
# rewrite.
INIT_SCORE_COLUMNS = {'baseline': BASELINE_LOGIT, 'market': MARKET_LOGIT}


def attach_market_logit(table: pd.DataFrame,
                        column: str = 'market_probability') -> pd.DataFrame:
    """Attach `market_probability_logit` for a market-initialised model.

    Rows with no quote stay NaN rather than borrowing the baseline. A market-init
    model that silently fell back to the baseline on a missing quote would be a
    baseline-init model wearing the other one's provenance, which is the precise
    failure this whole arrangement exists to avoid — and it would be invisible,
    because the two produce identically well-formed numbers.
    """
    out = table.copy()
    if column not in out.columns:
        raise ValueError(
            f'{column} is missing. A market-initialised model can only be scored '
            f'where a real quote was recorded; a backtest has no book, so this is '
            f'the expected failure there rather than something to work around.'
        )
    values = out[column].to_numpy(dtype=float)
    out[MARKET_LOGIT] = np.where(np.isfinite(values), logit(clip_prob(values)),
                                 np.nan)
    return out
# Fraction of the training windows held back to fit `residual_scale` and to
# early-stop on. Taken from the *end* of training, so it is the most recent
# data and the shrinkage is measured on the regime nearest the test block.
# Fewest windows an inner block may hold and still be split in two. Below
# this, early stopping and the shrinkage fit share rows and alpha reads high.
MIN_INNER_BLOCK_WINDOWS = 200

# Fewest scoreable rows the shrinkage may be fitted on. Small enough that an
# outage does not stop an evaluation, large enough that alpha is not noise.
MIN_SHRINKAGE_ROWS = 500

INNER_VALIDATION_FRACTION = 0.2

# Bounds on the blend's init coefficient and intercept, and the ridge pulling
# them to (1, 0). The ridge is in pseudo-rows of log loss, so it fades as the
# sample grows and only restrains a small one.
BLEND_INIT_BOUNDS = (0.5, 1.5)
BLEND_INTERCEPT_BOUNDS = (-0.5, 0.5)
BLEND_RIDGE = 50.0

# Standardized logistic inputs are clipped here so one outlier cannot dominate.
LOGISTIC_CLIP = 5.0


def _feature_matrix(table: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f'feature columns absent from the table: {missing}')
    return table.loc[:, list(columns)].to_numpy(dtype=float)


@dataclass
class LinearCorrection:
    """An L2 logistic correction exposing the Booster surface ForecastModel uses.

    No intercept: a base-rate shift is not a correction to the forecaster, and
    an untrained (all-zero) model must reproduce the init exactly.
    """

    names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    l2: float = 1.0
    n_iter: int = 0
    best_iteration: Optional[int] = None

    def _standardize(self, matrix: np.ndarray) -> np.ndarray:
        z = (np.asarray(matrix, dtype=float) - self.mean) / self.scale
        z = np.clip(z, -LOGISTIC_CLIP, LOGISTIC_CLIP)
        return np.where(np.isfinite(z), z, 0.0)

    def predict(self, matrix: np.ndarray, raw_score: bool = True) -> np.ndarray:
        raw = self._standardize(matrix) @ self.coef
        return raw if raw_score else expit(raw)

    def feature_name(self) -> list[str]:
        return list(self.names)

    def num_feature(self) -> int:
        return len(self.names)

    def feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        if importance_type == 'split':
            return (self.coef != 0).astype(int)
        return np.abs(self.coef)

    @classmethod
    def fit(cls, matrix: np.ndarray, names: Sequence[str], init: np.ndarray,
            outcome: np.ndarray, *, weights: Optional[np.ndarray] = None,
            l2: float = 1.0) -> 'LinearCorrection':
        from scipy import optimize

        matrix = np.asarray(matrix, dtype=float)
        init = np.asarray(init, dtype=float)
        outcome = np.asarray(outcome, dtype=float)
        keep = np.isfinite(init) & np.isfinite(outcome)
        matrix, init, outcome = matrix[keep], init[keep], outcome[keep]
        w = (np.ones(len(outcome)) if weights is None
             else np.asarray(weights, dtype=float)[keep])
        w = w / w.sum()
        with np.errstate(invalid='ignore'):
            mean = np.nan_to_num(np.nanmean(matrix, axis=0))
            scale = np.nan_to_num(np.nanstd(matrix, axis=0))
        # A constant column standardizes to zero and so contributes nothing.
        scale = np.where(scale > 0, scale, 1.0)
        model = cls(names=list(names), mean=mean, scale=scale,
                    coef=np.zeros(matrix.shape[1]), l2=float(l2))
        if matrix.shape[1] == 0:
            return model
        z = model._standardize(matrix)
        n = len(outcome)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            eta = init + z @ beta
            # log(1 + e^eta) - y*eta, stably.
            loss = float(np.sum(w * (np.logaddexp(0.0, eta) - outcome * eta)))
            grad = z.T @ (w * (expit(eta) - outcome))
            return (loss + 0.5 * l2 * float(beta @ beta) / n,
                    grad + l2 * beta / n)

        result = optimize.minimize(objective, model.coef, jac=True,
                                   method='L-BFGS-B')
        if not np.all(np.isfinite(result.x)):
            raise ValueError(f'the logistic correction did not converge '
                             f'({result.message!r})')
        model.coef = np.asarray(result.x, dtype=float)
        model.n_iter = int(getattr(result, 'nit', 0))
        return model


def _select_logistic_features(frame: pd.DataFrame, columns: Sequence[str],
                              init_column: str, top_n: int) -> list[str]:
    """The `top_n` columns most correlated with the residual on the fit rows."""
    residual = (frame['outcome'].to_numpy(dtype=float)
                - expit(frame[init_column].to_numpy(dtype=float)))
    rows = np.isfinite(residual)
    residual = residual[rows]
    # Standardized exactly as LinearCorrection will see it: clipped, NaN -> 0.
    probe = LinearCorrection(names=list(columns), mean=np.zeros(len(columns)),
                             scale=np.ones(len(columns)), coef=np.zeros(len(columns)))
    matrix = _feature_matrix(frame, columns)[rows]
    with np.errstate(invalid='ignore'):
        probe.mean = np.nan_to_num(np.nanmean(matrix, axis=0))
        scale = np.nan_to_num(np.nanstd(matrix, axis=0))
    probe.scale = np.where(scale > 0, scale, 1.0)
    z = probe._standardize(matrix)
    scores = {}
    for i, column in enumerate(columns):
        if np.std(z[:, i]) > 0 and np.std(residual) > 0:
            scores[column] = abs(float(np.corrcoef(z[:, i], residual)[0, 1]))
    ranked = sorted(scores, key=lambda c: -scores[c])
    return ranked[:max(int(top_n), 0)]


@dataclass
class ForecastModel:
    """A fitted correction to a fitted baseline. One object, one provenance."""

    booster: Any
    features: list[str]
    baseline: BarrierBaseline
    # Everything else the live path needs: the fitted volatility models and
    # seasonality factors. Without these the artifact can be evaluated and not
    # deployed, which is a distinction nothing surfaced until the live path
    # tried to score a window.
    scoring: Any = None
    residual_scale: float = 1.0
    groups: tuple[str, ...] = ()
    n_train_rows: int = 0
    n_train_windows: int = 0
    empty_features: tuple[str, ...] = ()
    best_iteration: Optional[int] = None
    train_log_loss: float = float('nan')
    inner_log_loss: float = float('nan')
    inner_baseline_log_loss: float = float('nan')
    # Which forecaster the correction sits on top of. Stored, because scoring a
    # market-init correction on a baseline logit (or the reverse) produces a
    # perfectly well-formed probability that answers a different question.
    init_score_source: str = 'baseline'
    config_provenance: dict = field(default_factory=dict)
    # The final layer is p = expit(init_coefficient*init + residual_scale*corr
    # + intercept). 'scale' pins the coefficient at 1 and the intercept at 0.
    # Artifacts pickled before these existed read them through getattr.
    final_layer: str = 'scale'
    init_coefficient: float = 1.0
    intercept: float = 0.0
    # Every booster when ensembled (booster is members[0]); empty for one.
    members: tuple = ()
    model_type: str = 'lgbm'

    @property
    def init_score_column(self) -> str:
        try:
            return INIT_SCORE_COLUMNS[self.init_score_source]
        except KeyError:                     # pragma: no cover - guarded on fit
            raise ValueError(
                f'unknown init_score_source {self.init_score_source!r}; '
                f'expected one of {sorted(INIT_SCORE_COLUMNS)}') from None

    # ---- prediction -----------------------------------------------------
    @property
    def boosters(self) -> list:
        return list(getattr(self, 'members', ()) or ()) or [self.booster]

    @property
    def ensemble_size(self) -> int:
        return len(self.boosters)

    def member_corrections(self, table: pd.DataFrame) -> np.ndarray:
        """Each member's raw logit correction, shape (members, rows)."""
        matrix = _feature_matrix(table, self.features)
        return np.vstack([np.asarray(b.predict(matrix, raw_score=True), dtype=float)
                          for b in self.boosters])

    def raw_correction(self, table: pd.DataFrame) -> np.ndarray:
        """The model's logit correction, before shrinkage (the member mean)."""
        if self.ensemble_size == 1:
            matrix = _feature_matrix(table, self.features)
            return np.asarray(self.booster.predict(matrix, raw_score=True), dtype=float)
        return self.member_corrections(table).mean(axis=0)

    def _init_logit(self, table: pd.DataFrame) -> np.ndarray:
        column = self.init_score_column
        if column not in table.columns:
            raise ValueError(
                f'{column} is missing — score through core.dataset.apply_fold, '
                f'which attaches it from the fold\'s own baseline, or '
                f'attach_market_logit for a market-initialised model. Recomputing '
                f'it here would silently use a different one.'
            )
        return table[column].to_numpy(dtype=float)

    def _final(self, base: np.ndarray, correction: np.ndarray, *,
               shrink: bool = True) -> np.ndarray:
        if not shrink:
            return clip_prob(expit(base + correction))
        alpha = self.residual_scale
        if getattr(self, 'final_layer', 'scale') == 'blend':
            a = float(getattr(self, 'init_coefficient', 1.0))
            c = float(getattr(self, 'intercept', 0.0))
            return clip_prob(expit(a * base + alpha * correction + c))
        return clip_prob(expit(base + alpha * correction))

    def score(self, table: pd.DataFrame) -> pd.DataFrame:
        """`model_probability`, plus the members' spread when ensembled.

        Member probabilities go through the same final layer as the mean, so
        `model_probability` always lies between `_min` and `_max`.
        """
        out = pd.DataFrame(index=table.index)
        out['model_probability'] = self.predict(table)
        if self.ensemble_size > 1:
            base = self._init_logit(table)
            members = np.vstack([self._final(base, c)
                                 for c in self.member_corrections(table)])
            members = np.where(np.isfinite(base), members, np.nan)
            with np.errstate(invalid='ignore'):
                out['model_probability_sd'] = members.std(axis=0)
                out['model_probability_min'] = members.min(axis=0)
                out['model_probability_max'] = members.max(axis=0)
        return out

    def predict(self, table: pd.DataFrame, *, shrink: bool = True) -> np.ndarray:
        """Calibrated probability that the window settles above its strike.

        A row whose init column is NaN comes back NaN. That is a market-init model
        meeting a window with no quote, and it must not fall through to the
        baseline: `decide()` abstains on a non-finite probability, and the
        `non_finite_share` gate counts them, so the abstention is both safe and
        visible. Substituting the baseline would be silent and would put the
        wrong provenance on a real trade.
        """
        base = self._init_logit(table)
        out = self._final(base, self.raw_correction(table), shrink=shrink)
        # Redundant today and kept deliberately. NaN already propagates through
        # `expit` and `np.clip`, so mutating this line away changes nothing and
        # mutation testing cannot tell the difference — which is exactly why the
        # guarantee is written down here and pinned by
        # `test_clip_prob_propagates_nan_which_this_relies_on`. A `clip_prob` that
        # one day filled NaN with 0.5 would turn "no quote" into "a coin flip,
        # confidently asserted" on a real-money path, silently.
        return np.where(np.isfinite(base), out, np.nan)

    def predict_baseline(self, table: pd.DataFrame) -> np.ndarray:
        return clip_prob(table['baseline_probability'].to_numpy(dtype=float))

    # ---- provenance -----------------------------------------------------
    def importance(self, kind: str = 'gain') -> pd.DataFrame:
        values = np.mean([np.asarray(b.feature_importance(importance_type=kind),
                                     dtype=float) for b in self.boosters], axis=0) \
            if self.ensemble_size > 1 else \
            self.booster.feature_importance(importance_type=kind)
        control = {c for g in CONTROL_GROUPS for c in feature_columns([g])}
        frame = pd.DataFrame({'feature': self.features, kind: values})
        frame['is_control'] = frame['feature'].isin(control)
        total = frame[kind].sum()
        frame['share'] = frame[kind] / total if total else np.nan
        return frame.sort_values(kind, ascending=False, ignore_index=True)

    @property
    def control_importance_share(self) -> float:
        """Share of total gain taken by the control group.

        Hour-of-day cannot forecast direction. A large share here does not mean
        the clock works; it means the measurement is picking up something that
        is not a forecast, and the last version of this project had its 27-cell
        survey won by exactly this group.
        """
        frame = self.importance()
        return float(frame.loc[frame['is_control'], 'share'].sum())

    def provenance(self) -> dict:
        return {
            'features': list(self.features),
            'n_features': len(self.features),
            # `features` is already the populated list, so subtracting the empty
            # ones again reported 28 for 35. It goes into every provenance record
            # the trial count is read from.
            'n_features_populated': len(self.features),
            'empty_features': list(self.empty_features),
            'groups': list(self.groups),
            'residual_scale': self.residual_scale,
            'final_layer': getattr(self, 'final_layer', 'scale'),
            'init_coefficient': float(getattr(self, 'init_coefficient', 1.0)),
            'intercept': float(getattr(self, 'intercept', 0.0)),
            'ensemble_size': self.ensemble_size,
            'model_type': getattr(self, 'model_type', 'lgbm'),
            # Which forecaster this is a correction to. A promoted artifact that
            # does not say is an artifact whose skill number means one of two
            # different things.
            'init_score_source': self.init_score_source,
            'n_train_rows': self.n_train_rows,
            'n_train_windows': self.n_train_windows,
            'best_iteration': self.best_iteration,
            'train_log_loss': self.train_log_loss,
            'inner_log_loss': self.inner_log_loss,
            'inner_baseline_log_loss': self.inner_baseline_log_loss,
            'inner_log_loss_skill': self.inner_baseline_log_loss - self.inner_log_loss,
            'control_importance_share': self.control_importance_share,
            'baseline': self.baseline.provenance(),
            'deployable': self.scoring is not None,
            'config': self.config_provenance,
        }

    @property
    def deployable(self) -> bool:
        """Can this artifact score a window it has never seen?"""
        return self.scoring is not None

    def _blend_summary(self) -> str:
        if getattr(self, 'final_layer', 'scale') != 'blend':
            return ''
        return (f' a={float(getattr(self, "init_coefficient", 1.0)):.3f}'
                f' c={float(getattr(self, "intercept", 0.0)):+.3f}')

    def summary(self) -> str:
        skill = self.inner_baseline_log_loss - self.inner_log_loss
        return (
            f'model: {len(self.features)} features '
            f'({len(self.empty_features)} empty), {self.best_iteration} trees'
            f'{f" x{self.ensemble_size}" if self.ensemble_size > 1 else ""}, '
            f'alpha={self.residual_scale:.3f}'
            f'{self._blend_summary()} | inner log loss {self.inner_log_loss:.5f} '
            f'vs baseline {self.inner_baseline_log_loss:.5f} '
            f'(skill {skill:+.5f}) | control gain share '
            f'{self.control_importance_share:.1%}'
        )

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix('.provenance.json').write_text(
            json.dumps(self.provenance(), indent=2, default=str))
        return path

    def integrity(self) -> dict:
        """A snapshot of the artifact's feature shape, for a human or a script.

        **This is not the safety check** — `verify()` below is. `verify()` is
        what `load()` calls and what actually raises on a mismatched feature
        list; this docstring used to describe that same failure mode as if
        `integrity()` were the thing guarding against it, while nothing called
        it and it never raised anything. It is introspection: `load()` logs it
        at debug level so a mismatch that `verify()` allowed through (no config
        passed, so no comparison to make) is still visible in the log rather
        than only inferable from wrong numbers downstream.
        """
        names = list(self.booster.feature_name())
        return {
            'features': list(self.features),
            'booster_features': names,
            'n_features': self.booster.num_feature(),
        }

    def verify(self, config: Optional[Config] = None) -> None:
        """Raise unless this artifact can be scored as it stands.

        Called at load. The fields compared against `config` are the ones that
        change an answer rather than a path: a model fitted at offsets (3,6,9,12)
        scored at (1,2) is being asked a question it was never fitted for, and
        `core/dataset.py` only warned. `scripts/live.py` did not read
        `config_provenance` at all.
        """
        names = list(self.booster.feature_name())
        if names != list(self.features):
            raise ValueError(
                f'the booster was trained on {len(names)} columns and the artifact '
                f'lists {len(self.features)}; the first disagreement is at '
                f'{next((i for i, (a, b) in enumerate(zip(names, self.features)) if a != b), 0)}. '
                f'The feature matrix is built by name from the artifact\'s list, so '
                f'this would score a well-formed matrix of the wrong columns.'
            )
        if self.init_score_source not in INIT_SCORE_COLUMNS:
            raise ValueError(
                f'this artifact records init_score_source='
                f'{self.init_score_source!r}, which nothing can score')
        if config is None:
            return
        wanted = getattr(config, 'init_score_source', 'baseline')
        if wanted != self.init_score_source:
            raise ValueError(
                f'this artifact was fitted on the {self.init_score_source} logit '
                f'and the running configuration asks for {wanted}. The correction '
                f'is a residual of one specific forecaster, so adding it to the '
                f'other one produces a well-formed probability that answers a '
                f'different question — which is exactly how this would go '
                f'unnoticed.'
            )
        stored = dict(self.config_provenance or {})
        if not stored:
            logger.warning('this artifact records no config provenance, so nothing '
                           'can be checked against the running configuration')
            return
        current = config.provenance()
        # Only the fields that change a number. Paths, versions and CLI notes
        # differ legitimately between the run that fitted and the run that scores.
        material = ('window_minutes', 'decision_offsets', 'vol_lookbacks_minutes',
                    'embargo_minutes', 'fee_rate', 'maker_fee_rate', 'assume_maker',
                    'half_spread_cents', 'min_traded_price', 'max_traded_price')
        drift = {
            key: (stored.get(key), current.get(key))
            for key in material
            if key in stored and key in current and stored[key] != current[key]
        }
        if drift:
            detail = '; '.join(f'{k}: fitted {a!r}, running {b!r}'
                              for k, (a, b) in sorted(drift.items()))
            raise ValueError(
                f'this artifact was fitted under a different configuration — '
                f'{detail}. Scoring it here would answer a different question '
                f'than the one the gates were evaluated on.'
            )

    @staticmethod
    def load(path: str | Path, config: Optional[Config] = None) -> 'ForecastModel':
        import joblib
        model = joblib.load(Path(path))
        # Verified at load, not at first use. An artifact that cannot be scored
        # should fail where the operator is looking, not several minutes into a
        # cycle with a quote in hand.
        model.verify(config)
        logger.debug('artifact integrity: %s', model.integrity())
        return model


def _fit_residual_scale(
    baseline_logit: np.ndarray,
    correction: np.ndarray,
    outcome: np.ndarray,
    *,
    focus_quantile: float = 0.0,
    max_scale: float = 2.0,
) -> float:
    """One coefficient: how much of the claimed correction survives.

    Fitted by minimising held-out log loss over a single scalar, which is the
    smallest recalibration that can exist and therefore the one least able to
    manufacture skill. Clipped to [0, 2]: a negative alpha would mean the model
    is anti-predictive and the honest response is to report zero skill rather
    than to invert it, which is curve-fitting on the validation split.

    **`focus_quantile` fits it where the money is.** Alpha over ALL rows
    optimises the average, and the average is fine: measured on 7,611 live
    rows, the model's pooled ECE is only 0.0032 worse than the market's. But
    the ~9% of rows that become trades are a selected sample -- `decide()`
    takes them precisely where the model most disagrees with the price, which
    is where it is most likely to be wrong -- and there the model claimed 65.07%
    and delivered 58.53%, a **6.7pp overconfidence** (90% CI [+0.73, +12.55],
    P(real) 97%) that accounts for essentially the whole gap between a +4.90pp
    predicted edge and a -1.32pp realised one.

    Alpha had no reason to correct that: it was never shown those rows
    separately. The shrinkage was fitted where the money is not.

    Set to `q`, the fit runs on rows whose `|correction|` is in the top `1-q`
    of the distribution. That subset is a proxy for "rows we would trade" and,
    crucially, is **alpha-invariant**: alpha scales the correction
    monotonically, so the RANKING does not move and there is no fixed point to
    chase. 0.0 keeps the historical behaviour.

    This is not free. A smaller sample is a noisier alpha, and over-shrinking
    is its own error -- so the subset is still held to `MIN_SHRINKAGE_ROWS`,
    and the quantile backs off rather than raising when there is not enough.
    """
    from scipy import optimize

    # Exclude rows that carry no forecast, rather than optimising over them.
    #
    # A single NaN makes the objective NaN at every alpha, `minimize_scalar` gives
    # up, and it returns its golden-section bracket seed 0.7639320225 with
    # `success=False` — which was never checked, so the *overfitting detector*
    # returned a search constant that sails past its own `residual_scale >= 0.25`
    # gate. On a five-year BTC walk-forward four of six folds returned exactly
    # that number, and those four folds carried the reported skill.
    #
    # Refusing outright was the first fix here and it was wrong: a 6.5-hour
    # Coinbase outage leaves ~83 rows in 26,488 without a volatility estimate, and
    # that killed the whole evaluation. The defect was never the NaN, it was
    # reporting an abandoned search as a fitted value. So drop the unscoreable
    # rows, insist enough remain to mean anything, and still raise if the fit
    # itself does not converge.
    finite = (np.isfinite(baseline_logit) & np.isfinite(correction)
              & np.isfinite(np.asarray(outcome, dtype=float)))
    dropped = int((~finite).sum())
    if dropped:
        logger.info(
            '%d of %d shrinkage rows carry no forecast (a NaN sigma, usually the '
            'tail of a data outage) and are excluded', dropped, finite.size)
    baseline_logit = np.asarray(baseline_logit)[finite]
    correction = np.asarray(correction)[finite]
    outcome = np.asarray(outcome, dtype=float)[finite]
    if outcome.size < MIN_SHRINKAGE_ROWS:
        raise ValueError(
            f'only {outcome.size} scoreable rows remain of {finite.size}, which '
            f'is under the {MIN_SHRINKAGE_ROWS} needed to fit a shrinkage that '
            f'means anything. That is a data problem, not a model one.'
        )

    # **Restrict to the rows that become trades, if asked.** Done after the
    # finite filter and the minimum-rows check, so a focus that would leave too
    # little to fit backs off to the full sample with a line saying so rather
    # than raising -- a noisy alpha fitted on 40 rows is worse than an honest
    # one fitted on all of them.
    if focus_quantile and 0.0 < focus_quantile < 1.0:
        cut = float(np.quantile(np.abs(correction), focus_quantile))
        keep = np.abs(correction) >= cut
        if int(keep.sum()) >= MIN_SHRINKAGE_ROWS:
            logger.info(
                'fitting the shrinkage on the %d row(s) above the %.0fth '
                'percentile of |correction| — the subset that becomes trades, '
                'where the model is selected for disagreeing with the price',
                int(keep.sum()), 100 * focus_quantile)
            baseline_logit = baseline_logit[keep]
            correction = correction[keep]
            outcome = outcome[keep]
        else:
            logger.warning(
                'the top %.0f%% of |correction| is only %d row(s), under the %d '
                'minimum, so the shrinkage is fitted on all %d instead',
                100 * (1 - focus_quantile), int(keep.sum()),
                MIN_SHRINKAGE_ROWS, outcome.size)

    def objective(alpha: float) -> float:
        return log_loss(outcome, expit(baseline_logit + float(alpha) * correction))

    # **The ceiling is a claim about what alpha MEANS.** It answers "how much
    # of the claimed correction survives out of sample", and more than 100%
    # surviving is not an answer to that question -- it is the validation split
    # asking to AMPLIFY a correction, which is an overfitting signature rather
    # than a finding.
    #
    # Measured over 28 weekly refits with the ceiling at 2.0: alpha ran 0.532
    # to 2.000, median 1.231, sd 0.426, and amplified on 19 of 28. It does not
    # settle with more data (sd 0.49 on the larger folds against 0.35 on the
    # smaller, correlation with fold size -0.13), so this is not small-sample
    # noise. And it is the wrong direction where the money is: the rows that
    # become trades are 6.7pp OVERconfident, so amplifying damages exactly the
    # subset that pays.
    #
    # `max_residual_scale` is the ceiling. 1.0 says a correction may be
    # discounted but never inflated; 2.0 is the historical behaviour.
    result = optimize.minimize_scalar(objective, bounds=(0.0, max_scale),
                                      method='bounded')
    if not result.success or not np.isfinite(result.x):
        raise ValueError(
            f'the shrinkage fit did not converge ({getattr(result, "message", "")!r}). '
            f'Reporting its abandoned bracket point as a fitted alpha is how an '
            f'unfitted constant reaches a gate.'
        )
    return float(np.clip(result.x, 0.0, max_scale))


def _fit_blend(
    init_logit: np.ndarray,
    correction: np.ndarray,
    outcome: np.ndarray,
    *,
    min_edge: float,
    max_scale: float = 1.0,
    intercept: bool = False,
) -> tuple[float, float, float]:
    """Fit `p = expit(a*init + b*correction + c)` on the rows that would trade.

    Returns `(a, b, c)`. "Would trade" is the raw correction moving the
    probability by at least `min_edge`: the selected subset where a single
    alpha fitted on every row was measured to be overconfident. The selection
    reads only covariates, never the outcome, so it does not bias the fit.
    """
    from scipy import optimize

    init_logit = np.asarray(init_logit, dtype=float)
    correction = np.asarray(correction, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    finite = np.isfinite(init_logit) & np.isfinite(correction) & np.isfinite(outcome)
    init_logit, correction, outcome = (init_logit[finite], correction[finite],
                                       outcome[finite])
    if outcome.size < MIN_SHRINKAGE_ROWS:
        raise ValueError(
            f'only {outcome.size} scoreable rows remain of {finite.size}, which '
            f'is under the {MIN_SHRINKAGE_ROWS} needed to fit a final layer')

    move = np.abs(expit(init_logit + correction) - expit(init_logit))
    trade = move >= float(min_edge)
    if int(trade.sum()) >= MIN_SHRINKAGE_ROWS:
        logger.info('fitting the blend on the %d of %d row(s) that would trade',
                    int(trade.sum()), outcome.size)
        init_logit, correction, outcome = (init_logit[trade], correction[trade],
                                           outcome[trade])
    else:
        logger.warning(
            'only %d row(s) would trade, under the %d minimum, so the blend is '
            'fitted on all %d', int(trade.sum()), MIN_SHRINKAGE_ROWS, outcome.size)

    n = outcome.size
    ridge = BLEND_RIDGE / n

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = theta[0], theta[1]
        c = theta[2] if intercept else 0.0
        eta = a * init_logit + b * correction + c
        loss = float(np.mean(np.logaddexp(0.0, eta) - outcome * eta))
        resid = (expit(eta) - outcome) / n
        grad = [float(resid @ init_logit) + 2 * ridge * (a - 1.0),
                float(resid @ correction)]
        loss += ridge * ((a - 1.0) ** 2 + c ** 2)
        if intercept:
            grad.append(float(resid.sum()) + 2 * ridge * c)
        return loss, np.asarray(grad)

    bounds = [BLEND_INIT_BOUNDS, (0.0, float(max_scale))]
    start = [1.0, min(1.0, float(max_scale))]
    if intercept:
        bounds.append(BLEND_INTERCEPT_BOUNDS)
        start.append(0.0)
    result = optimize.minimize(objective, np.asarray(start), jac=True,
                               method='L-BFGS-B', bounds=bounds)
    if not np.all(np.isfinite(result.x)):
        raise ValueError(f'the blend fit did not converge ({result.message!r})')
    if not result.success:
        logger.warning('the blend fit stopped early (%s); using its last point',
                       result.message)
    a, b = float(result.x[0]), float(result.x[1])
    c = float(result.x[2]) if intercept else 0.0
    return a, b, c


def _finite_log_loss(outcome: np.ndarray, base_logit: np.ndarray,
                     correction: np.ndarray, alpha: float,
                     a: float = 1.0, c: float = 0.0) -> float:
    """Log loss over the rows that carry a forecast.

    These three numbers go into the artifact's provenance and its `summary()`
    line, and they were computed over unfiltered arrays — so a single row with a
    NaN sigma printed `inner log loss nan vs baseline nan (skill +nan)` for the
    whole model. Observed on real bars, where one venue outage is enough.
    """
    outcome = np.asarray(outcome, dtype=float)
    keep = (np.isfinite(outcome) & np.isfinite(base_logit) & np.isfinite(correction))
    if not keep.any():
        return float('nan')
    return log_loss(outcome[keep],
                    expit(a * base_logit[keep] + alpha * correction[keep] + c))


def fit_model(
    train: pd.DataFrame,
    baseline: BarrierBaseline,
    config: Config = DEFAULT_CONFIG,
    *,
    groups: Optional[Sequence[str]] = None,
    weights: Optional[np.ndarray] = None,
    scoring: Any = None,
) -> ForecastModel:
    """Fit the residual classifier on a training slice.

    The inner validation split is taken from the *end* of training — the most
    recent windows — and is used for both early stopping and the shrinkage
    coefficient. Splitting at random would put four offsets of the same window
    on both sides and the early-stopping signal would be measuring memorisation.
    """
    import lightgbm as lgb

    columns = feature_columns(groups)
    source = getattr(config, 'init_score_source', 'baseline')
    if source not in INIT_SCORE_COLUMNS:
        raise ValueError(f'init_score_source={source!r}; expected one of '
                         f'{sorted(INIT_SCORE_COLUMNS)}')
    init_column = INIT_SCORE_COLUMNS[source]
    if init_column not in train.columns:
        extra = ''
        if source == 'market':
            # The likeliest way to arrive here, and it is not a bug to route
            # around: `walk_forward` builds its tables from bars and has no book,
            # so a market-initialised model cannot be backtested at all. Saying
            # that here is the difference between a clear refusal and someone
            # attaching the baseline column to make the error go away.
            extra = (' A backtest has no order book, so a market-initialised '
                     'model can only be fitted and scored on live-recorded '
                     'quotes. Substituting the baseline here would make "beat '
                     'the price" and "beat the baseline" the same question '
                     'answered twice with the same number.')
        raise ValueError(f'{init_column} is missing from the training table '
                         f'(init_score_source={source!r}).{extra}')
    if source == 'market':
        # Only on this path, and it is not a tolerance dial. A row with no quote
        # cannot carry a market residual at all — there is nothing to be a
        # residual of — so it is dropped rather than tolerated. Keeping it would
        # hand LightGBM an init score that is NaN precisely where our recording
        # failed, and "did we have a quote" is a property of our uptime, not of
        # the market.
        #
        # A blanket coverage floor across both sources was tried first and was
        # wrong: the baseline logit is legitimately non-finite on a small share of
        # rows (a venue outage leaves a real hole), test fixtures sit at 98.5%,
        # and `non_finite_share` already gates that end. Two earlier fixes in this
        # repo failed the same way — a `non_finite_rows == 0` gate and a shrinkage
        # guard that raised on any NaN — so the baseline path is left exactly as
        # it was.
        finite = np.isfinite(train[init_column].to_numpy(dtype=float))
        dropped = int((~finite).sum())
        if dropped:
            logger.warning(
                'dropping %d of %d training rows with no recorded quote: a '
                'market-initialised correction has nothing to correct there',
                dropped, len(train))
        train = train.loc[finite].copy()
        if train.empty:
            raise ValueError(
                f'every training row has a non-finite {init_column}, so there is '
                f'no market to fit a residual against')

    if getattr(config, 'label_source', 'proxy') == 'venue':
        # Everything downstream (fit, early stopping, final layer, the logged
        # losses) reads `outcome`, so the label is swapped once, here.
        proxy = train['outcome'].to_numpy(dtype=float)
        venue = (pd.to_numeric(train['venue_outcome'], errors='coerce').to_numpy(dtype=float)
                 if 'venue_outcome' in train.columns else np.full(len(train), np.nan))
        has_venue = np.isfinite(venue)
        logger.info(
            'label_source=venue: %d of %d training rows carry the venue label '
            '(%d disagree with the proxy); %d fall back to the proxy',
            int(has_venue.sum()), len(train),
            int((has_venue & np.isfinite(proxy) & (venue != proxy)).sum()),
            int((~has_venue).sum()))
        train = train.assign(outcome=np.where(has_venue, venue, proxy))

    populated = [
        c for c in columns
        if c in train.columns and np.isfinite(train[c].to_numpy(dtype=float)).any()
    ]
    empty = tuple(c for c in columns if c not in populated)
    if empty:
        logger.warning(
            'these features are entirely NaN and carry no information: %s. '
            'A group that produced nothing arrives with the same shape as one '
            'that worked, so this is the only place it is visible.', list(empty))
    if not populated:
        raise ValueError('no populated features')

    windows = pd.DatetimeIndex(sorted(train['window_open'].unique()))
    if len(windows) < 20:
        raise ValueError(f'{len(windows)} training windows is not enough to fit')
    cut = windows[int(len(windows) * (1.0 - INNER_VALIDATION_FRACTION))]

    # Purge the inner split the same way the outer one is purged.
    #
    # This boundary used to be hard: the last inner-training window was fifteen
    # minutes from the first inner-validation window, against the 1,440 the outer
    # CV insists on for exactly the same stated reason. Two things are fitted
    # here — LightGBM's `best_iteration` and `residual_scale` — and
    # `residual_scale` is the single number guarding against overconfidence, with
    # its own gate. Leaking into it does not inflate the reported out-of-sample
    # skill, which is measured on the properly embargoed outer fold; it ships a
    # model whose shrinkage is too weak and makes its gate easier to pass than
    # intended.
    embargo = pd.Timedelta(minutes=int(config.embargo_minutes))
    inner_train = train.loc[train['window_open'] < cut - embargo]
    if inner_train.empty:
        # A short training slice cannot afford the full embargo. Say so rather
        # than silently dropping it — the window is still purged by the label's
        # own horizon, which is the part that must not be skipped.
        fallback = pd.Timedelta(minutes=int(config.window_minutes))
        logger.warning(
            'the %d-minute inner embargo leaves no training rows in a slice of '
            '%d windows; falling back to %s. The shrinkage this fits will be '
            'optimistic.', config.embargo_minutes, len(windows), fallback)
        inner_train = train.loc[train['window_open'] < cut - fallback]

    # Early stopping and the shrinkage must not share rows. `residual_scale`
    # answers "how much of the claimed correction survives out of sample", and
    # measured on the same rows that chose the tree count it answers "how much
    # survives on the rows the tree count was selected for" — it read 0.902 on a
    # provably zero-signal null. Split the validation block in two.
    valid_windows = windows[windows >= cut]
    holdout = train.loc[train['window_open'] >= cut]
    inner_valid, alpha_rows = holdout, holdout
    if len(valid_windows) >= 2 * MIN_INNER_BLOCK_WINDOWS:
        mid = valid_windows[len(valid_windows) // 2]
        stop_block = train.loc[(train['window_open'] >= cut)
                               & (train['window_open'] < mid - embargo)]
        alpha_block = train.loc[train['window_open'] >= mid]
        if not stop_block.empty and not alpha_block.empty:
            inner_valid, alpha_rows = stop_block, alpha_block
    if inner_valid is alpha_rows:
        logger.warning(
            'the inner validation block (%d windows) is too small to separate '
            'early stopping from the shrinkage fit; alpha will be measured on '
            'the rows the tree count was chosen for and will read high.',
            len(valid_windows))

    if inner_valid.empty or inner_train.empty or alpha_rows.empty:
        raise ValueError('inner validation split is empty')

    def frame_weights(frame: pd.DataFrame) -> Optional[np.ndarray]:
        if weights is None:
            return None
        return np.asarray(weights, dtype=float)[frame.index.to_numpy()]

    # The rows the correction is fitted on. Mirrored copies join here and only
    # here: early stopping and the final layer must judge real rows.
    fit_rows, fit_weights = inner_train, frame_weights(inner_train)
    if getattr(config, 'mirror_augment', False):
        fit_rows = pd.concat([inner_train, mirror_rows(inner_train)],
                             ignore_index=True)
        if fit_weights is not None:
            fit_weights = np.concatenate([fit_weights, fit_weights])
        logger.info('mirror augmentation: %d inner-train rows fitted as %d',
                    len(inner_train), len(fit_rows))

    def dataset(frame: pd.DataFrame, w: Optional[np.ndarray], reference=None):
        return lgb.Dataset(
            _feature_matrix(frame, populated),
            label=frame['outcome'].to_numpy(dtype=float),
            init_score=frame[init_column].to_numpy(dtype=float),
            weight=w, feature_name=list(populated), reference=reference,
            free_raw_data=False,
        )

    model_type = getattr(config, 'model_type', 'lgbm')
    features = list(populated)
    if model_type == 'logistic':
        chosen = tuple(getattr(config, 'logistic_features', ()) or ())
        if chosen:
            unknown = [c for c in chosen if c not in populated]
            if unknown:
                raise ValueError(f'logistic_features not among the populated '
                                 f'features of these groups: {unknown}')
            features = list(chosen)
        else:
            features = _select_logistic_features(
                fit_rows, populated, init_column,
                int(getattr(config, 'logistic_top_n', 8)))
        logger.info('logistic correction on %d feature(s): %s',
                    len(features), features)
        members = [LinearCorrection.fit(
            _feature_matrix(fit_rows, features), features,
            fit_rows[init_column].to_numpy(dtype=float),
            fit_rows['outcome'].to_numpy(dtype=float),
            weights=fit_weights, l2=float(getattr(config, 'logistic_l2', 1.0)))]
    else:
        members = []
        for k in range(max(int(getattr(config, 'ensemble_size', 1) or 1), 1)):
            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
                'learning_rate': config.learning_rate, 'num_leaves': config.num_leaves,
                'max_depth': config.max_depth,
                'min_child_samples': config.min_child_samples,
                'bagging_fraction': config.subsample, 'bagging_freq': 1,
                'feature_fraction': config.colsample_bytree,
                'lambda_l2': config.reg_lambda,
                'deterministic': True, 'seed': 17 + k,
            }
            # Fresh datasets per member: a constructed Dataset pins seed-derived
            # parameters, and the members must differ in their bagging draws.
            train_set = dataset(fit_rows, fit_weights)
            valid_set = dataset(inner_valid, frame_weights(inner_valid),
                                reference=train_set)
            members.append(lgb.train(
                params, train_set, num_boost_round=config.n_estimators,
                valid_sets=[valid_set], valid_names=['inner'],
                callbacks=[lgb.early_stopping(config.early_stopping_rounds,
                                              verbose=False)],
            ))
    booster = members[0]

    def mean_correction(frame: pd.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, features)
        return np.mean([np.asarray(m.predict(matrix, raw_score=True), dtype=float)
                        for m in members], axis=0)

    correction = mean_correction(alpha_rows)
    base_logit = alpha_rows[init_column].to_numpy(dtype=float)
    outcome = alpha_rows['outcome'].to_numpy(dtype=float)
    max_scale = float(getattr(config, 'max_residual_scale', 2.0) or 2.0)
    final_layer = getattr(config, 'final_layer', 'scale')
    init_coefficient, intercept = 1.0, 0.0
    if final_layer == 'blend':
        init_coefficient, alpha, intercept = _fit_blend(
            base_logit, correction, outcome,
            min_edge=float(config.min_edge_pp) / 100.0, max_scale=max_scale,
            intercept=bool(getattr(config, 'final_layer_intercept', False)))
    else:
        alpha = _fit_residual_scale(
            base_logit, correction, outcome,
            focus_quantile=float(getattr(config, 'shrinkage_focus_quantile', 0.0) or 0.0),
            max_scale=max_scale)

    model = ForecastModel(
        booster=booster, features=features, baseline=baseline,
        scoring=scoring, residual_scale=alpha,
        groups=tuple(groups) if groups else (),
        n_train_rows=len(train), n_train_windows=len(windows),
        empty_features=empty, best_iteration=booster.best_iteration,
        init_score_source=source,
        train_log_loss=_finite_log_loss(
            inner_train['outcome'].to_numpy(dtype=float),
            inner_train[init_column].to_numpy(dtype=float),
            mean_correction(inner_train), alpha, init_coefficient, intercept),
        inner_log_loss=_finite_log_loss(outcome, base_logit, correction, alpha,
                                        init_coefficient, intercept),
        inner_baseline_log_loss=_finite_log_loss(outcome, base_logit, correction, 0.0),
        config_provenance=config.provenance(),
        final_layer=final_layer, init_coefficient=init_coefficient,
        intercept=intercept,
        members=tuple(members) if len(members) > 1 else (),
        model_type=model_type,
    )
    logger.info(model.summary())
    return model
