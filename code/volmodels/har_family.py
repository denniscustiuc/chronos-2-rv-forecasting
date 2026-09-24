"""HAR extensions that need realized measures beyond RV itself.

These are the specifications Brini (2026) benchmarks alongside HAR and Log-HAR.
Each adds a regressor built from a companion measure that a 5-minute dataset
supplies but daily OHLC cannot reconstruct -- which is exactly why this family
only becomes available once VOLARE is wired in:

``HARJ``
    Adds the **jump** component ``J = max(RV - BV, 0)`` (Andersen, Bollerslev &
    Diebold 2007).  Bipower variation is robust to jumps, so the gap between RV
    and BV isolates the discontinuous part of the price path -- which is known
    to predict future volatility differently from the continuous part.

``HARRS``
    Splits the daily term into **realized semivariances** ``RS+`` and ``RS-``
    (Patton & Sheppard 2015).  Downside realized variance predicts future
    volatility far more strongly than upside does; forcing a single coefficient
    on their sum throws that asymmetry away.

``HARQ``
    Interacts the daily coefficient with **realized quarticity**
    (Bollerslev, Patton & Quaedvlieg 2016): ``(beta_1 + beta_1Q * sqrt(RQ_t))
    * RV_t``.  RQ estimates the measurement error in RV, so the model leans on
    the daily term less when today's RV is itself noisily measured.

**Units.** Every model here estimates in *variance* space -- these regressors
are variances and mixing them with a volatility target would be dimensionally
incoherent.  The harness feeds volatility, so the input is squared on the way in
and the forecast is square-rooted on the way out.  That round trip is exactly
what QLIKE undoes when it squares the forecast again, so the loss is computed on
the variance forecast the model actually produced.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from .base import DEFAULT_QUANTILE_LEVELS, NotEnoughData
from .har import MIN_REGRESSION_ROWS, MONTHLY_LAG, WEEKLY_LAG, HARFit, _HARBase

logger = logging.getLogger(__name__)

__all__ = ["HARJ", "HARRS", "HARQ", "EXOG_COLUMNS"]

#: Exogenous columns each model requires, using the canonical names
#: :mod:`volpipe.volare` emits.
EXOG_COLUMNS: dict[str, tuple[str, ...]] = {
    "har_j": ("jump",),
    "har_rs": ("var_rsp", "var_rsn"),
    "harq": ("var_rq",),
}


def _cascade(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily, weekly and monthly trailing averages ending at each index."""
    series = pd.Series(values, dtype="float64")
    return (
        series.to_numpy(),
        series.rolling(WEEKLY_LAG, min_periods=WEEKLY_LAG).mean().to_numpy(),
        series.rolling(MONTHLY_LAG, min_periods=MONTHLY_LAG).mean().to_numpy(),
    )


class _VarianceHARBase(_HARBase):
    """Shared machinery for the variance-space HAR extensions.

    Subclasses override :meth:`_extra_regressors` to append their own columns to
    the standard ``[1, RV_d, RV_w, RV_m]`` cascade.
    """

    #: Canonical exog columns this model cannot run without.
    required_exog: tuple[str, ...] = ()

    def _transform(self, values: np.ndarray) -> np.ndarray:
        """Volatility in, variance out."""
        return values**2

    def _inverse_transform(self, value):
        """Variance out, volatility back -- negatives clipped to zero."""
        return np.sqrt(np.clip(value, 0.0, None))

    def _extra_regressors(self) -> np.ndarray:
        """Model-specific columns, shape ``(n, k)``, aligned to the history."""
        raise NotImplementedError

    def _rebuild(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Recompute the cascade design plus this model's extra regressors."""
        clean = self._store_history(history, exog)
        variance = self._transform(clean.to_numpy())
        self._target = variance

        daily, weekly, monthly = _cascade(variance)
        base = np.column_stack([np.ones(variance.size), daily, weekly, monthly])
        extra = self._extra_regressors()
        self._design = np.column_stack([base, extra]) if extra.size else base

    def _estimate(self, h: int) -> HARFit:
        """Direct-``h`` OLS on the extended design matrix."""
        design, target = self._design, self._target
        n = target.size
        if n <= h:
            raise NotEnoughData(f"{self.name}: history of {n} is too short for horizon {h}")

        features, outcomes = design[: n - h], target[h:]
        usable = np.isfinite(features).all(axis=1) & np.isfinite(outcomes)
        features, outcomes = features[usable], outcomes[usable]

        minimum = max(MIN_REGRESSION_ROWS, 5 * features.shape[1])
        if features.shape[0] < minimum:
            raise NotEnoughData(
                f"{self.name}: only {features.shape[0]} usable rows for horizon {h}, need {minimum}"
            )

        coefficients, *_ = np.linalg.lstsq(features, outcomes, rcond=None)
        residuals = outcomes - features @ coefficients
        dof = max(1, features.shape[0] - features.shape[1])
        return HARFit(
            horizon=h,
            coefficients=coefficients,
            residuals=residuals,
            residual_variance=float(residuals @ residuals / dof),
            n_obs=int(features.shape[0]),
        )


class HARJ(_VarianceHARBase):
    """HAR plus the jump component ``J = max(RV - BV, 0)``.

    Regression: ``RV_{t+h} = b0 + b1*RV_d + b2*RV_w + b3*RV_m + bJ*J_t``.

    Requires the ``jump`` column, which :mod:`volpipe.volare` derives from
    ``rv5 - bv``.
    """

    required_exog = EXOG_COLUMNS["har_j"]

    def __init__(self, *, quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS) -> None:
        super().__init__("har_j", quantile_levels=quantile_levels, insanity_filter=True)

    def _extra_regressors(self) -> np.ndarray:
        jump = self._exog_column("jump")
        return jump.reshape(-1, 1)


class HARRS(_VarianceHARBase):
    """HAR with the daily term split into realized semivariances.

    Regression:
    ``RV_{t+h} = b0 + b_p*RS+_t + b_n*RS-_t + b2*RV_w + b3*RV_m``.

    The daily RV column is dropped, since ``RS+ + RS- = RV`` would make the
    design exactly collinear.
    """

    required_exog = EXOG_COLUMNS["har_rs"]

    def __init__(self, *, quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS) -> None:
        super().__init__("har_rs", quantile_levels=quantile_levels, insanity_filter=True)

    def _rebuild(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Build ``[1, RS+, RS-, RV_w, RV_m]`` -- daily RV replaced by its parts."""
        clean = self._store_history(history, exog)
        variance = self._transform(clean.to_numpy())
        self._target = variance

        _, weekly, monthly = _cascade(variance)
        positive = self._exog_column("var_rsp")
        negative = self._exog_column("var_rsn")
        self._design = np.column_stack(
            [np.ones(variance.size), positive, negative, weekly, monthly]
        )

    def _extra_regressors(self) -> np.ndarray:  # pragma: no cover - _rebuild overridden
        return np.empty((0, 0))


class HARQ(_VarianceHARBase):
    """HAR with the daily coefficient scaled by realized quarticity.

    Regression:
    ``RV_{t+h} = b0 + (b1 + b1Q*sqrt(RQ_t))*RV_d + b2*RV_w + b3*RV_m``,
    implemented by adding ``sqrt(RQ_t) * RV_d`` as a fourth regressor.

    ``sqrt(RQ)`` is on a very different scale from ``RV``, and the interaction
    term can be enormous, so it is standardised by its in-sample mean before
    entering the design.  Without that the OLS is badly conditioned -- which is
    the most likely reason HARQ posts such poor loss ratios in Brini's Table 6
    (5.13 at h=1); the specification is genuinely fragile.
    """

    required_exog = EXOG_COLUMNS["harq"]

    def __init__(self, *, quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS) -> None:
        super().__init__("harq", quantile_levels=quantile_levels, insanity_filter=True)

    def _extra_regressors(self) -> np.ndarray:
        quarticity = self._exog_column("var_rq")
        daily = self._target
        root_rq = np.sqrt(np.clip(quarticity, 0.0, None))

        finite = root_rq[np.isfinite(root_rq)]
        scale = float(finite.mean()) if finite.size and finite.mean() > 0 else 1.0
        return ((root_rq / scale) * daily).reshape(-1, 1)
