"""The HAR family (Corsi 2009) -- the classical benchmark in this literature.

HAR ("Heterogeneous AutoRegressive") regresses future realized volatility on
three trailing averages of past realized volatility -- one day, one week, one
month -- as a deliberately crude stand-in for the long-memory behaviour of
volatility.  It is three regressors and an intercept, it costs microseconds to
fit, and it is notoriously hard to beat.  That is precisely why it, and not
GARCH, is the benchmark: a forecasting model that cannot clear Log-HAR has not
demonstrated anything.

**Log-HAR is our headline baseline.**  Estimating in logs handles the strong
right skew of volatility, keeps forecasts positive by construction, and is the
standard specification in the modern RV literature.

Both models use **direct** multi-horizon forecasting: a separate regression is
estimated per horizon, mapping today's features straight onto the value ``h``
steps ahead.  The alternative -- iterating a one-step model forward -- compounds
its own errors and requires assumptions about the innovation path that direct
estimation simply sidesteps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .base import DEFAULT_QUANTILE_LEVELS, EmpiricalQuantileForecaster, NotEnoughData

logger = logging.getLogger(__name__)

__all__ = ["HAR", "LogHAR", "HARFit", "har_features"]

#: Corsi's three cascade lags: one day, one trading week, one trading month.
DAILY_LAG, WEEKLY_LAG, MONTHLY_LAG = 1, 5, 22

#: Rows of usable regression data required before we will estimate a horizon.
#: Four parameters need a comfortable multiple of four observations.
MIN_REGRESSION_ROWS = 30


@dataclass(frozen=True)
class HARFit:
    """Estimated coefficients for one horizon.

    Attributes:
        horizon: The horizon this regression targets.
        coefficients: ``[intercept, beta_daily, beta_weekly, beta_monthly]``.
        residuals: In-sample residuals, in the estimation space.
        residual_variance: Their variance (``ddof=4`` for the four parameters),
            used by Log-HAR's bias correction.
        n_obs: Number of regression rows used.
    """

    horizon: int
    coefficients: np.ndarray
    residuals: np.ndarray
    residual_variance: float
    n_obs: int


def har_features(values: np.ndarray) -> np.ndarray:
    """Build the HAR design matrix from a series of values.

    Row ``i`` holds ``[1, RV_i, mean(RV_{i-4..i}), mean(RV_{i-21..i})]`` -- the
    daily, weekly and monthly cascade components, each *ending at and including*
    ``i``.  Rows before ``MONTHLY_LAG - 1`` cannot be formed and come back as
    ``NaN``; callers drop them.

    Args:
        values: 1-D array of the series, oldest first.

    Returns:
        ``(n, 4)`` design matrix with an intercept in column 0.
    """
    series = pd.Series(values, dtype="float64")
    design = np.column_stack(
        [
            np.ones(series.size),
            series.to_numpy(),
            series.rolling(WEEKLY_LAG, min_periods=WEEKLY_LAG).mean().to_numpy(),
            series.rolling(MONTHLY_LAG, min_periods=MONTHLY_LAG).mean().to_numpy(),
        ]
    )
    return design


class _HARBase(EmpiricalQuantileForecaster):
    """Shared machinery for HAR and Log-HAR.

    Subclasses choose the estimation space by overriding :meth:`_transform` and
    :meth:`_inverse_transform`.  Coefficients are cached per horizon and
    deliberately survive :meth:`update`, so a backtest can re-estimate on a
    schedule while still conditioning every forecast on data current as of the
    origin.
    """

    def __init__(
        self,
        name: str,
        *,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        **kwargs,
    ) -> None:
        super().__init__(name, quantile_levels=quantile_levels, **kwargs)
        self._design: np.ndarray = np.empty((0, 4))
        self._target: np.ndarray = np.empty(0)
        self._fits: dict[int, HARFit] = {}

    # -- estimation space -------------------------------------------------- #

    def _transform(self, values: np.ndarray) -> np.ndarray:
        """Map volatility units into the estimation space."""
        return values

    # -- lifecycle --------------------------------------------------------- #

    def _rebuild(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Recompute the design matrix and target from a new history."""
        clean = self._store_history(history, exog)
        values = clean.to_numpy()
        transformed = self._transform(values)
        self._target = transformed
        self._design = har_features(transformed)

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Rebuild the design matrix and discard cached coefficients."""
        self._rebuild(history, exog)
        self._fits = {}

    def update(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Refresh the conditioning data, keeping the cached coefficients.

        This is the cheap path the backtest's ``refit_frequency`` uses: the
        design matrix is rebuilt so the forecast conditions on data through the
        new origin, but the OLS is not re-run.
        """
        self._rebuild(history, exog)

    # -- estimation -------------------------------------------------------- #

    def _estimate(self, h: int) -> HARFit:
        """Estimate the direct-``h`` regression on the stored history.

        Pairs feature row ``i`` with target ``i + h`` for every row where both
        exist.  Overlapping targets make the residuals an MA(h-1) process, which
        leaves the OLS coefficients consistent but invalidates the textbook
        standard errors -- we only use the point estimates and the residual
        distribution, so that is harmless here.  (The Diebold-Mariano test in
        :mod:`voleval.significance` does correct for exactly this dependence.)

        Raises:
            NotEnoughData: If fewer than :data:`MIN_REGRESSION_ROWS` usable
                rows are available.
        """
        design, target = self._design, self._target
        n = target.size
        if n <= h:
            raise NotEnoughData(f"{self.name}: history of {n} is too short for horizon {h}")

        features = design[: n - h]
        outcomes = target[h:]
        usable = np.isfinite(features).all(axis=1) & np.isfinite(outcomes)
        features, outcomes = features[usable], outcomes[usable]

        if features.shape[0] < MIN_REGRESSION_ROWS:
            raise NotEnoughData(
                f"{self.name}: only {features.shape[0]} usable rows for horizon {h}, "
                f"need {MIN_REGRESSION_ROWS}"
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

    def _fit_for(self, h: int) -> HARFit:
        """Return the cached fit for horizon ``h``, estimating it on demand."""
        if h not in self._fits:
            self._fits[h] = self._estimate(h)
        return self._fits[h]

    def _latest_features(self) -> np.ndarray:
        """The design row at the forecast origin."""
        row = self._design[-1]
        if not np.isfinite(row).all():
            raise NotEnoughData(
                f"{self.name}: cannot form HAR features at the origin "
                f"(needs {MONTHLY_LAG} observations)"
            )
        return row

    # -- forecasting ------------------------------------------------------- #

    def _residual_space_point(self, h: int) -> float:
        fitted = self._fit_for(h)
        return float(self._latest_features() @ fitted.coefficients)

    def _in_sample_residuals(self, h: int) -> np.ndarray:
        return self._fit_for(h).residuals

    def coefficients(self, h: int) -> np.ndarray:
        """Estimated ``[intercept, daily, weekly, monthly]`` for horizon ``h``."""
        return self._fit_for(h).coefficients.copy()


class HAR(_HARBase):
    """HAR estimated in levels (Corsi 2009).

    Included mainly as the foil for Log-HAR: fitted on raw volatility it can and
    does predict negative values in calm regimes, which have to be floored (see
    :attr:`~volmodels.base.Forecaster.n_floored`), and its symmetric Gaussian
    residuals fit a strongly right-skewed target badly.
    """

    def __init__(self, *, quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS) -> None:
        # Levels-space HAR can predict a negative variance, so it carries the
        # standard insanity filter; Log-HAR cannot and therefore does not.
        super().__init__("har", quantile_levels=quantile_levels, insanity_filter=True)


class LogHAR(_HARBase):
    """HAR estimated on ``log(RV)`` -- **the headline baseline**.

    Working in logs gives three things at once: the target is far closer to
    Gaussian (on our AAPL sample, skew falls 1.78 -> 0.58), forecasts are
    positive by construction, and the multiplicative error structure matches how
    volatility actually behaves.

    **Bias correction.**  The regression predicts ``E[log RV]``, but we want
    ``E[RV]``.  By Jensen's inequality ``exp(E[log RV]) < E[RV]``, so naively
    exponentiating gives a systematically *low* forecast.  Under log-normal
    residuals the exact correction is ``E[RV] = exp(mu + sigma^2 / 2)``, so we
    add half the residual variance before exponentiating.

    Note that the correction applies to the point forecast **only**.  Quantiles
    are transformed without it: ``exp`` is monotone, so the alpha-quantile of
    ``log RV`` maps exactly onto the alpha-quantile of ``RV``.  A consequence
    worth expecting in the results: the reported ``q0.5`` (the median) sits
    *below* the point forecast (the mean), which is correct for a right-skewed
    predictive distribution rather than a bug.
    """

    def __init__(
        self,
        *,
        bias_correction: bool = True,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    ) -> None:
        super().__init__("log_har", quantile_levels=quantile_levels)
        self.bias_correction = bias_correction

    def _transform(self, values: np.ndarray) -> np.ndarray:
        """Log transform, masking any non-positive value as missing."""
        safe = np.where(values > 0.0, values, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(safe)

    def _inverse_transform(self, value):
        return np.exp(value)

    def _point_forecast(self, h: int) -> float:
        """Exponentiate the log prediction, with the Jensen bias correction."""
        log_point = self._residual_space_point(h)
        if self.bias_correction:
            log_point += 0.5 * self._fit_for(h).residual_variance
        return float(np.exp(log_point))
