"""ARMA and ARFIMA on log realized volatility.

Both are the time-series-econometrics counterweight to HAR in Brini's benchmark
set.  HAR is a deliberately crude approximation to long memory -- three
overlapping averages standing in for a slowly-decaying autocorrelation
function.  ARFIMA models that long memory directly, through a fractional
differencing parameter ``d``, and ARMA models it not at all.  Running all three
shows how much of HAR's performance is the long-memory structure and how much
is HAR's particular shortcut.

Both estimate on **log volatility**, matching Log-HAR: it keeps forecasts
positive, tames the right skew, and makes the comparison against the headline
benchmark a difference of dynamics rather than of transformation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .base import DEFAULT_QUANTILE_LEVELS, EmpiricalQuantileForecaster, NotEnoughData

logger = logging.getLogger(__name__)

__all__ = ["ARMA", "ARFIMA", "gph_estimate_d", "fractional_difference"]

MIN_OBSERVATIONS = 100


def _log_series(values: np.ndarray, name: str) -> np.ndarray:
    """Log-transform a strictly positive series."""
    if np.any(values <= 0.0):
        raise NotEnoughData(f"{name} requires a strictly positive series")
    return np.log(values)


def _binomial_weights(d: float, lags: int) -> np.ndarray:
    """Binomial expansion weights of ``(1 - L)^d``: ``w_k = w_{k-1} (k-1-d)/k``."""
    weights = np.empty(lags, dtype="float64")
    weights[0] = 1.0
    for k in range(1, lags):
        weights[k] = weights[k - 1] * (k - 1 - d) / k
    return weights


def fractional_difference(values: np.ndarray, d: float, *, truncation: int = 500) -> np.ndarray:
    """Apply the fractional difference operator ``(1 - L)^d``.

    The binomial expansion is infinite; it is truncated at ``truncation`` lags,
    which is ample because the weights decay like ``k^(-d-1)``.  The first
    ``truncation`` outputs are therefore only approximate and callers should
    treat the head of the series with suspicion.

    Args:
        values: Series to difference, oldest first.
        d: Differencing order, typically in ``(0, 0.5)`` for a stationary
            long-memory series.
        truncation: Number of binomial weights to retain.

    Returns:
        The fractionally differenced series, same length as ``values``.
    """
    n = values.size
    lags = min(truncation, n)
    weights = _binomial_weights(d, lags)
    # The operator is a causal FIR filter, so lfilter evaluates it in C.  The
    # naive double loop is O(n * lags) in Python and dominated the runtime of a
    # backtest that re-differences at every origin.
    return lfilter(weights, [1.0], values)


def _fractional_integrate(differenced: np.ndarray, history: np.ndarray, d: float, steps: int,
                          *, truncation: int = 500) -> np.ndarray:
    """Invert :func:`fractional_difference` forward over ``steps`` periods.

    Given the differenced-series forecasts and the observed history, rebuild the
    level forecasts by running the differencing recursion in reverse.
    """
    lags = min(truncation, history.size + steps)
    weights = _binomial_weights(d, lags)

    extended = list(history)
    for step in range(steps):
        # x_t = w_dif_t - sum_{k>=1} w_k * x_{t-k}
        span = min(len(extended), lags - 1)
        tail = np.array(extended[-span:][::-1]) if span else np.empty(0)
        correction = float(weights[1 : span + 1] @ tail) if span else 0.0
        extended.append(float(differenced[step]) - correction)
    return np.array(extended[-steps:])


def gph_estimate_d(values: np.ndarray, *, power: float = 0.5) -> float:
    """Geweke-Porter-Hudak log-periodogram estimate of the memory parameter ``d``.

    Regresses the log periodogram on ``log(4 sin^2(lambda_j / 2))`` over the
    lowest ``n**power`` Fourier frequencies; the slope is ``-d``.  Chosen over
    exact maximum likelihood because it is non-iterative and therefore cheap
    enough to re-estimate at every backtest origin.

    Args:
        values: Series (already log-transformed and demeaned by the caller).
        power: Bandwidth exponent; ``0.5`` is the standard choice.

    Returns:
        ``d`` clipped to ``[0, 0.499]`` so the result stays stationary and
        invertible for the downstream ARMA step.
    """
    n = values.size
    centred = values - values.mean()

    periodogram = np.abs(np.fft.rfft(centred)) ** 2 / (2.0 * np.pi * n)
    n_freq = max(4, int(n**power))
    n_freq = min(n_freq, periodogram.size - 1)

    j = np.arange(1, n_freq + 1)
    lam = 2.0 * np.pi * j / n
    regressor = np.log(4.0 * np.sin(lam / 2.0) ** 2)
    response = np.log(periodogram[1 : n_freq + 1])

    design = np.column_stack([np.ones(n_freq), regressor])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    raw = -float(coefficients[1])
    if raw >= 0.5:
        # d >= 0.5 means the series is non-stationary, not merely long-memory.
        # Clipping keeps the downstream ARMA well defined, but it is a real
        # signal about the data rather than a formality, so say so.
        logger.debug("gph: d estimated at %.3f, clipped to the stationarity bound 0.499", raw)
    return float(np.clip(raw, 0.0, 0.499))


class _StatsmodelsBase(EmpiricalQuantileForecaster):
    """Shared lifecycle for the statsmodels-backed models.

    Parameters are estimated in :meth:`fit` and deliberately survive
    :meth:`update`, which only refreshes the conditioning data -- the same
    contract HAR uses, and what makes ``refit_frequency`` meaningful for models
    whose estimation is an iterative MLE rather than an OLS.
    """

    def __init__(self, name: str, *, quantile_levels: Sequence[float]) -> None:
        super().__init__(name, quantile_levels=quantile_levels)
        self._log_values = np.empty(0)
        self._params = None
        self._residuals = np.empty(0)
        self._path: np.ndarray | None = None

    def _inverse_transform(self, value):
        return np.exp(value)

    def _rebuild(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        clean = self._store_history(history, exog)
        self._log_values = _log_series(clean.to_numpy(), self.name)
        self._path = None  # the conditioning data moved, so the cache is stale

    def _compute_path(self, max_h: int) -> np.ndarray:
        """Forecast ``max_h`` steps ahead in the estimation space."""
        raise NotImplementedError

    def _residual_space_point(self, h: int) -> float:
        """Return step ``h`` of the cached forecast path.

        The path is computed once per origin and reused across horizons: the
        expensive part (filtering the series through the fitted model, and for
        ARFIMA the fractional differencing) does not depend on ``h``, so doing
        it per horizon tripled the cost of every backtest origin for nothing.
        """
        if self._params is None:
            raise RuntimeError(f"{self.name}: forecast() called before fit()")
        if self._path is None or self._path.size < h:
            self._path = self._compute_path(h)
        return float(self._path[h - 1])

    def update(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Refresh the conditioning data without re-running the MLE."""
        if self._params is None:
            self.fit(history, exog)
        else:
            self._rebuild(history, exog)

    def _in_sample_residuals(self, h: int) -> np.ndarray:
        """Scale the one-step residual spread by ``sqrt(h)``.

        A closed-form ``h``-step predictive variance exists for these models,
        but the empirical-residual convention used across this project keeps the
        baselines' bands comparable to each other; the random-walk scaling is the
        honest simple approximation and is documented as such.
        """
        if self._residuals.size == 0:
            return np.empty(0)
        return self._residuals * np.sqrt(float(h))


class ARMA(_StatsmodelsBase):
    """ARMA(p, q) on log volatility.

    Short-memory benchmark: its autocorrelation decays geometrically, so it
    cannot reproduce the slow hyperbolic decay that realized volatility actually
    shows.  Included precisely to quantify what that costs.

    Attributes:
        order: ``(p, q)``.
    """

    def __init__(
        self,
        order: tuple[int, int] = (1, 1),
        *,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    ) -> None:
        super().__init__("arma", quantile_levels=quantile_levels)
        self.order = order
        self._model = None

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Estimate the ARMA parameters by maximum likelihood."""
        from statsmodels.tsa.arima.model import ARIMA

        self._rebuild(history, exog)
        if self._log_values.size < MIN_OBSERVATIONS:
            raise NotEnoughData(f"arma needs >= {MIN_OBSERVATIONS} observations")

        p, q = self.order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitted = ARIMA(self._log_values, order=(p, 0, q)).fit()
            except Exception as exc:
                raise NotEnoughData(f"arma: estimation failed ({exc})") from exc

        self._params = fitted.params
        self._model = fitted
        self._residuals = np.asarray(fitted.resid, dtype="float64")

    def _compute_path(self, max_h: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Re-apply the *stored* parameters to the current history: this is
            # what lets update() move the conditioning window forward without
            # paying for another MLE.
            applied = ARIMA(self._log_values, order=(self.order[0], 0, self.order[1])).filter(
                self._params
            )
            return np.asarray(applied.forecast(steps=max_h), dtype="float64")


class ARFIMA(_StatsmodelsBase):
    """ARFIMA(p, d, q) on log volatility, with ``d`` estimated by GPH.

    Three steps: estimate ``d`` from the log periodogram, fractionally difference
    the series to leave a short-memory remainder, fit an ARMA to that and
    forecast it, then fractionally integrate the forecast back to levels.

    This is the specification that takes long memory literally, and is the
    natural econometric rival to HAR -- HAR's three-component cascade is best
    understood as a cheap approximation to exactly this.

    Attributes:
        order: ``(p, q)`` of the ARMA applied after differencing.
        d: The estimated memory parameter (``None`` before fitting).
    """

    def __init__(
        self,
        order: tuple[int, int] = (1, 1),
        *,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    ) -> None:
        super().__init__("arfima", quantile_levels=quantile_levels)
        self.order = order
        self.d: float | None = None
        self._mean = 0.0

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Estimate ``d`` by GPH, then the ARMA on the differenced series."""
        from statsmodels.tsa.arima.model import ARIMA

        self._rebuild(history, exog)
        if self._log_values.size < MIN_OBSERVATIONS:
            raise NotEnoughData(f"arfima needs >= {MIN_OBSERVATIONS} observations")

        self.d = gph_estimate_d(self._log_values)
        if self.d >= 0.498:
            logger.debug(
                "arfima: memory parameter pinned at the stationarity bound; "
                "the log-volatility series looks near-integrated over this window"
            )
        self._mean = float(self._log_values.mean())
        differenced = fractional_difference(self._log_values - self._mean, self.d)

        p, q = self.order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitted = ARIMA(differenced, order=(p, 0, q)).fit()
            except Exception as exc:
                raise NotEnoughData(f"arfima: estimation failed ({exc})") from exc

        self._params = fitted.params
        self._residuals = np.asarray(fitted.resid, dtype="float64")

    def _compute_path(self, max_h: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA

        centred = self._log_values - self._mean
        differenced = fractional_difference(centred, self.d)

        p, q = self.order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = ARIMA(differenced, order=(p, 0, q)).filter(self._params)
            forecast_differenced = np.asarray(applied.forecast(steps=max_h), dtype="float64")

        levels = _fractional_integrate(forecast_differenced, centred, self.d, max_h)
        return levels + self._mean
