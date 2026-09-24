"""Naive floors: the bar any real model has to clear.

These exist to make "our model works" a falsifiable claim.  On a 21-day rolling
RV series the random walk is a deceptively strong opponent at short horizons --
consecutive windows overlap, so ``RV_{t+1}`` shares 20 of its 21 days with
``RV_t`` and the persistence forecast is nearly perfect for reasons that have
nothing to do with forecasting skill.  That artefact dies at ``h = window``,
which is exactly why the backtest reports every horizon separately.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from .base import DEFAULT_QUANTILE_LEVELS, EmpiricalQuantileForecaster, NotEnoughData

logger = logging.getLogger(__name__)

__all__ = ["RandomWalk", "HistoricalMean", "EWMA"]


class RandomWalk(EmpiricalQuantileForecaster):
    """Forecast = the last observed value, flat across all horizons.

    The martingale benchmark.  Its ``h``-step in-sample residuals are
    ``x_{i+h} - x_i``, which widen with ``h`` and give the prediction bands the
    right qualitative shape for free.
    """

    def __init__(self, *, quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS) -> None:
        super().__init__("random_walk", quantile_levels=quantile_levels)
        self._values = np.empty(0)

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Record the history; there are no parameters to estimate."""
        clean = self._store_history(history, exog)
        self._values = clean.to_numpy()

    def _residual_space_point(self, h: int) -> float:
        return float(self._values[-1])

    def _in_sample_residuals(self, h: int) -> np.ndarray:
        if self._values.size <= h:
            return np.empty(0)
        return self._values[h:] - self._values[:-h]


class HistoricalMean(EmpiricalQuantileForecaster):
    """Forecast = the mean of the training window, flat across all horizons.

    The unconditional benchmark.  Beating it requires only that volatility is
    persistent at all; failing to beat it means the model has found nothing.

    Attributes:
        window: Number of most-recent observations to average, or ``None`` for
            the whole expanding history.
    """

    def __init__(
        self,
        window: int | None = None,
        *,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    ) -> None:
        name = "historical_mean" if window is None else f"historical_mean_{window}"
        super().__init__(name, quantile_levels=quantile_levels)
        if window is not None and window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self._values = np.empty(0)

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Record the history; the mean is computed at forecast time."""
        clean = self._store_history(history, exog)
        self._values = clean.to_numpy()

    def _mean_of(self, values: np.ndarray) -> float:
        if self.window is not None:
            values = values[-self.window :]
        return float(values.mean())

    def _residual_space_point(self, h: int) -> float:
        return self._mean_of(self._values)

    def _in_sample_residuals(self, h: int) -> np.ndarray:
        """Residuals against the mean *as it stood at each point in time*.

        Using the expanding (or trailing-window) mean rather than the full-sample
        mean keeps the residual distribution honest: it reflects the error the
        model would actually have made, not the error it makes with hindsight
        about the sample average.
        """
        values = self._values
        if values.size <= h:
            return np.empty(0)

        if self.window is None:
            running = np.cumsum(values) / np.arange(1, values.size + 1)
        else:
            running = pd.Series(values).rolling(self.window, min_periods=1).mean().to_numpy()
        return values[h:] - running[:-h]


class EWMA(EmpiricalQuantileForecaster):
    """Exponentially weighted moving average, flat across all horizons.

    ``level_t = lambda * level_{t-1} + (1 - lambda) * x_t``, forecast
    ``= level_t`` for every ``h``.  This is RiskMetrics' recursion applied
    directly to the realized-volatility series rather than to squared returns,
    which keeps it in the same units as every other model here.

    Attributes:
        lam: Decay factor in ``(0, 1)``.  Higher means longer memory; 0.94 is
            the RiskMetrics daily convention.
    """

    def __init__(
        self,
        lam: float = 0.94,
        *,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    ) -> None:
        if not 0.0 < lam < 1.0:
            raise ValueError(f"lam must lie strictly between 0 and 1, got {lam}")
        super().__init__(f"ewma_{lam:g}", quantile_levels=quantile_levels)
        self.lam = float(lam)
        self._levels = np.empty(0)

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Run the EWMA recursion over the history and cache the level path."""
        clean = self._store_history(history, exog)
        values = clean.to_numpy()
        if values.size < 2:
            raise NotEnoughData("ewma needs at least 2 observations")

        levels = np.empty_like(values)
        levels[0] = values[0]
        one_minus = 1.0 - self.lam
        for i in range(1, values.size):
            levels[i] = self.lam * levels[i - 1] + one_minus * values[i]
        self._levels = levels
        self._values = values

    def _residual_space_point(self, h: int) -> float:
        return float(self._levels[-1])

    def _in_sample_residuals(self, h: int) -> np.ndarray:
        if self._values.size <= h:
            return np.empty(0)
        return self._values[h:] - self._levels[:-h]
