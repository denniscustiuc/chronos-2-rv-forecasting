"""Multiplicative Error Model (Engle 2002; Engle & Gallo 2006) -- optional bonus.

MEM is the natural GARCH analogue for a series that is positive by
construction.  Rather than modelling a *variance* of signed returns, it models
the conditional *mean* of a non-negative series multiplicatively::

    RV_t = mu_t * eps_t,    eps_t iid, E[eps_t] = 1
    mu_t = omega + alpha * RV_{t-1} + beta * mu_{t-1}

The multiplicative error keeps the series positive without a log transform, and
the ``mu`` recursion gives it GARCH-like persistence.  Estimation is by
exponential quasi-maximum-likelihood, which is consistent for the conditional
mean whatever the true error distribution actually is -- the standard reason to
prefer QMLE here over assuming a specific Gamma shape.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from .base import DEFAULT_QUANTILE_LEVELS, EmpiricalQuantileForecaster, NotEnoughData

logger = logging.getLogger(__name__)

__all__ = ["MEM", "AMEM"]

#: Keep alpha + beta strictly below this so the recursion stays stationary and
#: the multi-step forecast converges to a finite long-run mean.
MAX_PERSISTENCE = 0.9995

MIN_OBSERVATIONS = 100


def _recursion_loop(values: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Reference implementation of the ``mu`` recursion, written out literally.

    Kept as the readable definition and as the oracle that
    ``tests/test_forecasters.py::test_mem_recursion_matches_the_reference_loop``
    checks the fast path against.
    """
    mu = np.empty_like(values)
    mu[0] = values.mean()
    for i in range(1, values.size):
        mu[i] = omega + alpha * values[i - 1] + beta * mu[i - 1]
    return mu


def _recursion(values: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Run the ``mu`` recursion over ``values``, seeded at the sample mean.

    ``mu_t = omega + alpha * x_{t-1} + beta * mu_{t-1}`` is a first-order IIR
    filter, so ``lfilter`` evaluates it in C instead of a Python loop.  The
    QMLE optimiser calls this on every function evaluation at every refit, which
    is the difference between a backtest that takes seconds and one that takes
    minutes.
    """
    mu = np.empty_like(values)
    mu[0] = values.mean()
    if values.size > 1:
        driver = omega + alpha * values[:-1]
        mu[1:], _ = lfilter([1.0], [1.0, -beta], driver, zi=[beta * mu[0]])
    return mu


def _negative_qmle(params: np.ndarray, values: np.ndarray) -> float:
    """Negative exponential quasi-log-likelihood, ``sum(log mu + x / mu)``."""
    omega, alpha, beta = params
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= MAX_PERSISTENCE:
        return 1e12
    mu = _recursion(values, omega, alpha, beta)
    if not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
        return 1e12
    return float(np.sum(np.log(mu) + values / mu))


class MEM(EmpiricalQuantileForecaster):
    """Multiplicative Error Model for a realized-volatility series.

    Multi-step forecasts iterate the recursion forward.  Because
    ``E[RV_{t+k}] = E[mu_{t+k}]``, the ``h``-step forecast collapses to the
    closed form

    ``m_h = omega * (1 + s + ... + s^{h-2}) + s^{h-1} * m_1``

    with ``s = alpha + beta`` the persistence and ``m_1 = omega + alpha * RV_t +
    beta * mu_t``.  As ``h`` grows this decays to the unconditional mean
    ``omega / (1 - s)``, which is the behaviour you want from a mean-reverting
    volatility model.

    Attributes:
        omega, alpha, beta: Estimated parameters (``None`` before fitting).
    """

    def __init__(self, *, quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS) -> None:
        super().__init__("mem", quantile_levels=quantile_levels)
        self.omega: float | None = None
        self.alpha: float | None = None
        self.beta: float | None = None
        self._values = np.empty(0)
        self._mu = np.empty(0)

    @property
    def persistence(self) -> float:
        """``alpha + beta`` -- how slowly shocks decay."""
        if self.alpha is None or self.beta is None:
            raise RuntimeError("mem: not fitted")
        return self.alpha + self.beta

    def _rebuild(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Re-run the ``mu`` recursion with the current parameters."""
        clean = self._store_history(history, exog)
        values = clean.to_numpy()
        if np.any(values <= 0.0):
            raise NotEnoughData("mem requires a strictly positive series")
        self._values = values
        if self.omega is not None:
            self._mu = _recursion(values, self.omega, self.alpha, self.beta)

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Estimate ``(omega, alpha, beta)`` by exponential QMLE.

        Raises:
            NotEnoughData: If the history is shorter than
                :data:`MIN_OBSERVATIONS` or non-positive.
        """
        self._rebuild(history, exog)
        values = self._values
        if values.size < MIN_OBSERVATIONS:
            raise NotEnoughData(f"mem needs >= {MIN_OBSERVATIONS} observations, got {values.size}")

        mean = float(values.mean())
        # Start from a typical RV persistence split: most weight on the lagged
        # conditional mean, the rest on the last observation.
        start = np.array([mean * 0.1, 0.30, 0.60])
        bounds = [(1e-10, max(mean, 1.0)), (0.0, 0.999), (0.0, 0.999)]

        result = minimize(
            _negative_qmle, start, args=(values,), method="L-BFGS-B", bounds=bounds,
        )
        if not result.success and not np.all(np.isfinite(result.x)):
            raise NotEnoughData(f"mem: optimiser failed ({result.message})")

        omega, alpha, beta = (float(v) for v in result.x)
        if alpha + beta >= MAX_PERSISTENCE:
            # Nudge back inside the stationary region so multi-step forecasts
            # stay finite; log it, because a pinned boundary is worth knowing.
            scale = (MAX_PERSISTENCE - 1e-6) / (alpha + beta)
            alpha, beta = alpha * scale, beta * scale
            logger.warning("mem: persistence pinned at the stationarity boundary")

        self.omega, self.alpha, self.beta = omega, alpha, beta
        self._mu = _recursion(values, omega, alpha, beta)

    def update(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Refresh the ``mu`` recursion without re-running the optimiser."""
        if self.omega is None:
            self.fit(history, exog)
        else:
            self._rebuild(history, exog)

    def _step_ahead(self, one_step: np.ndarray, h: int) -> np.ndarray:
        """Iterate one-step forecasts out to horizon ``h`` (closed form)."""
        s = self.persistence
        if h == 1:
            return one_step
        geometric = (1.0 - s ** (h - 1)) / (1.0 - s) if s != 1.0 else float(h - 1)
        return self.omega * geometric + s ** (h - 1) * one_step

    def _residual_space_point(self, h: int) -> float:
        if self.omega is None:
            raise RuntimeError("mem: forecast() called before fit()")
        one_step = self.omega + self.alpha * self._values[-1] + self.beta * self._mu[-1]
        return float(self._step_ahead(np.array([one_step]), h)[0])

    def _in_sample_residuals(self, h: int) -> np.ndarray:
        if self.omega is None or self._values.size <= h:
            return np.empty(0)
        one_step = self.omega + self.alpha * self._values[:-1] + self.beta * self._mu[:-1]
        forecasts = self._step_ahead(one_step, h)
        # forecasts[i] targets values[i + h]; drop the tail with no realisation.
        usable = forecasts.size - (h - 1)
        if usable <= 0:
            return np.empty(0)
        return self._values[h:] - forecasts[: self._values.size - h]


class AMEM(MEM):
    """Asymmetric MEM -- **deliberately left as a stub**.

    AMEM extends MEM with a leverage term, typically

    ``mu_t = omega + alpha * RV_{t-1} + gamma * RV_{t-1} * 1[r_{t-1} < 0] + beta * mu_{t-1}``

    so that a negative *return* raises tomorrow's expected volatility by more
    than a positive one of the same size.

    It is not implemented because it needs something this project's forecaster
    interface deliberately does not carry: the **sign of the underlying
    return**.  Every model here consumes an RV series and nothing else, which is
    what keeps the harness data-source-agnostic (a VOLARE RV series and a
    yfinance RV series are interchangeable inputs).  Wiring AMEM up means
    widening the interface to accept exogenous inputs -- a real design decision
    worth making on purpose rather than smuggling in here.

    Raises:
        NotImplementedError: Always.
    """

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "AMEM needs signed returns, which the RV-series-only Forecaster interface "
            "does not carry. Extend the interface with an exogenous-input channel first."
        )
