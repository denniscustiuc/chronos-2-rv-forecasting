"""volmodels -- forecasting models behind one shared interface.

Every model implements :class:`~volmodels.base.Forecaster`, so the evaluation
harness in :mod:`voleval` drives all of them identically.  Adding Chronos-2 or
Kronos later means adding a subclass here and appending it to a factory list --
no change to the backtest, the metrics or the significance tests.

Models are registered as *factories* rather than instances so every ticker in a
backtest gets a fresh model and no state leaks between assets::

    from volmodels import baseline_factories
    from voleval import run_backtest

    result = run_backtest({"AAPL": rv_series}, baseline_factories())
"""

from __future__ import annotations

from typing import Callable, Sequence

from .base import (
    DEFAULT_QUANTILE_LEVELS,
    EmpiricalQuantileForecaster,
    Forecast,
    Forecaster,
    NotEnoughData,
    parse_quantile_column,
    quantile_column,
)
from .chronos2 import (
    DEFAULT_CHECKPOINT,
    Chronos2Covariate,
    Chronos2Multivariate,
    Chronos2Univariate,
    earnings_countdown,
)
from .classical import ARFIMA, ARMA
from .har import HAR, LogHAR, har_features
from .har_family import EXOG_COLUMNS, HARJ, HARQ, HARRS
from .mem import AMEM, MEM
from .naive import EWMA, HistoricalMean, RandomWalk

__version__ = "0.1.0"

#: The headline benchmark every other model is measured against.
BENCHMARK_MODEL = "log_har"

__all__ = [
    "Forecaster",
    "Forecast",
    "EmpiricalQuantileForecaster",
    "NotEnoughData",
    "DEFAULT_QUANTILE_LEVELS",
    "quantile_column",
    "parse_quantile_column",
    "RandomWalk",
    "HistoricalMean",
    "EWMA",
    "HAR",
    "LogHAR",
    "har_features",
    "HARJ",
    "HARRS",
    "HARQ",
    "EXOG_COLUMNS",
    "ARMA",
    "ARFIMA",
    "Chronos2Univariate",
    "Chronos2Covariate",
    "Chronos2Multivariate",
    "earnings_countdown",
    "DEFAULT_CHECKPOINT",
    "MEM",
    "AMEM",
    "BENCHMARK_MODEL",
    "baseline_factories",
    "brini_factories",
    "__version__",
]


def baseline_factories(
    *,
    include_mem: bool = False,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> list[Callable[[], Forecaster]]:
    """Return factories for the standard baseline suite.

    The three naive floors come first so any leaderboard shows immediately
    whether the fitted models are earning their keep, then HAR in levels, then
    Log-HAR (the benchmark).

    Args:
        include_mem: Also register the Multiplicative Error Model.  It is off by
            default because its QMLE refit is an order of magnitude slower than
            an OLS refit.
        quantile_levels: Quantile grid every model reports on.

    Returns:
        Zero-argument callables, each producing a fresh forecaster.
    """
    factories: list[Callable[[], Forecaster]] = [
        lambda: RandomWalk(quantile_levels=quantile_levels),
        lambda: HistoricalMean(quantile_levels=quantile_levels),
        lambda: EWMA(quantile_levels=quantile_levels),
        lambda: HAR(quantile_levels=quantile_levels),
        lambda: LogHAR(quantile_levels=quantile_levels),
    ]
    if include_mem:
        factories.append(lambda: MEM(quantile_levels=quantile_levels))
    return factories


def brini_factories(
    *,
    include_exog_models: bool = True,
    include_slow: bool = True,
    include_mem: bool = True,
    include_naive: bool = False,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> list[Callable[[], Forecaster]]:
    """Return factories for Brini (2026)'s eight econometric benchmarks.

    The paper's set is HAR, Log-HAR, HAR-J, HAR-RS, HARQ, ARFIMA, ARMA and MEM.
    All eight are implemented here; none of Brini's econometric benchmarks is
    omitted.

    Args:
        include_exog_models: Register HAR-J / HAR-RS / HARQ.  These need the
            companion realized measures (jump, semivariances, quarticity) that
            only a 5-minute dataset such as VOLARE supplies -- on the yfinance
            path they would fail every fit, so set ``False`` there.
        include_slow: Register ARMA and ARFIMA.  Both run an iterative MLE at
            every refit, which dominates the runtime of a large backtest.
        include_mem: Register MEM.  Its QMLE is an order of magnitude cheaper
            than ARMA/ARFIMA but still far above the HAR family's OLS, so it is
            separately switchable for quick HAR-only passes.
        include_naive: Also register the naive floors.  Brini does not report
            them; they are useful as a sanity reference in our own runs.
        quantile_levels: Quantile grid every model reports on.

    Returns:
        Zero-argument callables, each producing a fresh forecaster.
    """
    factories: list[Callable[[], Forecaster]] = []
    if include_naive:
        factories += [
            lambda: RandomWalk(quantile_levels=quantile_levels),
            lambda: HistoricalMean(quantile_levels=quantile_levels),
        ]

    factories += [
        lambda: HAR(quantile_levels=quantile_levels),
        lambda: LogHAR(quantile_levels=quantile_levels),
    ]
    if include_mem:
        factories.append(lambda: MEM(quantile_levels=quantile_levels))
    if include_exog_models:
        factories += [
            lambda: HARJ(quantile_levels=quantile_levels),
            lambda: HARRS(quantile_levels=quantile_levels),
            lambda: HARQ(quantile_levels=quantile_levels),
        ]
    if include_slow:
        factories += [
            lambda: ARMA(quantile_levels=quantile_levels),
            lambda: ARFIMA(quantile_levels=quantile_levels),
        ]
    return factories
