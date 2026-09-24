"""voleval -- walk-forward evaluation for volatility forecasts.

Data-source-agnostic by design: the harness takes a realized-volatility series
and a list of :class:`~volmodels.base.Forecaster` factories, and neither it nor
the metrics know or care whether the series came from the yfinance pipeline or
from VOLARE.

Typical use::

    from volmodels import baseline_factories
    from voleval import BacktestConfig, run_backtest, summarize

    result = run_backtest(series_by_ticker, baseline_factories(), BacktestConfig())
    tables = summarize(result.results)
"""

from __future__ import annotations

from .backtest import (
    RESULT_COLUMNS,
    BacktestConfig,
    BacktestResult,
    run_backtest,
    walk_forward,
)
from .brini import (
    BRINI_EQUITIES,
    BRINI_EQUITY_END,
    BRINI_EQUITY_START,
    BRINI_HORIZONS,
    brini_protocol,
    compare_with_brini,
    format_comparison,
    reference_table,
)
from .metrics import (
    COVID_BOUNDARY,
    LOSS_FUNCTIONS,
    PRIMARY_LOSS,
    absolute_error,
    add_losses,
    coverage_table,
    leaderboard,
    mincer_zarnowitz,
    qlike,
    squared_error,
    subsample_summaries,
    summarize,
)
from .significance import (
    DMResult,
    MCSResult,
    diebold_mariano,
    dm_against_benchmark,
    loss_matrix,
    mcs_inclusion_rates,
    model_confidence_set,
    per_ticker_dm,
)

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "RESULT_COLUMNS",
    "walk_forward",
    "run_backtest",
    "qlike",
    "squared_error",
    "absolute_error",
    "LOSS_FUNCTIONS",
    "PRIMARY_LOSS",
    "add_losses",
    "summarize",
    "coverage_table",
    "leaderboard",
    "mincer_zarnowitz",
    "subsample_summaries",
    "COVID_BOUNDARY",
    "DMResult",
    "MCSResult",
    "loss_matrix",
    "diebold_mariano",
    "dm_against_benchmark",
    "per_ticker_dm",
    "model_confidence_set",
    "mcs_inclusion_rates",
    "BRINI_EQUITIES",
    "BRINI_EQUITY_START",
    "BRINI_EQUITY_END",
    "BRINI_HORIZONS",
    "brini_protocol",
    "compare_with_brini",
    "format_comparison",
    "reference_table",
    "__version__",
]
