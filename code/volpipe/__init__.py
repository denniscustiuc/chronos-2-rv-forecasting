"""volpipe -- realized-volatility data pipeline (yfinance path).

Produces, per ticker, a clean daily time series of realized-volatility
estimates computed several ways plus aligned covariates, ready to feed a
forecasting layer.  No forecasting lives here.

Typical use::

    from volpipe import PipelineConfig, run_pipeline

    config = PipelineConfig(tickers=("AAPL", "MSFT"), start="2020-01-01", end="2025-01-01")
    result = run_pipeline(config)
    result.tickers["AAPL"].frame.tail()
"""

from .config import (
    COVARIATE_TAGS,
    DEFAULT_COVARIATES,
    DEFAULT_ESTIMATORS,
    ESTIMATOR_COLUMN_SUFFIX,
    PipelineConfig,
)
from .estimators import (
    EstimatorResult,
    close_to_close_variance,
    compute_estimator,
    garman_klass_variance,
    intraday_rv_variance,
    parkinson_variance,
    rogers_satchell_variance,
    rolling_variance,
    to_realized_vol,
    yang_zhang_variance,
)
from .covariates import build_covariates
from .ingest import DataQualityReport, clean_ohlcv, load_ticker_ohlcv
from .pipeline import PipelineResult, TickerResult, build_ticker_frame, run_pipeline
from .validation import (
    compare_with_volare,
    estimator_correlation_matrix,
    estimator_noise_ranking,
    plot_estimators,
)

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "COVARIATE_TAGS",
    "DEFAULT_COVARIATES",
    "DEFAULT_ESTIMATORS",
    "ESTIMATOR_COLUMN_SUFFIX",
    "EstimatorResult",
    "close_to_close_variance",
    "parkinson_variance",
    "garman_klass_variance",
    "rogers_satchell_variance",
    "yang_zhang_variance",
    "intraday_rv_variance",
    "rolling_variance",
    "to_realized_vol",
    "compute_estimator",
    "build_covariates",
    "DataQualityReport",
    "clean_ohlcv",
    "load_ticker_ohlcv",
    "build_ticker_frame",
    "run_pipeline",
    "PipelineResult",
    "TickerResult",
    "estimator_correlation_matrix",
    "estimator_noise_ranking",
    "plot_estimators",
    "compare_with_volare",
    "__version__",
]
