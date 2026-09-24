"""Configuration objects for the volpipe realized-volatility data pipeline.

Everything the pipeline does is driven by :class:`PipelineConfig`; no module
below this one reads global state or hard-codes tickers, dates or windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

__all__ = [
    "CovariateTag",
    "COVARIATE_TAGS",
    "ESTIMATOR_COLUMN_SUFFIX",
    "DEFAULT_ESTIMATORS",
    "DEFAULT_COVARIATES",
    "DEFAULT_SECTOR_ETF_MAP",
    "PipelineConfig",
]

#: A covariate is either known only in arrears (``past_only``) or scheduled in
#: advance (``known_future``).  Forecasting models may condition on a
#: ``known_future`` covariate at the forecast horizon; a ``past_only`` covariate
#: must be lagged.  This distinction is carried through to the JSON manifest.
CovariateTag = Literal["past_only", "known_future"]

COVARIATE_TAGS: dict[str, CovariateTag] = {
    "vix": "past_only",
    "volume": "past_only",
    "sector_etf_vol": "past_only",
    "risk_free_rate": "past_only",
    # The only known-future covariate: earnings dates are published weeks ahead,
    # so using the scheduled date at forecast time is legitimate, not lookahead.
    "earnings_flag": "known_future",
}

#: Short column suffixes used in the output schema (``rv_cc``, ``var_gk``, ...).
ESTIMATOR_COLUMN_SUFFIX: dict[str, str] = {
    "close_to_close": "cc",
    "parkinson": "parkinson",
    "garman_klass": "gk",
    "rogers_satchell": "rs",
    "intraday_rv": "intraday",
    "yang_zhang": "yz",
}

DEFAULT_ESTIMATORS: tuple[str, ...] = (
    "close_to_close",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang",
)

DEFAULT_COVARIATES: tuple[str, ...] = ("vix", "volume", "earnings_flag")

#: Coarse ticker -> sector ETF mapping used by the optional ``sector_etf_vol``
#: covariate.  Extend as needed; unmapped tickers fall back to
#: :attr:`PipelineConfig.default_sector_etf`.
DEFAULT_SECTOR_ETF_MAP: dict[str, str] = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "JPM": "XLF",
    "GS": "XLF",
    "XOM": "XLE",
    "CVX": "XLE",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Everything needed to reproduce one pipeline run.

    Attributes:
        tickers: Equity symbols to process, e.g. ``("AAPL", "MSFT")``.
        start: Inclusive ISO start date (``YYYY-MM-DD``) for the daily history.
        end: Exclusive ISO end date (``YYYY-MM-DD``), matching yfinance
            semantics.
        rv_window: Length in trading days of the rolling realized-volatility
            window.  The value at date ``t`` uses the window *ending* at ``t``.
        annualization_factor: Trading days per year; variances are scaled by
            this factor and volatilities by its square root.
        estimators: Names from :data:`ESTIMATOR_COLUMN_SUFFIX` to compute.
        covariates: Names from :data:`COVARIATE_TAGS` to attach.
        earnings_window_days: ``k`` in "flag is 1 within +/-k trading days of a
            release".  ``0`` flags only the release day itself.
        intraday_interval: Bar size for ``intraday_rv`` (yfinance intraday
            history is capped at roughly 60 days, so this estimator is expected
            to be NaN over most of a multi-year sample).
        intraday_lookback_days: Calendar days of intraday history to request.
        sector_etf_map: Per-ticker override of the sector ETF proxy.
        default_sector_etf: Fallback ETF when a ticker is unmapped.
        sector_etf_estimator: Estimator used for ``sector_etf_vol``.
        risk_free_series: FRED series id for ``risk_free_rate``.
        include_daily_variance: Also emit the per-day ``var_*`` variance series
            alongside the rolling ``rv_*`` volatility series.
        dropna_rv: Drop leading rows where no estimator has a full window yet.
        output_dir: Directory for parquet/csv/manifest artefacts.
        write_csv: Write a ``.csv`` next to each ``.parquet``.
        max_download_retries: Attempts per yfinance request before giving up.
    """

    tickers: tuple[str, ...]
    start: str
    end: str
    rv_window: int = 21
    annualization_factor: int = 252
    estimators: tuple[str, ...] = DEFAULT_ESTIMATORS
    covariates: tuple[str, ...] = DEFAULT_COVARIATES
    earnings_window_days: int = 1
    intraday_interval: str = "5m"
    intraday_lookback_days: int = 59
    sector_etf_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SECTOR_ETF_MAP))
    default_sector_etf: str = "SPY"
    sector_etf_estimator: str = "garman_klass"
    risk_free_series: str = "DGS3MO"
    include_daily_variance: bool = True
    dropna_rv: bool = True
    output_dir: str = "output"
    write_csv: bool = False
    max_download_retries: int = 3

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("config.tickers must not be empty")
        if self.rv_window < 2:
            raise ValueError("config.rv_window must be >= 2 (sample variance needs 2 points)")
        if self.annualization_factor <= 0:
            raise ValueError("config.annualization_factor must be positive")
        if self.earnings_window_days < 0:
            raise ValueError("config.earnings_window_days must be >= 0")
        if self.start >= self.end:
            raise ValueError(f"config.start ({self.start}) must be before config.end ({self.end})")

        unknown_est = set(self.estimators) - set(ESTIMATOR_COLUMN_SUFFIX)
        if unknown_est:
            raise ValueError(f"unknown estimator(s): {sorted(unknown_est)}")
        if not self.estimators:
            raise ValueError("config.estimators must not be empty")

        unknown_cov = set(self.covariates) - set(COVARIATE_TAGS)
        if unknown_cov:
            raise ValueError(f"unknown covariate(s): {sorted(unknown_cov)}")

        if self.sector_etf_estimator not in ESTIMATOR_COLUMN_SUFFIX:
            raise ValueError(f"unknown sector_etf_estimator: {self.sector_etf_estimator!r}")
        if self.sector_etf_estimator == "intraday_rv":
            raise ValueError("sector_etf_estimator cannot be 'intraday_rv' (insufficient history)")

    def sector_etf_for(self, ticker: str) -> str:
        """Return the sector ETF proxy used for ``ticker``."""
        return self.sector_etf_map.get(ticker.upper(), self.default_sector_etf)

    def covariate_tags(self) -> dict[str, CovariateTag]:
        """Return ``{covariate_name: tag}`` for the configured covariates."""
        return {name: COVARIATE_TAGS[name] for name in self.covariates}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the config, for the run manifest."""
        return {
            "tickers": list(self.tickers),
            "start": self.start,
            "end": self.end,
            "rv_window": self.rv_window,
            "annualization_factor": self.annualization_factor,
            "estimators": list(self.estimators),
            "covariates": list(self.covariates),
            "earnings_window_days": self.earnings_window_days,
            "intraday_interval": self.intraday_interval,
            "intraday_lookback_days": self.intraday_lookback_days,
            "sector_etf_map": dict(self.sector_etf_map),
            "default_sector_etf": self.default_sector_etf,
            "sector_etf_estimator": self.sector_etf_estimator,
            "risk_free_series": self.risk_free_series,
            "include_daily_variance": self.include_daily_variance,
            "dropna_rv": self.dropna_rv,
            "output_dir": self.output_dir,
            "write_csv": self.write_csv,
            "max_download_retries": self.max_download_retries,
        }
