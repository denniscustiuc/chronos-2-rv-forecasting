"""Covariate construction, aligned to a ticker's trading-date index.

Each covariate carries a tag:

``past_only``
    Observable only in arrears.  The value at date ``t`` is what was known by
    the close of ``t`` -- nothing later leaks in.  A forecasting layer that
    predicts ``t+h`` must lag these itself; this module deliberately does *not*
    apply a model-specific lag, it only guarantees no lookahead at ``t``.

``known_future``
    Scheduled in advance, so its value at a future date is legitimately known
    today.  ``earnings_flag`` is the only one here, and it is the sole place in
    the pipeline where a future-dated input is used on purpose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from .config import COVARIATE_TAGS, PipelineConfig
from .estimators import compute_estimator
from .ingest import download_series_close, load_ticker_ohlcv, normalize_index

logger = logging.getLogger(__name__)

__all__ = [
    "CovariateBundle",
    "vix_covariate",
    "volume_covariate",
    "earnings_flag_covariate",
    "sector_etf_vol_covariate",
    "risk_free_rate_covariate",
    "build_covariates",
]

VIX_SYMBOL = "^VIX"


@dataclass
class CovariateBundle:
    """Covariate columns plus the metadata the manifest needs.

    Attributes:
        frame: Covariate columns indexed by trading date.
        tags: ``{column: "past_only" | "known_future"}``.
        notes: Human-readable notes (fallbacks taken, series unavailable, ...).
    """

    frame: pd.DataFrame
    tags: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, series: pd.Series, tag: str, *, note: str | None = None) -> None:
        """Attach ``series`` as a column with the given tag."""
        self.frame[series.name] = series
        self.tags[str(series.name)] = tag
        if note:
            self.notes.append(note)


def _align(series: pd.Series, index: pd.DatetimeIndex, *, name: str) -> pd.Series:
    """Reindex an external daily series onto ``index``, forward-filling gaps.

    Forward fill carries the last *observed* value forward, which is past-only
    by construction.  Backward fill is never used: it would pull a future
    observation back to a date on which it was unknown.
    """
    if series.empty:
        return pd.Series(np.nan, index=index, name=name, dtype="float64")
    aligned = series.copy()
    aligned.index = normalize_index(aligned.index)
    aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()
    return aligned.reindex(index).ffill().rename(name).astype("float64")


def vix_covariate(index: pd.DatetimeIndex, config: PipelineConfig) -> pd.Series:
    """CBOE VIX daily close (``^VIX``), aligned to ``index``.  [past_only]

    The VIX close for date ``t`` is published at that session's close, so using
    it at ``t`` alongside an RV window ending at ``t`` introduces no lookahead.
    """
    raw = download_series_close(VIX_SYMBOL, config.start, config.end, retries=config.max_download_retries)
    if raw.empty:
        logger.warning("vix: download returned nothing; column will be all-NaN")
    aligned = _align(raw, index, name="vix")
    missing = int(aligned.isna().sum())
    if missing:
        logger.info("vix: %d/%d dates unfilled (leading gap before first VIX print)", missing, len(index))
    return aligned


def volume_covariate(ohlcv: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """The ticker's own daily share volume.  [past_only]"""
    if "volume" not in ohlcv.columns:
        return pd.Series(np.nan, index=index, name="volume", dtype="float64")
    return ohlcv["volume"].reindex(index).rename("volume").astype("float64")


def _fetch_earnings_dates(ticker: str, limit: int) -> pd.DatetimeIndex:
    """Fetch scheduled + historical earnings timestamps, normalised to dates."""
    try:
        frame = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception as exc:  # yfinance surfaces network/parse errors freely
        logger.warning("%s: earnings dates unavailable (%s)", ticker, exc)
        return pd.DatetimeIndex([])
    if frame is None or len(frame) == 0:
        logger.warning("%s: no earnings dates returned", ticker)
        return pd.DatetimeIndex([])
    return normalize_index(frame.index).unique().sort_values()


def earnings_flag_covariate(
    ticker: str,
    index: pd.DatetimeIndex,
    config: PipelineConfig,
    *,
    limit: int = 80,
) -> pd.Series:
    """Binary flag for proximity to an earnings release.  [KNOWN_FUTURE]

    The flag is ``1`` on the anchor trading day of a release and on the
    ``+/- earnings_window_days`` trading days around it, ``0`` otherwise.

    The anchor is the first trading day *on or after* the release date, which
    handles the common after-the-close announcement (its market impact lands on
    the following session) and releases that fall on a weekend or holiday.

    This is the pipeline's one deliberate use of forward-looking information:
    an earnings date is published well in advance, so at forecast time the
    schedule for the horizon is genuinely known.  It is tagged
    ``known_future`` so a modelling layer can treat it accordingly and never
    confuse it with a past-only feature.

    Args:
        ticker: Symbol whose earnings calendar to use.
        index: Trading-date index to align to.
        config: Supplies ``earnings_window_days`` (``k``).
        limit: Number of earnings rows to request from yfinance.

    Returns:
        Int8 series named ``earnings_flag``.
    """
    flag = pd.Series(0, index=index, dtype="int8", name="earnings_flag")
    releases = _fetch_earnings_dates(ticker, limit)
    if len(releases) == 0:
        return flag

    k = config.earnings_window_days
    # Position of the first trading day on or after each release date.
    positions = index.searchsorted(releases, side="left")
    hits = 0
    for pos in positions:
        if pos >= len(index):
            continue  # release scheduled past the end of our sample
        lo = max(0, int(pos) - k)
        hi = min(len(index) - 1, int(pos) + k)
        flag.iloc[lo : hi + 1] = 1
        hits += 1

    logger.info(
        "%s: earnings_flag set from %d/%d release date(s) in range, k=%d (%d flagged days)",
        ticker, hits, len(releases), k, int(flag.sum()),
    )
    return flag


def sector_etf_vol_covariate(
    ticker: str,
    index: pd.DatetimeIndex,
    config: PipelineConfig,
) -> pd.Series:
    """Realized volatility of the ticker's sector ETF.  [past_only]

    Computed with the very same estimator functions used for the ticker itself
    (``config.sector_etf_estimator``, same window and annualisation), so the
    covariate is on the same scale as the targets.
    """
    etf = config.sector_etf_for(ticker)
    name = "sector_etf_vol"
    ohlcv, report = load_ticker_ohlcv(etf, config)
    if ohlcv.empty:
        logger.warning("%s: sector ETF %s returned no data; column will be all-NaN", ticker, etf)
        return pd.Series(np.nan, index=index, name=name, dtype="float64")

    result = compute_estimator(
        config.sector_etf_estimator,
        ohlcv,
        window=config.rv_window,
        annualization_factor=config.annualization_factor,
    )
    logger.info("%s: sector_etf_vol from %s using %s", ticker, etf, config.sector_etf_estimator)
    return _align(result.realized_vol, index, name=name)


def risk_free_rate_covariate(index: pd.DatetimeIndex, config: PipelineConfig) -> pd.Series:
    """FRED short-rate series (default ``DGS3MO``), in percent.  [past_only, OPTIONAL]

    Requires ``pandas-datareader``.  If the package is missing or FRED is
    unreachable the column is emitted as all-``NaN`` rather than failing the
    run -- this covariate is explicitly optional.
    """
    name = "risk_free_rate"
    empty = pd.Series(np.nan, index=index, name=name, dtype="float64")
    try:
        from pandas_datareader import data as pdr  # imported lazily: optional dependency
    except ImportError:
        logger.warning("risk_free_rate: pandas-datareader not installed; column will be all-NaN")
        return empty

    try:
        frame = pdr.DataReader(config.risk_free_series, "fred", config.start, config.end)
    except Exception as exc:
        logger.warning("risk_free_rate: FRED fetch failed (%s); column will be all-NaN", exc)
        return empty

    if frame is None or frame.empty:
        return empty
    return _align(frame.iloc[:, 0], index, name=name)


def build_covariates(
    ticker: str,
    ohlcv: pd.DataFrame,
    index: pd.DatetimeIndex,
    config: PipelineConfig,
) -> CovariateBundle:
    """Build every covariate named in ``config.covariates`` for one ticker.

    Args:
        ticker: Symbol being processed.
        ohlcv: The ticker's cleaned OHLCV (source of ``volume``).
        index: Trading-date index every covariate is aligned to.
        config: Pipeline configuration.

    Returns:
        A :class:`CovariateBundle` whose frame shares ``index``.
    """
    bundle = CovariateBundle(frame=pd.DataFrame(index=index))

    for name in config.covariates:
        tag = COVARIATE_TAGS[name]
        if name == "vix":
            bundle.add(vix_covariate(index, config), tag)
        elif name == "volume":
            bundle.add(volume_covariate(ohlcv, index), tag)
        elif name == "earnings_flag":
            bundle.add(earnings_flag_covariate(ticker, index, config), tag)
        elif name == "sector_etf_vol":
            series = sector_etf_vol_covariate(ticker, index, config)
            bundle.add(series, tag, note=f"sector_etf_vol proxy for {ticker}: {config.sector_etf_for(ticker)}")
        elif name == "risk_free_rate":
            bundle.add(risk_free_rate_covariate(index, config), tag)
        else:  # pragma: no cover - guarded by PipelineConfig validation
            raise ValueError(f"unknown covariate: {name!r}")

        column = bundle.frame.columns[-1]
        na = int(bundle.frame[column].isna().sum())
        if na:
            bundle.notes.append(f"{column}: {na}/{len(index)} missing values")

    return bundle
