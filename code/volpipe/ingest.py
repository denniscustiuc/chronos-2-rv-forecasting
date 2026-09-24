"""yfinance ingestion and OHLCV cleaning.

Everything is pulled with ``auto_adjust=True`` so that open, high, low and
close are adjusted by the *same* split/dividend factors and stay internally
consistent -- an estimator like Garman-Klass compares ``H/L`` against ``C/O``
within one bar, so mixing an unadjusted high with an adjusted close would
corrupt the estimate outright.

Note for VOLARE comparisons: VOLARE's realized measures are computed from
UNADJUSTED prices.  Our adjusted-price estimates therefore will not reproduce
VOLARE's numbers exactly; the levels track closely but diverge around splits
and ex-dividend dates.  This is expected and not a bug.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from .config import PipelineConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DataQualityReport",
    "normalize_index",
    "download_daily_ohlcv",
    "clean_ohlcv",
    "load_ticker_ohlcv",
    "download_intraday",
    "download_series_close",
]

_OHLCV_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


@dataclass
class DataQualityReport:
    """Record of what ingestion dropped or repaired, for the run manifest.

    Attributes:
        ticker: Symbol the report refers to.
        rows_downloaded: Rows returned by yfinance before cleaning.
        rows_kept: Rows surviving cleaning.
        dropped: ``{reason: count}`` for every removed row.
        dropped_dates: Sample of removed dates (ISO strings, capped).
        warnings: Free-form notes worth surfacing to the user.
    """

    ticker: str
    rows_downloaded: int = 0
    rows_kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    dropped_dates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    _MAX_LOGGED_DATES = 25

    def record_drop(self, reason: str, dates: pd.Index) -> None:
        """Note ``len(dates)`` rows removed for ``reason``."""
        count = int(len(dates))
        if count == 0:
            return
        self.dropped[reason] = self.dropped.get(reason, 0) + count
        room = self._MAX_LOGGED_DATES - len(self.dropped_dates)
        if room > 0:
            self.dropped_dates.extend(d.strftime("%Y-%m-%d") for d in dates[:room])
        logger.warning(
            "%s: dropped %d row(s) [%s]; first: %s",
            self.ticker,
            count,
            reason,
            dates[0].strftime("%Y-%m-%d") if count else "-",
        )

    @property
    def total_dropped(self) -> int:
        """Total rows removed across all reasons."""
        return sum(self.dropped.values())

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for the manifest."""
        return {
            "ticker": self.ticker,
            "rows_downloaded": self.rows_downloaded,
            "rows_kept": self.rows_kept,
            "rows_dropped": self.total_dropped,
            "dropped_by_reason": dict(self.dropped),
            "dropped_dates_sample": list(self.dropped_dates),
            "warnings": list(self.warnings),
        }


def normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    """Coerce any yfinance index to tz-naive, midnight-normalised dates.

    yfinance hands back tz-aware timestamps for some symbols and tz-naive ones
    for others; aligning a ticker against ``^VIX`` fails unless both are
    reduced to the same representation first.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def _retry(fn, attempts: int, label: str):
    """Call ``fn`` up to ``attempts`` times with linear backoff."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # yfinance raises a wide variety of network errors
            last_error = exc
            logger.warning("%s: attempt %d/%d failed: %s", label, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(1.5 * attempt)
    logger.error("%s: giving up after %d attempts (%s)", label, attempts, last_error)
    return None


def download_daily_ohlcv(
    ticker: str,
    start: str,
    end: str,
    *,
    retries: int = 3,
) -> pd.DataFrame:
    """Download adjusted daily OHLCV for one ticker.

    Args:
        ticker: Symbol to fetch.
        start: Inclusive ISO start date.
        end: Exclusive ISO end date.
        retries: Attempts before returning an empty frame.

    Returns:
        Frame indexed by tz-naive trading date with lowercase ``open``,
        ``high``, ``low``, ``close``, ``volume`` columns.  Empty if the
        download failed.
    """
    def _pull() -> pd.DataFrame:
        return yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            # Adjusted consistently across O/H/L/C -- see module docstring.
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
            threads=False,
        )

    raw = _retry(_pull, retries, f"download {ticker}")
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = raw.rename(columns=_OHLCV_RENAME)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = normalize_index(frame.index)
    frame.index.name = "date"

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
    return frame[keep].sort_index()


def clean_ohlcv(frame: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    """Drop unusable OHLCV rows, recording every removal in ``report``.

    Removes, in order: duplicate dates (keeping the last), rows with any missing
    OHLC price, rows with a non-positive price, and rows that violate
    ``low <= min(open, close) <= max(open, close) <= high``.  Holidays and
    weekends simply never appear in a yfinance daily frame, so gaps in the
    calendar need no special handling -- the index *is* the trading calendar.

    Args:
        frame: Raw OHLCV frame from :func:`download_daily_ohlcv`.
        report: Mutated in place with the drop counts.

    Returns:
        The cleaned frame.
    """
    report.rows_downloaded = int(len(frame))
    if frame.empty:
        report.warnings.append("no rows returned by yfinance")
        return frame

    out = frame.sort_index()

    duplicated = out.index.duplicated(keep="last")
    if duplicated.any():
        report.record_drop("duplicate_date", out.index[duplicated])
        out = out[~duplicated]

    price_cols = ["open", "high", "low", "close"]
    missing = out[price_cols].isna().any(axis=1)
    if missing.any():
        report.record_drop("missing_ohlc", out.index[missing])
        out = out[~missing]

    nonpositive = (out[price_cols] <= 0).any(axis=1)
    if nonpositive.any():
        report.record_drop("nonpositive_price", out.index[nonpositive])
        out = out[~nonpositive]

    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)
    inconsistent = (out["high"] < body_high) | (out["low"] > body_low) | (out["high"] < out["low"])
    if inconsistent.any():
        report.record_drop("ohlc_inconsistent", out.index[inconsistent])
        out = out[~inconsistent]

    if "volume" in out.columns:
        # A zero/NaN volume bar is a real (if illiquid) session: keep the prices,
        # just mark the volume missing so downstream code does not read 0 as a fact.
        bad_volume = out["volume"].isna() | (out["volume"] <= 0)
        if bad_volume.any():
            out["volume"] = out["volume"].where(~bad_volume)
            report.warnings.append(f"{int(bad_volume.sum())} row(s) with zero/missing volume set to NaN")

    report.rows_kept = int(len(out))
    return out


def load_ticker_ohlcv(
    ticker: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Download and clean daily OHLCV for ``ticker`` per ``config``.

    Returns:
        ``(cleaned_frame, report)``.
    """
    report = DataQualityReport(ticker=ticker)
    raw = download_daily_ohlcv(
        ticker, config.start, config.end, retries=config.max_download_retries
    )
    cleaned = clean_ohlcv(raw, report)
    logger.info(
        "%s: %d/%d daily rows kept (%s -> %s)",
        ticker,
        report.rows_kept,
        report.rows_downloaded,
        cleaned.index.min().date() if not cleaned.empty else "-",
        cleaned.index.max().date() if not cleaned.empty else "-",
    )
    return cleaned, report


def download_intraday(
    ticker: str,
    *,
    interval: str = "5m",
    lookback_days: int = 59,
    retries: int = 2,
) -> pd.DataFrame | None:
    """Best-effort download of recent intraday bars.

    yfinance caps intraday history at roughly 60 days for 5-minute bars, so this
    covers only the tail of a multi-year sample.  Any failure returns ``None``
    and the caller degrades to an all-``NaN`` intraday RV series.

    Args:
        ticker: Symbol to fetch.
        interval: Bar size, e.g. ``"5m"``.
        lookback_days: Calendar days of history to request.
        retries: Attempts before giving up.

    Returns:
        Frame indexed by timestamp with a ``close`` column, or ``None``.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    def _pull() -> pd.DataFrame:
        return yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
            threads=False,
        )

    raw = _retry(_pull, retries, f"intraday {ticker} @{interval}")
    if raw is None or raw.empty:
        logger.info("%s: no intraday data for interval %s", ticker, interval)
        return None

    frame = raw.rename(columns=_OHLCV_RENAME)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if "close" not in frame.columns:
        return None
    frame = frame[["close"]].dropna()
    return frame if not frame.empty else None


def download_series_close(
    symbol: str,
    start: str,
    end: str,
    *,
    retries: int = 3,
) -> pd.Series:
    """Download one symbol's adjusted daily close as a named Series.

    Used for ``^VIX`` and the sector-ETF covariates.  Returns an empty Series
    when the download fails.
    """
    frame = download_daily_ohlcv(symbol, start, end, retries=retries)
    if frame.empty or "close" not in frame.columns:
        return pd.Series(dtype="float64", name=symbol)
    return frame["close"].rename(symbol).astype("float64")
