"""End-to-end orchestration: ingest -> estimators -> covariates -> panel + manifest.

The output for one ticker is a tidy frame indexed by trading date::

    date | rv_cc rv_parkinson rv_gk rv_rs [rv_yz] [rv_intraday]
         | var_cc var_parkinson var_gk var_rs [var_yz] [var_intraday]
         | vix volume earnings_flag [sector_etf_vol] [risk_free_rate]

``rv_*`` is annualised realized volatility over the rolling window ending at
that date; ``var_*`` is the raw per-day variance estimate in daily units.  Every
column is aligned to the same index and free of lookahead, with the single
deliberate exception of ``earnings_flag`` (tagged ``known_future``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ESTIMATOR_COLUMN_SUFFIX, PipelineConfig
from .covariates import build_covariates
from .estimators import compute_estimator
from .ingest import DataQualityReport, download_intraday, load_ticker_ohlcv

logger = logging.getLogger(__name__)

__all__ = ["TickerResult", "PipelineResult", "build_ticker_frame", "run_pipeline"]


@dataclass
class TickerResult:
    """Per-ticker pipeline output.

    Attributes:
        ticker: Symbol.
        frame: The tidy output frame (may be empty if ingestion failed).
        quality: Ingestion drop report.
        covariate_tags: ``{column: tag}`` for the covariate columns.
        estimator_columns: ``{estimator_name: rv_column}``.
        notes: Free-form notes (unavailable estimators, covariate fallbacks).
    """

    ticker: str
    frame: pd.DataFrame
    quality: DataQualityReport
    covariate_tags: dict[str, str] = field(default_factory=dict)
    estimator_columns: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def rv_columns(self) -> list[str]:
        """The ``rv_*`` columns present in :attr:`frame`."""
        return [c for c in self.frame.columns if c.startswith("rv_")]

    def summary(self) -> dict[str, Any]:
        """Row counts, date range and per-column coverage, for the manifest."""
        frame = self.frame
        return {
            "ticker": self.ticker,
            "rows": int(len(frame)),
            "start": frame.index.min().strftime("%Y-%m-%d") if not frame.empty else None,
            "end": frame.index.max().strftime("%Y-%m-%d") if not frame.empty else None,
            "columns": list(frame.columns),
            "estimator_columns": dict(self.estimator_columns),
            "covariate_tags": dict(self.covariate_tags),
            "non_null_counts": {c: int(frame[c].notna().sum()) for c in frame.columns},
            "data_quality": self.quality.to_dict(),
            "notes": list(self.notes),
        }


@dataclass
class PipelineResult:
    """Everything one run produced.

    Attributes:
        config: The config that produced this run.
        tickers: ``{ticker: TickerResult}``.
        panel: Combined multi-ticker panel with a ``(date, ticker)`` MultiIndex.
        manifest: JSON-serialisable run manifest.
        written: Paths of files saved by :meth:`save`.
    """

    config: PipelineConfig
    tickers: dict[str, TickerResult]
    panel: pd.DataFrame
    manifest: dict[str, Any]
    written: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path | None = None) -> list[str]:
        """Write per-ticker frames, the panel and the manifest to disk.

        Parquet is the primary format; CSV is written alongside when
        ``config.write_csv`` is set.

        Args:
            output_dir: Override for ``config.output_dir``.

        Returns:
            The paths written.
        """
        directory = Path(output_dir or self.config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        for ticker, result in self.tickers.items():
            if result.frame.empty:
                logger.warning("%s: nothing to save (empty frame)", ticker)
                continue
            path = directory / f"{ticker}.parquet"
            result.frame.to_parquet(path)
            written.append(str(path))
            if self.config.write_csv:
                csv_path = directory / f"{ticker}.csv"
                result.frame.to_csv(csv_path)
                written.append(str(csv_path))

        if not self.panel.empty:
            panel_path = directory / "panel.parquet"
            self.panel.to_parquet(panel_path)
            written.append(str(panel_path))
            if self.config.write_csv:
                panel_csv = directory / "panel.csv"
                self.panel.to_csv(panel_csv)
                written.append(str(panel_csv))

        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, indent=2, default=str))
        written.append(str(manifest_path))

        self.written = written
        logger.info("wrote %d file(s) to %s", len(written), directory)
        return written


def build_ticker_frame(
    ticker: str,
    config: PipelineConfig,
    *,
    ohlcv: pd.DataFrame | None = None,
    intraday: pd.DataFrame | None = None,
    fetch_covariates: bool = True,
) -> TickerResult:
    """Build the tidy output frame for one ticker.

    Args:
        ticker: Symbol to process.
        config: Pipeline configuration.
        ohlcv: Pre-loaded cleaned OHLCV; downloaded when ``None``.
        intraday: Pre-loaded intraday bars; downloaded when ``None`` and
            ``intraday_rv`` is among the configured estimators.
        fetch_covariates: Set ``False`` to skip covariates entirely -- useful
            for offline tests and for any caller that only wants the estimator
            columns.

    Returns:
        A :class:`TickerResult`.
    """
    if ohlcv is None:
        ohlcv, quality = load_ticker_ohlcv(ticker, config)
    else:
        quality = DataQualityReport(
            ticker=ticker, rows_downloaded=len(ohlcv), rows_kept=len(ohlcv)
        )

    result = TickerResult(ticker=ticker, frame=pd.DataFrame(), quality=quality)

    if ohlcv.empty:
        result.notes.append("no usable price history; ticker skipped")
        logger.error("%s: no usable price history", ticker)
        return result

    index = ohlcv.index
    columns: dict[str, pd.Series] = {}

    if "intraday_rv" in config.estimators and intraday is None:
        intraday = download_intraday(
            ticker,
            interval=config.intraday_interval,
            lookback_days=config.intraday_lookback_days,
            retries=max(1, config.max_download_retries - 1),
        )

    for name in config.estimators:
        estimate = compute_estimator(
            name,
            ohlcv,
            window=config.rv_window,
            annualization_factor=config.annualization_factor,
            intraday=intraday,
        )
        suffix = ESTIMATOR_COLUMN_SUFFIX[name]
        columns[f"rv_{suffix}"] = estimate.realized_vol
        if config.include_daily_variance:
            columns[f"var_{suffix}"] = estimate.daily_variance
        result.estimator_columns[name] = f"rv_{suffix}"
        if not estimate.available:
            note = f"{name}: no data available; rv_{suffix}/var_{suffix} are all-NaN"
            result.notes.append(note)
            logger.warning("%s: %s", ticker, note)

    # rv_* first, then var_*, then covariates -- stable, readable column order.
    frame = pd.DataFrame(index=index)
    for prefix in ("rv_", "var_"):
        for key, series in columns.items():
            if key.startswith(prefix):
                frame[key] = series

    if fetch_covariates and config.covariates:
        bundle = build_covariates(ticker, ohlcv, index, config)
        for column in bundle.frame.columns:
            frame[column] = bundle.frame[column]
        result.covariate_tags = bundle.tags
        result.notes.extend(bundle.notes)

    if config.dropna_rv:
        rv_cols = [c for c in frame.columns if c.startswith("rv_")]
        # Keep only dates where at least one estimator has a complete window;
        # this trims the leading warm-up, it never removes interior rows that
        # a later estimator could have filled.
        usable = frame[rv_cols].notna().any(axis=1)
        dropped = int((~usable).sum())
        if dropped:
            first_kept = frame.index[usable][0] if usable.any() else None
            result.notes.append(
                f"dropped {dropped} warm-up row(s) with no complete {config.rv_window}-day window"
                + (f"; series starts {first_kept:%Y-%m-%d}" if first_kept is not None else "")
            )
        frame = frame[usable]

    frame.index.name = "date"
    result.frame = frame
    logger.info("%s: built frame with %d rows x %d columns", ticker, len(frame), frame.shape[1])
    return result


def _build_panel(results: dict[str, TickerResult]) -> pd.DataFrame:
    """Stack per-ticker frames into a ``(date, ticker)`` MultiIndexed panel."""
    frames = []
    for ticker, result in results.items():
        if result.frame.empty:
            continue
        frame = result.frame.copy()
        frame["ticker"] = ticker
        frames.append(frame.set_index("ticker", append=True))
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames).sort_index()
    panel.index.names = ["date", "ticker"]
    return panel


def _build_manifest(
    config: PipelineConfig,
    results: dict[str, TickerResult],
    panel: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble the JSON run manifest."""
    per_ticker = {ticker: result.summary() for ticker, result in results.items()}
    non_empty = [r for r in results.values() if not r.frame.empty]

    covered_start = min((r.frame.index.min() for r in non_empty), default=None)
    covered_end = max((r.frame.index.max() for r in non_empty), default=None)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "date_range": {
            "requested_start": config.start,
            "requested_end": config.end,
            "covered_start": covered_start.strftime("%Y-%m-%d") if covered_start is not None else None,
            "covered_end": covered_end.strftime("%Y-%m-%d") if covered_end is not None else None,
        },
        "estimators": {
            name: {
                "column_rv": f"rv_{ESTIMATOR_COLUMN_SUFFIX[name]}",
                "column_var": f"var_{ESTIMATOR_COLUMN_SUFFIX[name]}" if config.include_daily_variance else None,
                "window": config.rv_window,
                "annualization_factor": config.annualization_factor,
            }
            for name in config.estimators
        },
        "covariate_tags": config.covariate_tags(),
        "lookahead_policy": {
            "rolling_windows": "value at t uses only the window ending at and including t",
            "past_only_covariates": "forward-filled from last observed value; never back-filled",
            "known_future_exception": "earnings_flag uses scheduled release dates by design",
            "price_adjustment": "auto_adjust=True; O/H/L/C adjusted consistently. VOLARE uses UNADJUSTED prices, so exact agreement is not expected.",
        },
        "row_counts": {t: int(len(r.frame)) for t, r in results.items()},
        "panel_rows": int(len(panel)),
        "tickers": per_ticker,
        "gaps_dropped": {
            t: r.quality.to_dict() for t, r in results.items() if r.quality.total_dropped
        },
        "skipped_tickers": [t for t, r in results.items() if r.frame.empty],
    }


def run_pipeline(config: PipelineConfig, *, save: bool = True) -> PipelineResult:
    """Run the full pipeline for every configured ticker.

    Args:
        config: Pipeline configuration.
        save: Write parquet/csv/manifest artefacts to ``config.output_dir``.

    Returns:
        A :class:`PipelineResult` holding per-ticker frames, the combined panel
        and the run manifest.
    """
    logger.info(
        "running pipeline: %s | %s -> %s | window=%d | estimators=%s",
        ", ".join(config.tickers), config.start, config.end, config.rv_window,
        ", ".join(config.estimators),
    )

    results: dict[str, TickerResult] = {}
    for ticker in config.tickers:
        results[ticker] = build_ticker_frame(ticker, config)

    panel = _build_panel(results)
    manifest = _build_manifest(config, results, panel)
    result = PipelineResult(config=config, tickers=results, panel=panel, manifest=manifest)

    if save:
        result.save()
    return result
