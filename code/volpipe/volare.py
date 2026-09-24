"""VOLARE dataset loader -- the rigorous historical data source.

VOLARE (https://volare.unime.it, Bucci et al., arXiv:2602.19732) publishes
pre-computed realized measures built from tick-level data.  This module ingests
a local copy of those files and re-emits them in **exactly the schema
:mod:`volpipe.pipeline` produces**, so the existing forecasters and the
walk-forward harness consume VOLARE with no changes at all -- only a
``--input-dir`` switch.

Why VOLARE rather than our own yfinance estimators:

* ``rv5`` is a true 5-minute realized variance, not a daily-OHLC approximation.
* It carries the companion measures (bipower variation, quarticity,
  semivariances) that the HAR-J / HAR-RS / HARQ specifications need, and which
  cannot be recovered from daily bars at all.
* It is the dataset Brini (2026, arXiv:2607.05291) benchmarks on, so using it
  is what makes our numbers comparable to that paper's.

**VOLARE uses UNADJUSTED prices**, whereas :mod:`volpipe.estimators` works from
split/dividend-adjusted OHLC.  The two series therefore will not agree exactly
-- they diverge around corporate actions.  That is expected, and it is precisely
why the rigorous core of this project runs on VOLARE while the yfinance path
covers live data and tickers VOLARE does not reach.

Expected on-disk layouts (all auto-detected by :func:`discover_symbols`)::

    <root>/<SYMBOL>/YYYY_MM_DD.parquet          # VOLARE native, per-day files
    <root>/<asset_class>/<SYMBOL>/YYYY_MM_DD.parquet
    <root>/<SYMBOL>.parquet | <SYMBOL>.csv      # already concatenated per asset
    <root>/<any>.parquet with a symbol column   # one combined file
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "VolareConfig",
    "VolareLoadReport",
    "MEASURE_ALIASES",
    "discover_symbols",
    "read_symbol_frame",
    "to_pipeline_schema",
    "load_symbol",
    "build_volare_dataset",
]

#: Our canonical measure name -> the VOLARE column names that can supply it,
#: in preference order.  The ``_ss`` variants are subsampled estimators, which
#: VOLARE recommends where available; the plain 5-minute column is the default
#: because it is what Brini's ``rv5`` target refers to.
MEASURE_ALIASES: dict[str, tuple[str, ...]] = {
    "rv5": ("rv5", "RV5", "rv_5", "rv5min", "realized_variance_5min"),
    "rv5_ss": ("rv5_ss", "RV5_SS", "rv5ss"),
    "rv1": ("rv1", "RV1"),
    "bv": ("bv5", "bv", "BV5", "bpv5", "bipower_variation"),
    "rq": ("rq5", "rq", "RQ5", "realized_quarticity"),
    "rsp": ("rsp5", "rsp", "RSP5", "rs_plus", "realized_semivariance_positive"),
    "rsn": ("rsn5", "rsn", "RSN5", "rs_minus", "realized_semivariance_negative"),
    "medrv": ("medrv5", "medrv", "MEDRV5"),
    "minrv": ("minrv5", "minrv", "MINRV5"),
    "rk": ("rk", "RK", "realized_kernel", "rk_parzen"),
    "rr5": ("rr5", "RR5", "realized_range"),
    # VOLARE's own range-based estimators.  These are the tick-data versions of
    # what volpipe.estimators computes from daily OHLC, so carrying them gives a
    # direct like-for-like check on our Parkinson / Garman-Klass implementations.
    "pr": ("pr", "PR", "parkinson_range"),
    "gkr": ("gkr", "GKR", "garman_klass_range"),
    # The distributed stock files name these *_price, not bare open/high/low/close.
    "open": ("open_price", "open", "Open", "OPEN"),
    "high": ("high_price", "high", "High", "HIGH"),
    "low": ("low_price", "low", "Low", "LOW"),
    "close": ("close_price", "close", "Close", "CLOSE"),
    "volume": ("volume", "Volume", "VOLUME"),
    "trades": ("trades", "Trades", "n_trades"),
}

#: Column names that can carry the observation date.
DATE_ALIASES: tuple[str, ...] = ("date", "Date", "DATE", "datetime", "timestamp", "day", "dt")

#: Column names that can carry the asset identifier in a combined file.
SYMBOL_ALIASES: tuple[str, ...] = ("symbol", "Symbol", "SYMBOL", "ticker", "Ticker", "asset", "id")

#: Measures that are *variances* and get a ``var_`` column in our schema.
VARIANCE_MEASURES: tuple[str, ...] = (
    "rv5", "rv5_ss", "rv1", "bv", "rsp", "rsn", "medrv", "minrv", "rk", "rr5", "pr", "gkr",
)

#: Measures we additionally expose as a volatility (``rv_``) column.
VOLATILITY_MEASURES: tuple[str, ...] = ("rv5", "rv5_ss", "bv", "medrv", "rk", "pr", "gkr")

_DATE_IN_NAME = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


@dataclass(frozen=True)
class VolareConfig:
    """Configuration for one VOLARE ingest run.

    Attributes:
        root: Directory holding the downloaded VOLARE files.
        symbols: Assets to load; ``None`` loads everything discovered.
        measure: Which realized-variance column becomes the primary target.
            ``"rv5"`` matches Brini; ``"rv5_ss"`` selects the subsampled variant.
        start: Inclusive ISO start date, or ``None`` for no lower bound.
        end: Inclusive ISO end date, or ``None`` for no upper bound.
        annualization_factor: Scales variances by this factor and volatilities
            by its square root.  **Default 1 (daily units), matching Brini.**
            QLIKE is invariant to any common positive rescaling, so this choice
            does not affect the headline metric -- but it does move the
            Mincer-Zarnowitz intercept, which is reported in raw units.
        min_observations: Assets with fewer usable days are skipped.
        output_dir: Where the per-asset parquet and manifest are written.
        write_csv: Also write a CSV next to each parquet.
    """

    root: str
    symbols: tuple[str, ...] | None = None
    measure: str = "rv5"
    start: str | None = None
    end: str | None = None
    annualization_factor: int = 1
    min_observations: int = 250
    output_dir: str = "output/volare"
    write_csv: bool = False

    def __post_init__(self) -> None:
        if self.measure not in MEASURE_ALIASES:
            raise ValueError(f"unknown measure {self.measure!r}; known: {sorted(MEASURE_ALIASES)}")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"start ({self.start}) must not be after end ({self.end})")

    @property
    def target_column(self) -> str:
        """The ``rv_*`` column downstream code should forecast."""
        return f"rv_{self.measure}"

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view for the manifest."""
        return {
            "root": str(Path(self.root).resolve()),
            "symbols": list(self.symbols) if self.symbols else None,
            "measure": self.measure,
            "target_column": self.target_column,
            "start": self.start,
            "end": self.end,
            "annualization_factor": self.annualization_factor,
            "min_observations": self.min_observations,
            "output_dir": self.output_dir,
        }


@dataclass
class VolareLoadReport:
    """What one asset's ingest produced or discarded.

    Attributes:
        symbol: Asset identifier.
        files_read: Number of source files concatenated.
        rows_raw: Rows before cleaning.
        rows_kept: Rows surviving cleaning.
        dropped: ``{reason: count}``.
        measures_found: Canonical measures successfully mapped.
        measures_missing: Canonical measures absent from the source.
        warnings: Notes worth surfacing.
    """

    symbol: str
    files_read: int = 0
    rows_raw: int = 0
    rows_kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    measures_found: list[str] = field(default_factory=list)
    measures_missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def record_drop(self, reason: str, count: int) -> None:
        """Note ``count`` rows removed for ``reason``."""
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    @property
    def total_dropped(self) -> int:
        """Total rows removed."""
        return sum(self.dropped.values())

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view for the manifest."""
        return {
            "symbol": self.symbol,
            "files_read": self.files_read,
            "rows_raw": self.rows_raw,
            "rows_kept": self.rows_kept,
            "rows_dropped": self.total_dropped,
            "dropped_by_reason": dict(self.dropped),
            "measures_found": list(self.measures_found),
            "measures_missing": list(self.measures_missing),
            "warnings": list(self.warnings),
        }


def _read_any(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read one parquet/CSV file into a frame.

    ``nrows`` limits a CSV read to a header probe; parquet ignores it, since
    reading its schema is already cheap.
    """
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
        return frame.head(nrows) if nrows else frame
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path, nrows=nrows)
    if suffix == ".gz":
        return pd.read_csv(path, compression="gzip", nrows=nrows)
    raise ValueError(f"unsupported file type: {path}")


def _resolve(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    """Return the first alias present in ``frame``, case-insensitively."""
    lowered = {str(c).lower(): str(c) for c in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def discover_symbols(root: str | Path) -> dict[str, list[Path]]:
    """Find every asset under ``root`` and the files that belong to it.

    Handles all four layouts documented in the module docstring, including the
    VOLARE-native one where each symbol is a directory of per-day parquet files
    (optionally nested one level under an asset-class directory).

    Args:
        root: Directory to scan.

    Returns:
        ``{symbol: [paths]}``, sorted.  A combined single-file dataset comes
        back under the sentinel key ``"__COMBINED__"``.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"VOLARE root does not exist: {base.resolve()}")

    found: dict[str, list[Path]] = {}

    # Layout A/B: symbol directories, possibly nested under an asset class.
    for directory in sorted(p for p in base.rglob("*") if p.is_dir()):
        files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in (".parquet", ".pq", ".csv")
        )
        if files:
            found.setdefault(directory.name.upper(), []).extend(files)

    # Layout C: one file per asset, named after the symbol.
    for path in sorted(base.glob("*")):
        if path.is_file() and path.suffix.lower() in (".parquet", ".pq", ".csv"):
            stem = path.stem.upper()
            # A date-named file at the top level is not an asset name.
            if _DATE_IN_NAME.fullmatch(stem.replace("-", "_")):
                continue
            found.setdefault(stem, []).append(path)

    if not found:
        raise FileNotFoundError(
            f"no parquet/csv files found under {base.resolve()}. Expected VOLARE bulk "
            f"download layout, e.g. <root>/<SYMBOL>/YYYY_MM_DD.parquet"
        )

    # Layout D: a single top-level file carrying a symbol column.
    if len(found) == 1:
        (only_symbol, paths), = found.items()
        if len(paths) == 1:
            probe = _read_any(paths[0], nrows=5)  # headers only; the file may be huge
            if _resolve(probe, SYMBOL_ALIASES) is not None:
                logger.info("detected a combined multi-asset file: %s", paths[0].name)
                return {"__COMBINED__": paths}

    return {symbol: sorted(paths) for symbol, paths in sorted(found.items())}


def _date_from_filename(path: Path) -> pd.Timestamp | None:
    """Recover a date from a ``YYYY_MM_DD``-style filename."""
    match = _DATE_IN_NAME.search(path.stem)
    if not match:
        return None
    try:
        return pd.Timestamp(f"{match.group(1)}-{match.group(2)}-{match.group(3)}")
    except ValueError:
        return None


def read_symbol_frame(paths: Sequence[Path], report: VolareLoadReport) -> pd.DataFrame:
    """Concatenate one asset's source files into a single date-indexed frame.

    VOLARE ships one parquet per trading day, each of which may hold a single
    row (the day's measures) or intraday rows.  Multi-row days are collapsed by
    taking the last row, since the realized measures are daily summaries
    repeated across the file rather than per-bar values.

    Args:
        paths: Files belonging to one asset.
        report: Mutated with the file count and any warnings.

    Returns:
        Frame indexed by normalised date, original VOLARE column names intact.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = _read_any(path)
        except Exception as exc:
            report.warnings.append(f"could not read {path.name}: {exc}")
            logger.warning("%s: could not read %s (%s)", report.symbol, path.name, exc)
            continue
        if frame.empty:
            continue

        date_column = _resolve(frame, DATE_ALIASES)
        if date_column is not None:
            frame = frame.assign(_date=pd.to_datetime(frame[date_column], errors="coerce"))
        else:
            stamp = _date_from_filename(path)
            if stamp is None:
                report.warnings.append(f"{path.name}: no date column or date in filename")
                continue
            frame = frame.assign(_date=stamp)

        frames.append(frame)
        report.files_read += 1

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["_date"])
    index = pd.DatetimeIndex(combined["_date"])
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    combined.index = index.normalize()
    combined.index.name = "date"
    combined = combined.drop(columns=["_date"])

    report.rows_raw = int(len(combined))
    duplicates = combined.index.duplicated(keep="last")
    if duplicates.any():
        report.record_drop("duplicate_date", int(duplicates.sum()))
        combined = combined[~duplicates]

    return combined.sort_index()


def to_pipeline_schema(
    raw: pd.DataFrame,
    config: VolareConfig,
    report: VolareLoadReport,
) -> pd.DataFrame:
    """Map VOLARE's columns onto the schema :mod:`volpipe.pipeline` emits.

    Produces, for every measure present in the source:

    * ``var_<measure>`` -- the realized **variance**, annualised by
      ``config.annualization_factor``.
    * ``rv_<measure>`` -- the corresponding **volatility**, ``sqrt(variance)``.
      ``rv_rv5`` is the primary forecasting target and matches Brini's
      ``sigma_t = sqrt(RV_t)`` definition.
    * ``jump`` -- ``max(rv5 - bv, 0)``, the Barndorff-Nielsen/Shephard jump
      component that HAR-J needs.
    * OHLCV columns where present.

    Args:
        raw: Concatenated VOLARE frame with its original column names.
        config: Ingest configuration.
        report: Mutated with which measures were found or missing.

    Returns:
        Frame indexed by trading date in our canonical schema.
    """
    if raw.empty:
        return pd.DataFrame()

    factor = float(config.annualization_factor)
    out = pd.DataFrame(index=raw.index)

    for canonical, aliases in MEASURE_ALIASES.items():
        column = _resolve(raw, aliases)
        if column is None:
            if canonical in VARIANCE_MEASURES:
                report.measures_missing.append(canonical)
            continue

        values = pd.to_numeric(raw[column], errors="coerce").astype("float64")
        if canonical in ("open", "high", "low", "close", "volume", "trades"):
            out[canonical] = values
            continue

        report.measures_found.append(canonical)
        # A realized variance cannot be negative; treat any such value as bad data.
        variance = values.where(values >= 0.0)
        out[f"var_{canonical}"] = variance * factor
        if canonical in VOLATILITY_MEASURES:
            out[f"rv_{canonical}"] = np.sqrt(variance * factor)

    target = f"var_{config.measure}"
    if target not in out.columns:
        raise KeyError(
            f"{report.symbol}: no column supplying {config.measure!r}. "
            f"Looked for {MEASURE_ALIASES[config.measure]}; source has {list(raw.columns)[:25]}"
        )

    # Jump component for HAR-J: the part of RV not explained by bipower variation.
    if "var_bv" in out.columns:
        out["jump"] = (out[target] - out["var_bv"]).clip(lower=0.0)
    else:
        report.warnings.append("no bipower variation column; HAR-J will be unavailable")

    before = len(out)
    usable = out[target].notna() & (out[target] > 0.0)
    out = out[usable]
    report.record_drop("missing_or_nonpositive_target", before - len(out))

    if config.start is not None:
        before = len(out)
        out = out[out.index >= pd.Timestamp(config.start)]
        report.record_drop("before_start", before - len(out))
    if config.end is not None:
        before = len(out)
        out = out[out.index <= pd.Timestamp(config.end)]
        report.record_drop("after_end", before - len(out))

    report.rows_kept = int(len(out))
    out.index.name = "date"
    return out


def load_symbol(
    symbol: str,
    paths: Sequence[Path],
    config: VolareConfig,
) -> tuple[pd.DataFrame, VolareLoadReport]:
    """Load and convert one asset.

    Args:
        symbol: Asset identifier.
        paths: Its source files.
        config: Ingest configuration.

    Returns:
        ``(frame_in_our_schema, report)``.
    """
    report = VolareLoadReport(symbol=symbol)
    raw = read_symbol_frame(paths, report)
    if raw.empty:
        report.warnings.append("no readable rows")
        return pd.DataFrame(), report

    frame = to_pipeline_schema(raw, config, report)
    logger.info(
        "%s: %d rows from %d file(s) (%s -> %s); measures: %s",
        symbol, len(frame), report.files_read,
        frame.index.min().date() if not frame.empty else "-",
        frame.index.max().date() if not frame.empty else "-",
        ", ".join(report.measures_found) or "none",
    )
    return frame, report


def _split_combined(paths: Sequence[Path]) -> dict[str, pd.DataFrame]:
    """Split a single multi-asset file into per-symbol frames."""
    frame = pd.concat([_read_any(p) for p in paths], ignore_index=True)
    symbol_column = _resolve(frame, SYMBOL_ALIASES)
    if symbol_column is None:  # pragma: no cover - guarded by discover_symbols
        raise KeyError("combined file has no symbol column")
    return {
        str(symbol).upper(): group.drop(columns=[symbol_column])
        for symbol, group in frame.groupby(symbol_column)
    }


def build_volare_dataset(config: VolareConfig) -> dict[str, object]:
    """Ingest VOLARE and write it out in the pipeline's schema.

    Writes ``<output_dir>/<SYMBOL>.parquet`` for each asset, a combined
    ``panel.parquet`` with a ``(date, ticker)`` MultiIndex, and a
    ``manifest.json`` recording the config, per-asset coverage and every
    dropped row with its reason -- the same artefacts
    :func:`volpipe.pipeline.run_pipeline` produces, so downstream code cannot
    tell the two sources apart.

    Args:
        config: Ingest configuration.

    Returns:
        The manifest dict.

    Raises:
        FileNotFoundError: If ``config.root`` holds no usable files.
    """
    discovered = discover_symbols(config.root)

    if "__COMBINED__" in discovered:
        raw_by_symbol = _split_combined(discovered["__COMBINED__"])
    else:
        raw_by_symbol = None

    wanted = {s.upper() for s in config.symbols} if config.symbols else None
    frames: dict[str, pd.DataFrame] = {}
    reports: list[VolareLoadReport] = []
    skipped: dict[str, str] = {}

    symbols = sorted(raw_by_symbol) if raw_by_symbol is not None else sorted(discovered)
    for symbol in symbols:
        if wanted is not None and symbol not in wanted:
            continue

        if raw_by_symbol is not None:
            report = VolareLoadReport(symbol=symbol, files_read=1)
            raw = raw_by_symbol[symbol]
            date_column = _resolve(raw, DATE_ALIASES)
            if date_column is None:
                skipped[symbol] = "no date column"
                continue
            raw = raw.set_index(pd.DatetimeIndex(pd.to_datetime(raw[date_column])).normalize())
            raw.index.name = "date"
            report.rows_raw = int(len(raw))
            frame = to_pipeline_schema(raw.sort_index(), config, report)
        else:
            frame, report = load_symbol(symbol, discovered[symbol], config)

        reports.append(report)
        if frame.empty:
            skipped[symbol] = "no usable rows"
            continue
        if len(frame) < config.min_observations:
            skipped[symbol] = f"only {len(frame)} rows (< min_observations={config.min_observations})"
            logger.warning("%s: %s", symbol, skipped[symbol])
            continue
        frames[symbol] = frame

    if wanted:
        for missing in sorted(wanted - set(frames) - set(skipped)):
            skipped[missing] = "not found in the VOLARE root"
            logger.warning("%s: requested but not found under %s", missing, config.root)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for symbol, frame in frames.items():
        path = output_dir / f"{symbol}.parquet"
        frame.to_parquet(path)
        written.append(str(path))
        if config.write_csv:
            frame.to_csv(output_dir / f"{symbol}.csv")
            written.append(str(output_dir / f"{symbol}.csv"))

    if frames:
        panel = pd.concat(
            {symbol: frame for symbol, frame in frames.items()}, names=["ticker", "date"]
        ).reorder_levels(["date", "ticker"]).sort_index()
        panel.to_parquet(output_dir / "panel.parquet")
        written.append(str(output_dir / "panel.parquet"))
    else:
        panel = pd.DataFrame()

    manifest: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "VOLARE (volare.unime.it, arXiv:2602.19732)",
        "price_basis": "UNADJUSTED tick data -- differs from volpipe's adjusted-OHLC estimators by design",
        "config": config.to_dict(),
        "assets": {
            symbol: {
                "rows": int(len(frame)),
                "start": frame.index.min().strftime("%Y-%m-%d"),
                "end": frame.index.max().strftime("%Y-%m-%d"),
                "columns": list(frame.columns),
            }
            for symbol, frame in frames.items()
        },
        "panel_rows": int(len(panel)),
        "load_reports": [r.to_dict() for r in reports],
        "skipped": skipped,
        "files_written": written,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    logger.info(
        "VOLARE ingest: %d asset(s), %d panel rows -> %s",
        len(frames), len(panel), output_dir.resolve(),
    )
    return manifest
