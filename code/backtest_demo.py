#!/usr/bin/env python
"""End-to-end baseline backtest: naive floors + HAR + Log-HAR, walk-forward.

Consumes the realized-volatility series produced by the yfinance pipeline
(``output/<TICKER>.parquet``) -- but note the harness itself is data-source
agnostic: point ``--input-dir`` at VOLARE-derived parquet with the same shape
and nothing else changes.

Usage::

    python backtest_demo.py                              # AAPL/AMZN/MSFT, h=1,5,21
    python backtest_demo.py --target rv_yz --with-mem
    python backtest_demo.py --horizons 21 --refit 1

Writes the tidy results frame, the metric summaries, a run manifest and a
forecast-vs-realized plot into ``output/backtest/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from volmodels import BENCHMARK_MODEL, baseline_factories
from voleval import (
    BacktestConfig,
    add_losses,
    coverage_table,
    dm_against_benchmark,
    leaderboard,
    model_confidence_set,
    per_ticker_dm,
    run_backtest,
    summarize,
)

DEFAULT_TICKERS = ("AAPL", "AMZN", "MSFT")

# Chart palette: categorical slots 1 and 2 of the reference design system.
# Validated (light surface, all pairs): CVD dE 24.7, normal-vision dE 33.6,
# both marks >= 3:1 against the surface.
COLOR_REALIZED = "#2a78d6"
COLOR_FORECAST = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--input-dir", default="output", help="directory of per-ticker parquet")
    parser.add_argument("--target", default="rv_gk", help="RV column to forecast (default: rv_gk)")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 21])
    parser.add_argument("--min-train", type=int, default=250)
    parser.add_argument("--window", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--rolling-size", type=int, default=None)
    parser.add_argument("--refit", type=int, default=5, help="re-estimate every k origins")
    parser.add_argument("--with-mem", action="store_true", help="also run the MEM baseline")
    parser.add_argument("--plot-ticker", default=None, help="ticker for the plot (default: first)")
    parser.add_argument("--plot-horizon", type=int, default=21)
    parser.add_argument("--output-dir", default="output/backtest")
    parser.add_argument("--bootstrap", type=int, default=1000, help="MCS bootstrap replications")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_series(input_dir: str, tickers: list[str], column: str) -> dict[str, pd.Series]:
    """Load one RV column per ticker from the pipeline's parquet output.

    Args:
        input_dir: Directory holding ``<TICKER>.parquet``.
        tickers: Symbols to load.
        column: RV column to extract.

    Returns:
        ``{ticker: series}``, skipping tickers whose file or column is missing.

    Raises:
        FileNotFoundError: If no ticker could be loaded at all.
    """
    directory = Path(input_dir)
    series: dict[str, pd.Series] = {}

    for ticker in tickers:
        path = directory / f"{ticker}.parquet"
        if not path.exists():
            logging.error("%s: %s not found -- run `python demo.py` first", ticker, path)
            continue
        frame = pd.read_parquet(path)
        if column not in frame.columns:
            available = [c for c in frame.columns if c.startswith("rv_")]
            logging.error("%s: no %r column (available: %s)", ticker, column, available)
            continue
        values = frame[column].dropna()
        if values.empty:
            logging.error("%s: %r is entirely missing", ticker, column)
            continue
        series[ticker] = values.rename(ticker)
        logging.info(
            "%s: %d observations of %s (%s -> %s)",
            ticker, len(values), column,
            values.index.min().date(), values.index.max().date(),
        )

    if not series:
        raise FileNotFoundError(
            f"no usable series in {directory.resolve()}. Run `python demo.py` to build the pipeline output."
        )
    return series


def plot_forecast_vs_realized(
    results: pd.DataFrame,
    ticker: str,
    horizon: int,
    output_path: Path,
    *,
    model: str = BENCHMARK_MODEL,
) -> Path | None:
    """Plot realized vs forecast with the model's 90% prediction band shaded.

    A fan chart: the band is the ``q0.05``-``q0.95`` interval, so roughly one
    realized point in ten should fall outside it.  Both series are plotted
    against the **target** date -- the day the forecast is about -- so the
    comparison is vertical rather than offset by the horizon.

    Args:
        results: Tidy results frame with quantile columns.
        ticker: Asset to plot.
        horizon: Horizon to slice.
        output_path: Destination PNG.
        model: Which model's forecast and band to draw.

    Returns:
        The path written, or ``None`` if matplotlib is missing or there is
        nothing to plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping plot")
        return None

    frame = results[
        (results["ticker"] == ticker)
        & (results["horizon"] == horizon)
        & (results["model"] == model)
    ].sort_values("target_date")
    if frame.empty:
        logging.warning("nothing to plot for %s/%s at h=%d", ticker, model, horizon)
        return None

    dates = frame["target_date"]
    figure, axes = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)

    has_band = {"q0.05", "q0.95"} <= set(frame.columns)
    if has_band:
        axes.fill_between(
            dates, frame["q0.05"], frame["q0.95"],
            color=COLOR_FORECAST, alpha=0.18, linewidth=0,
            label=f"{model} 90% interval",
        )

    axes.plot(dates, frame["realized"], color=COLOR_REALIZED, linewidth=2.0, label="realized")
    axes.plot(
        dates, frame["forecast"], color=COLOR_FORECAST, linewidth=2.0,
        label=f"{model} forecast",
    )

    if has_band:
        inside = frame["realized"].between(frame["q0.05"], frame["q0.95"]).mean()
        subtitle = f"90% band covers {inside:.0%} of realisations (nominal 90%)"
    else:
        subtitle = ""

    axes.set_title(
        f"{ticker} — {horizon}-day-ahead realized volatility forecast",
        color=INK_PRIMARY, fontsize=13, pad=30, loc="left",
    )
    if subtitle:
        axes.text(
            0.0, 1.022, subtitle, transform=axes.transAxes,
            color=INK_SECONDARY, fontsize=10,
        )

    axes.set_ylabel("annualised volatility", color=INK_SECONDARY, fontsize=10)
    axes.set_xlabel("target date", color=INK_SECONDARY, fontsize=10)
    axes.tick_params(colors=INK_SECONDARY, labelsize=9)
    axes.grid(alpha=0.18, linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color("#d8d7d2")

    legend = axes.legend(frameon=False, fontsize=10, loc="upper left", ncol=3)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logging.info("wrote plot to %s", output_path)
    return output_path


def print_report(
    scored: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: BacktestConfig,
    *,
    bootstrap: int,
) -> dict[str, object]:
    """Print the leaderboards, significance tests and calibration summary.

    Returns:
        A dict of the significance findings, for the manifest.
    """
    pd.set_option("display.width", 200)
    findings: dict[str, object] = {}

    print("\n" + "=" * 78)
    print("LEADERBOARD  (QLIKE, lower is better; ratio < 1 beats Log-HAR)")
    print("=" * 78)
    for horizon in config.horizons:
        print(f"\n--- h = {horizon} trading days " + "-" * 45)
        print("\n  pooled (every observation weighted equally)")
        print(leaderboard(tables["pooled"], horizon).round(5).to_string())
        print("\n  equal-weighted across assets")
        print(leaderboard(tables["equal_weighted"], horizon).round(5).to_string())

    print("\n" + "=" * 78)
    print(f"DIEBOLD-MARIANO vs {BENCHMARK_MODEL}  (negative stat favours the challenger)")
    print("=" * 78)
    for horizon in config.horizons:
        table = dm_against_benchmark(scored, horizon, benchmark=BENCHMARK_MODEL)
        if table.empty:
            continue
        print(f"\n--- h = {horizon} (pooled) " + "-" * 45)
        print(table.round(4).to_string(index=False))
        findings[f"dm_pooled_h{horizon}"] = table.to_dict("records")

        per_asset = per_ticker_dm(scored, horizon, benchmark=BENCHMARK_MODEL)
        if not per_asset.empty:
            verdicts = (
                per_asset.groupby("model_a", observed=True)["better"]
                .value_counts()
                .unstack(fill_value=0)
            )
            print(f"\n  per-ticker verdicts (n={per_asset['ticker'].nunique()} assets):")
            print(verdicts.to_string())
            findings[f"dm_per_ticker_h{horizon}"] = per_asset.to_dict("records")

    print("\n" + "=" * 78)
    print(f"MODEL CONFIDENCE SET  (alpha=0.10, {bootstrap} block-bootstrap draws)")
    print("=" * 78)
    for horizon in config.horizons:
        outcome = model_confidence_set(scored, horizon, n_bootstrap=bootstrap, seed=0)
        print(f"\n--- h = {horizon} " + "-" * 55)
        print(outcome.table.round(5).to_string(index=False))
        print(f"  surviving set: {', '.join(outcome.included) or '(none)'}")
        findings[f"mcs_h{horizon}"] = outcome.table.to_dict("records")

    print("\n" + "=" * 78)
    print("CALIBRATION  (empirical vs nominal quantile coverage)")
    print("=" * 78)
    coverage = coverage_table(scored)
    if not coverage.empty:
        headline = coverage[coverage["level"].isin([0.05, 0.5, 0.95])]
        pivot = headline.pivot_table(
            index=["model", "horizon"], columns="level", values="empirical", observed=True
        )
        print("\n  fraction of realisations at or below each quantile")
        print("  (well-calibrated => column 0.05 ~ 0.05, 0.5 ~ 0.5, 0.95 ~ 0.95)")
        print(pivot.round(3).to_string())

    return findings


def main() -> None:
    """Run the baseline backtest end to end."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    series = load_series(args.input_dir, args.tickers, args.target)
    config = BacktestConfig(
        horizons=tuple(args.horizons),
        window=args.window,
        min_train_size=args.min_train,
        rolling_window_size=args.rolling_size,
        refit_frequency=args.refit,
    )

    outcome = run_backtest(series, baseline_factories(include_mem=args.with_mem), config)
    if outcome.results.empty:
        raise SystemExit("backtest produced no results -- is min_train_size larger than the sample?")

    scored = add_losses(outcome.results)
    tables = summarize(scored, benchmark=BENCHMARK_MODEL)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    scored.to_parquet(output_dir / "results.parquet")
    written.append(str(output_dir / "results.parquet"))
    for name, table in tables.items():
        path = output_dir / f"summary_{name}.csv"
        table.to_csv(path, index=False)
        written.append(str(path))
    coverage = coverage_table(scored)
    if not coverage.empty:
        coverage.to_csv(output_dir / "calibration.csv", index=False)
        written.append(str(output_dir / "calibration.csv"))
    outcome.diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    written.append(str(output_dir / "diagnostics.csv"))

    findings = print_report(scored, tables, config, bootstrap=args.bootstrap)

    if not args.no_plot:
        ticker = args.plot_ticker or next(iter(series))
        path = plot_forecast_vs_realized(
            scored, ticker, args.plot_horizon,
            output_dir / f"{ticker}_h{args.plot_horizon}_forecast.png",
        )
        if path:
            written.append(str(path))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_column": args.target,
        "input_dir": str(Path(args.input_dir).resolve()),
        "tickers": {t: {"rows": int(len(s)),
                        "start": s.index.min().strftime("%Y-%m-%d"),
                        "end": s.index.max().strftime("%Y-%m-%d")}
                    for t, s in series.items()},
        "backtest_config": config.to_dict(),
        "models": outcome.models,
        "benchmark": BENCHMARK_MODEL,
        "result_rows": int(len(scored)),
        "elapsed_seconds": round(outcome.elapsed_seconds, 2),
        "diagnostics": outcome.diagnostics.to_dict("records"),
        "significance": findings,
        "notes": [
            "The target for origin t and horizon h is the RV value at t+h; using that "
            "future value is what makes it a forecast target. Model INPUTS never extend past t.",
            f"On a rolling-window RV series, horizons below the RV window overlap the "
            f"inputs and inflate apparent skill; read h >= window as the honest comparison.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    written.append(str(output_dir / "manifest.json"))

    print("\n" + "=" * 78)
    print(f"Artefacts written to {output_dir.resolve()}")
    for path in written:
        print(f"  {path}")
    print(f"\nTotal: {len(scored)} forecast rows in {outcome.elapsed_seconds:.1f}s")
    print()


if __name__ == "__main__":
    main()
