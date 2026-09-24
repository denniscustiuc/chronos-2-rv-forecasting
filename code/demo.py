#!/usr/bin/env python
"""End-to-end demo: build realized-volatility features for a few tickers.

Usage::

    python demo.py                                  # AAPL, AMZN, MSFT, last 5 years
    python demo.py --tickers NVDA TSLA --years 3
    python demo.py --with-intraday --covariates vix volume earnings_flag sector_etf_vol

Writes ``output/<TICKER>.parquet``, ``output/panel.parquet``,
``output/manifest.json`` and one estimator-overlay PNG per ticker, then prints
a summary: rows per ticker, date range, and the estimator correlation matrix.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from volpipe import (
    DEFAULT_COVARIATES,
    DEFAULT_ESTIMATORS,
    PipelineConfig,
    compare_with_volare,
    estimator_correlation_matrix,
    estimator_noise_ranking,
    plot_estimators,
    run_pipeline,
)

DEFAULT_TICKERS = ("AAPL", "AMZN", "MSFT")


def parse_args() -> argparse.Namespace:
    """Parse command-line options into an argparse namespace."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS), help="symbols to process")
    parser.add_argument("--years", type=float, default=5.0, help="years of history (default: 5)")
    parser.add_argument("--start", default=None, help="explicit ISO start date; overrides --years")
    parser.add_argument("--end", default=None, help="explicit ISO end date (exclusive)")
    parser.add_argument("--window", type=int, default=21, help="RV window in trading days (default: 21)")
    parser.add_argument("--annualization", type=int, default=252, help="trading days per year (default: 252)")
    parser.add_argument("--estimators", nargs="+", default=list(DEFAULT_ESTIMATORS))
    parser.add_argument("--covariates", nargs="+", default=list(DEFAULT_COVARIATES))
    parser.add_argument("--earnings-k", type=int, default=1, help="+/-k trading days around a release")
    parser.add_argument("--with-intraday", action="store_true", help="also compute intraday_rv (~60d only)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--csv", action="store_true", help="write CSV alongside parquet")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="warnings and errors only")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Turn parsed CLI arguments into a :class:`PipelineConfig`."""
    end = args.end or date.today().isoformat()
    start = args.start or (date.fromisoformat(end) - timedelta(days=int(args.years * 365.25))).isoformat()

    estimators = list(args.estimators)
    if args.with_intraday and "intraday_rv" not in estimators:
        estimators.append("intraday_rv")

    return PipelineConfig(
        tickers=tuple(t.upper() for t in args.tickers),
        start=start,
        end=end,
        rv_window=args.window,
        annualization_factor=args.annualization,
        estimators=tuple(estimators),
        covariates=tuple(args.covariates),
        earnings_window_days=args.earnings_k,
        output_dir=args.output_dir,
        write_csv=args.csv,
    )


def print_summary(result, *, make_plots: bool) -> None:
    """Print rows/date-range/correlations per ticker and write overlay plots."""
    config = result.config
    output_dir = Path(config.output_dir)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"window={config.rv_window}d  annualization={config.annualization_factor}  "
          f"requested {config.start} -> {config.end}")

    rows = []
    for ticker, ticker_result in result.tickers.items():
        frame = ticker_result.frame
        if frame.empty:
            rows.append({"ticker": ticker, "rows": 0, "start": "-", "end": "-", "columns": 0})
            continue
        rows.append(
            {
                "ticker": ticker,
                "rows": len(frame),
                "start": frame.index.min().strftime("%Y-%m-%d"),
                "end": frame.index.max().strftime("%Y-%m-%d"),
                "columns": frame.shape[1],
            }
        )
    print("\nRows per ticker")
    print(pd.DataFrame(rows).set_index("ticker").to_string())

    for ticker, ticker_result in result.tickers.items():
        frame = ticker_result.frame
        if frame.empty:
            print(f"\n[{ticker}] skipped: {'; '.join(ticker_result.notes) or 'no data'}")
            continue

        print(f"\n[{ticker}] estimator correlation matrix (annualised RV)")
        print(estimator_correlation_matrix(frame).round(3).to_string())

        print(f"\n[{ticker}] noise ranking (std of daily change / mean level; CC expected noisiest)")
        print(estimator_noise_ranking(frame).round(4).to_string())

        if ticker_result.covariate_tags:
            tags = ", ".join(f"{c} [{t}]" for c, t in ticker_result.covariate_tags.items())
            print(f"\n[{ticker}] covariates: {tags}")
        for note in ticker_result.notes:
            print(f"  note: {note}")

        if make_plots:
            path = plot_estimators(frame, ticker, output_dir / f"{ticker}_estimators.png")
            if path:
                print(f"  plot: {path}")

    print("\nVOLARE cross-check (optional, not yet wired up):")
    first = next((r for r in result.tickers.values() if not r.frame.empty), None)
    if first is not None:
        print(f"  {compare_with_volare(first.frame)['detail']}")

    print(f"\nArtefacts written to {output_dir.resolve()}:")
    for path in result.written:
        print(f"  {path}")
    print()


def main() -> None:
    """Run the demo pipeline end to end."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    # yfinance chatters on its own logger about every empty/partial response.
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    config = build_config(args)
    result = run_pipeline(config)
    print_summary(result, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
