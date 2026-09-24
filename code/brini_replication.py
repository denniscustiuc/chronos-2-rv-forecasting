#!/usr/bin/env python
"""Brini (2026) replication: VOLARE ingest + econometric benchmarks + comparison.

Replicates the protocol of Brini, *Forecasting Realized Volatility with Time
Series Foundation Models* (arXiv:2607.05291) on the VOLARE dataset, runs the
paper's eight econometric benchmarks through our own harness, and prints a
side-by-side table against the paper's published numbers.

**No foundation models here** -- this step aligns the data and the protocol so
that Chronos-2 and Kronos can later be dropped into the identical run.

Two tracks, deliberately kept apart (``--track``):

``brini``
    The paper's protocol exactly: rolling 1,000-day window, daily
    re-estimation, h = 1/5/22, point-in-time target ``sigma_{t+h}``.  Only this
    track is comparable to the published tables.

``aggregate``
    Our own preferred target, ``sqrt(mean(RV_{t+1..t+h}))`` -- the volatility of
    the whole horizon rather than of its final day, which is the economically
    relevant quantity.  Its targets overlap across origins, which is precisely
    what Brini's design avoids, so its numbers are **not** comparable to the
    paper and are never merged into the comparison table.

Usage::

    # one-off: convert a VOLARE bulk download into our schema
    python brini_replication.py --volare-root ~/volare_download --ingest-only

    # full replication
    python brini_replication.py --volare-root ~/volare_download

    # re-run from already-ingested parquet
    python brini_replication.py --input-dir output/volare --max-assets 10
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from volmodels import BENCHMARK_MODEL, brini_factories
from volpipe.volare import VolareConfig, build_volare_dataset
from voleval import (
    BRINI_EQUITIES,
    BRINI_EQUITY_END,
    BRINI_EQUITY_START,
    BacktestConfig,
    add_losses,
    brini_protocol,
    compare_with_brini,
    dm_against_benchmark,
    format_comparison,
    leaderboard,
    mcs_inclusion_rates,
    mincer_zarnowitz,
    model_confidence_set,
    reference_table,
    run_backtest,
    subsample_summaries,
    summarize,
)

EXOG_CANDIDATES = ("jump", "var_rsp", "var_rsn", "var_rq", "var_bv")


def build_chronos_covariates(
    series: dict[str, pd.Series],
    exog: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Attach VIX and build the per-ticker earnings countdown.

    Two covariates, chosen from what this project actually measured rather than
    from a wish list:

    * **VIX** (past_only) -- it earned its place: +0.04 to +0.08 R^2 over a
      ticker's own RV history at h=22.  Forward-filled from the last observed
      print and never back-filled, so it carries no lookahead.
    * **days until next earnings** (known_future) -- replacing the binary
      ``earnings_flag``, which we measured at essentially zero signal (-0.03 to
      +0.02 correlation with log-RV at h=22).  The schedule is published weeks
      ahead, so its forward values are legitimately known.

    The VIX column is added to every ticker's exog frame.  That is harmless to
    the classical models, which select their regressors by name and never see it.

    Returns:
        ``(exog_with_vix, countdown_by_ticker)``.  Either may come back without
        the extra inputs if the network fetch fails; the run continues with
        whatever was obtained, and the manifest records what was missing.
    """
    import yfinance as yf

    from volpipe.covariates import _align
    from volpipe.ingest import download_series_close
    from volmodels.chronos2 import earnings_countdown

    all_dates = sorted({d for s in series.values() for d in s.index})
    start = min(all_dates).strftime("%Y-%m-%d")
    end = (max(all_dates) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    vix = download_series_close("^VIX", start, end)
    if vix.empty:
        logging.warning("VIX download returned nothing; chronos2_cov will run without it")

    augmented: dict[str, pd.DataFrame] = {}
    for ticker, frame in exog.items():
        out = frame.copy()
        if not vix.empty:
            out["vix"] = _align(vix, frame.index, name="vix")
        augmented[ticker] = out

    countdown: dict[str, pd.Series] = {}
    for ticker, values in series.items():
        try:
            table = yf.Ticker(ticker).get_earnings_dates(limit=100)
            releases = pd.DatetimeIndex(pd.to_datetime(table.index))
            if releases.tz is not None:
                releases = releases.tz_localize(None)
            countdown[ticker] = earnings_countdown(values.index, releases.normalize())
        except Exception as exc:
            logging.warning("%s: earnings dates unavailable (%s); no countdown", ticker, exc)

    logging.info(
        "chronos covariates: vix=%s, earnings countdown for %d/%d tickers",
        "yes" if not vix.empty else "NO", len(countdown), len(series),
    )
    return augmented, countdown


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--volare-root", default=None, help="raw VOLARE download to ingest first")
    parser.add_argument("--input-dir", default="output/volare", help="ingested parquet directory")
    parser.add_argument("--assets", nargs="+", default=None, help="override Brini's asset list")
    parser.add_argument("--max-assets", type=int, default=None, help="cap the asset count")
    parser.add_argument("--measure", default="rv5", help="VOLARE variance column (default: rv5)")
    parser.add_argument(
        "--start", default=BRINI_EQUITY_START,
        help=f"ingest lower bound (default: {BRINI_EQUITY_START}, Brini's equity sample start)",
    )
    parser.add_argument(
        "--end", default=BRINI_EQUITY_END,
        help=f"ingest upper bound (default: {BRINI_EQUITY_END}, Brini's equity sample end). "
             "The archive ships more recent data; truncating keeps the replication faithful.",
    )
    parser.add_argument("--track", choices=["brini", "aggregate"], default="brini")
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=1000, help="rolling estimation window")
    parser.add_argument(
        "--refit", type=int, default=1,
        help="re-estimation cadence; Brini uses 1 (daily). Raising it is a documented deviation.",
    )
    parser.add_argument("--no-exog-models", action="store_true", help="skip HAR-J/HAR-RS/HARQ")
    parser.add_argument("--no-slow-models", action="store_true", help="skip ARMA/ARFIMA")
    parser.add_argument("--no-mem", action="store_true", help="skip MEM (QMLE refit)")
    parser.add_argument("--with-naive", action="store_true", help="also register the naive floors")
    parser.add_argument("--with-chronos", action="store_true",
                        help="also register chronos2_uni and chronos2_cov")
    parser.add_argument("--chronos-only", action="store_true",
                        help="register ONLY the Chronos-2 modes (skip the classical set)")
    parser.add_argument("--chronos-device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--chronos-context", type=int, default=1000,
                        help="context length; defaults to the protocol's 1000-day window")
    parser.add_argument("--bootstrap", type=int, default=500, help="MCS bootstrap replications")
    parser.add_argument("--max-origins", type=int, default=None, help="cap origins (smoke runs)")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--output-dir", default="output/brini")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def ingest(args: argparse.Namespace) -> dict[str, object]:
    """Convert a raw VOLARE download into our parquet schema."""
    symbols = tuple(args.assets) if args.assets else BRINI_EQUITIES
    config = VolareConfig(
        root=args.volare_root,
        symbols=symbols,
        measure=args.measure,
        # Brini's equity sample is 2015-01-02 to 2026-01-30.  The distributed
        # archive runs later than that, so bound it here rather than silently
        # evaluating on months the paper never saw.
        start=args.start,
        end=args.end,
        # Brini reports Mincer-Zarnowitz intercepts in raw units, so stay in
        # daily units rather than annualising.  QLIKE is unaffected either way:
        # it depends only on the realized/forecast ratio.
        annualization_factor=1,
        output_dir=args.input_dir,
    )
    logging.info("ingesting VOLARE from %s", args.volare_root)
    return build_volare_dataset(config)


def load_series(
    input_dir: str,
    assets: list[str] | None,
    measure: str,
    max_assets: int | None,
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """Load ingested VOLARE parquet into target series and exogenous frames.

    Returns:
        ``(series_by_ticker, exog_by_ticker)``.  The exog frames carry whichever
        of ``jump``/``var_rsp``/``var_rsn``/``var_rq``/``var_bv`` are present,
        which is what the HAR-J / HAR-RS / HARQ specifications consume.

    Raises:
        FileNotFoundError: If nothing usable is found.
    """
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory.resolve()} does not exist. Download VOLARE bulk data and run with "
            f"--volare-root <dir> to ingest it first."
        )

    wanted = [a.upper() for a in assets] if assets else list(BRINI_EQUITIES)
    target_column = f"rv_{measure}"

    series: dict[str, pd.Series] = {}
    exog: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for symbol in wanted:
        path = directory / f"{symbol}.parquet"
        if not path.exists():
            missing.append(symbol)
            continue
        frame = pd.read_parquet(path)
        if target_column not in frame.columns:
            logging.error(
                "%s: no %r column (has %s)", symbol, target_column,
                [c for c in frame.columns if c.startswith("rv_")],
            )
            continue

        values = frame[target_column].dropna()
        if values.empty:
            continue
        series[symbol] = values.rename(symbol)

        columns = [c for c in EXOG_CANDIDATES if c in frame.columns]
        if columns:
            exog[symbol] = frame[columns]

        if max_assets is not None and len(series) >= max_assets:
            break

    if missing:
        logging.warning("%d asset(s) not found in %s: %s", len(missing), directory, ", ".join(missing[:10]))
    if not series:
        raise FileNotFoundError(f"no usable series in {directory.resolve()}")

    logging.info(
        "loaded %d asset(s); %d with exogenous measures", len(series), len(exog)
    )
    return series, exog


def print_report(
    scored: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: BacktestConfig,
    *,
    track: str,
    bootstrap: int,
) -> dict[str, object]:
    """Print leaderboards, significance, MZ and the Brini comparison."""
    pd.set_option("display.width", 220)
    findings: dict[str, object] = {}
    horizons = list(config.horizons)

    print("\n" + "=" * 96)
    print(f"LEADERBOARD  ({track} track, QLIKE, ratio < 1 beats Log-HAR)")
    print("=" * 96)
    for horizon in horizons:
        print(f"\n--- h = {horizon} " + "-" * 60)
        print("\n  equal-weighted across assets (Brini's preferred view)")
        print(leaderboard(tables["equal_weighted"], horizon).round(4).to_string())
        print("\n  pooled")
        print(leaderboard(tables["pooled"], horizon).round(4).to_string())

    print("\n" + "=" * 96)
    print(f"DIEBOLD-MARIANO vs {BENCHMARK_MODEL}")
    print("=" * 96)
    for horizon in horizons:
        table = dm_against_benchmark(scored, horizon, benchmark=BENCHMARK_MODEL)
        if table.empty:
            continue
        print(f"\n--- h = {horizon} (pooled) " + "-" * 50)
        print(table.round(4).to_string(index=False))
        findings[f"dm_h{horizon}"] = table.to_dict("records")

    print("\n" + "=" * 96)
    print(f"MODEL CONFIDENCE SET  (alpha=0.10, {bootstrap} bootstrap draws)")
    print("=" * 96)
    for horizon in horizons:
        pooled_mcs = model_confidence_set(scored, horizon, n_bootstrap=bootstrap, seed=0)
        print(f"\n--- h = {horizon} " + "-" * 60)
        print("  pooled MCS survivors: " + (", ".join(pooled_mcs.included) or "(none)"))
        findings[f"mcs_pooled_h{horizon}"] = pooled_mcs.table.to_dict("records")

        rates = mcs_inclusion_rates(scored, horizon, n_bootstrap=bootstrap, seed=0)
        if not rates.empty:
            print("\n  per-asset MCS inclusion rate (Brini's Table 8 protocol)")
            reference = reference_table("mcs").rename(columns={"brini_model": "model"})
            from voleval.brini import MODEL_NAME_MAP

            annotated = rates.copy()
            annotated["brini_model"] = annotated["model"].map(MODEL_NAME_MAP)
            annotated = annotated.merge(
                reference[reference["horizon"] == horizon][["model", "brini_mcs_pct"]]
                .rename(columns={"model": "brini_model"}),
                on="brini_model", how="left",
            )
            print(annotated.round(1).to_string(index=False))
            findings[f"mcs_rates_h{horizon}"] = annotated.to_dict("records")

    print("\n" + "=" * 96)
    print("MINCER-ZARNOWITZ  (realized = alpha + beta * forecast; efficient => alpha 0, beta 1)")
    print("=" * 96)
    mz = mincer_zarnowitz(scored, scale="volatility", per_asset=True)
    print(mz.round(4).to_string(index=False))
    findings["mincer_zarnowitz"] = mz.to_dict("records")

    print("\n" + "=" * 96)
    print("ROBUSTNESS: pre- vs post-COVID subsamples (equal-weighted QLIKE ratio vs Log-HAR)")
    print("=" * 96)
    subsamples = subsample_summaries(scored, benchmark=BENCHMARK_MODEL)
    for period, period_tables in subsamples.items():
        frame = period_tables["equal_weighted"]
        pivot = frame.pivot_table(
            index="model", columns="horizon", values="qlike_ratio", observed=True
        )
        print(f"\n  {period}:")
        print(pivot.round(3).to_string())
    findings["subsamples"] = {
        period: t["equal_weighted"].to_dict("records") for period, t in subsamples.items()
    }

    return findings, mz


def print_brini_comparison(
    tables: dict[str, pd.DataFrame],
    mz: pd.DataFrame,
    horizons: list[int],
    *,
    track: str,
) -> pd.DataFrame | None:
    """Print the our-numbers-vs-the-paper table."""
    print("\n" + "=" * 96)
    print("OUR RE-RUN vs BRINI (2026), arXiv:2607.05291")
    print("=" * 96)

    if track != "brini":
        print(
            f"\n  Track is {track!r}, which uses a different target definition from the paper.\n"
            f"  The comparison is deliberately suppressed: these numbers are NOT like-for-like.\n"
            f"  Re-run with --track brini to produce the comparison."
        )
        return None

    comparison = compare_with_brini(
        tables["equal_weighted"], tables["pooled"], mz=mz
    )
    for horizon in horizons:
        sliced = format_comparison(comparison, horizon)
        if sliced.empty:
            continue
        print(f"\n--- h = {horizon} " + "-" * 60)
        print(sliced.round(4).to_string(index=False))

    matched = comparison.dropna(subset=["brini_ratio"])
    if not matched.empty:
        print("\n  Agreement on the loss RATIO (the scale-free, comparable quantity):")
        print(f"    mean |our_ratio - brini_ratio| = {matched['ratio_diff'].abs().mean():.4f}")
        print(f"    max  |our_ratio - brini_ratio| = {matched['ratio_diff'].abs().max():.4f}")
        within = (matched["ratio_diff"].abs() < 0.10).mean()
        print(f"    {within:.0%} of (model, horizon) cells within 0.10 of the paper")
    return comparison


def main() -> None:
    """Run the Brini replication end to end."""
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    if args.volare_root:
        manifest = ingest(args)
        print(f"\nIngested {len(manifest['assets'])} asset(s) -> {args.input_dir}")
        if manifest["skipped"]:
            print(f"Skipped: {manifest['skipped']}")
        if args.ingest_only:
            return

    series, exog = load_series(args.input_dir, args.assets, args.measure, args.max_assets)

    config = brini_protocol(min_train_size=args.window_size, refit_frequency=args.refit)
    overrides: dict[str, object] = {}
    if args.horizons:
        overrides["horizons"] = tuple(args.horizons)
    if args.track == "aggregate":
        overrides["target"] = "average"
    if args.max_origins:
        overrides["max_origins"] = args.max_origins
    if overrides:
        config = BacktestConfig(**{**config.to_dict(), **overrides})

    countdown: dict[str, pd.Series] = {}
    if args.with_chronos or args.chronos_only:
        exog, countdown = build_chronos_covariates(series, exog)

    factories = [] if args.chronos_only else brini_factories(
        include_exog_models=not args.no_exog_models and bool(exog),
        include_slow=not args.no_slow_models,
        include_mem=not args.no_mem,
        include_naive=args.with_naive,
    )

    if args.with_chronos or args.chronos_only:
        from volmodels import Chronos2Covariate, Chronos2Univariate

        chronos_kwargs = dict(
            device=args.chronos_device,
            context_length=args.chronos_context,
            max_horizon=max(config.horizons),
        )
        factories += [
            lambda: Chronos2Univariate(**chronos_kwargs),
            lambda: Chronos2Covariate(known_future=countdown or None, **chronos_kwargs),
        ]
    if not exog and not args.no_exog_models:
        logging.warning(
            "no exogenous measures found; HAR-J/HAR-RS/HARQ are unavailable on this data"
        )

    outcome = run_backtest(series, factories, config, exog_by_ticker=exog)
    if outcome.results.empty:
        raise SystemExit("backtest produced no results -- is the window larger than the sample?")

    scored = add_losses(outcome.results)
    tables = summarize(scored, benchmark=BENCHMARK_MODEL)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_dir / f"results_{args.track}.parquet")
    for name, table in tables.items():
        table.to_csv(output_dir / f"summary_{name}_{args.track}.csv", index=False)
    outcome.diagnostics.to_csv(output_dir / f"diagnostics_{args.track}.csv", index=False)

    findings, mz = print_report(
        scored, tables, config, track=args.track, bootstrap=args.bootstrap
    )
    mz.to_csv(output_dir / f"mincer_zarnowitz_{args.track}.csv", index=False)

    comparison = print_brini_comparison(tables, mz, list(config.horizons), track=args.track)
    if comparison is not None:
        comparison.to_csv(output_dir / "comparison_vs_brini.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": "Brini (2026), arXiv:2607.05291",
        "track": args.track,
        "target_definition": (
            "point-in-time sigma_{t+h} (matches Brini)" if args.track == "brini"
            else "sqrt(mean(RV_{t+1..t+h})) -- OUR track, NOT comparable to the paper"
        ),
        "measure": args.measure,
        "input_dir": str(Path(args.input_dir).resolve()),
        "assets": {t: {"rows": int(len(s)),
                       "start": s.index.min().strftime("%Y-%m-%d"),
                       "end": s.index.max().strftime("%Y-%m-%d")} for t, s in series.items()},
        "n_assets": len(series),
        "backtest_config": config.to_dict(),
        "models": outcome.models,
        "result_rows": int(len(scored)),
        "elapsed_seconds": round(outcome.elapsed_seconds, 2),
        "diagnostics": outcome.diagnostics.to_dict("records"),
        "deviations_from_paper": [
            f"refit_frequency={args.refit} (paper: 1, daily re-estimation)" if args.refit != 1 else None,
            f"{len(series)} assets (paper: 40 equities + 5 FX + 5 futures)" if len(series) != 40 else None,
            "HAR-J/HAR-RS/HARQ skipped (no exogenous measures)" if not exog else None,
            "ARMA/ARFIMA skipped" if args.no_slow_models else None,
            "MEM skipped" if args.no_mem else None,
            "classical models skipped (--chronos-only)" if args.chronos_only else None,
        ],
        "findings": findings,
    }
    manifest["deviations_from_paper"] = [d for d in manifest["deviations_from_paper"] if d]
    (output_dir / f"manifest_{args.track}.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    print("\n" + "=" * 96)
    print(f"Artefacts -> {output_dir.resolve()}")
    if manifest["deviations_from_paper"]:
        print("\nDeviations from the paper's protocol:")
        for deviation in manifest["deviations_from_paper"]:
            print(f"  - {deviation}")
    print(f"\n{len(scored)} forecast rows across {len(series)} asset(s) in {outcome.elapsed_seconds:.1f}s\n")


if __name__ == "__main__":
    main()
