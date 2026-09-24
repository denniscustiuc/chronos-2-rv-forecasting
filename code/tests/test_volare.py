"""VOLARE ingest, schema conformance and the Brini-protocol models.

The central claim these tests defend: **VOLARE data and yfinance data are
interchangeable inputs.**  Once ingested, a VOLARE asset is a parquet file with
the same index contract and the same ``rv_*`` / ``var_*`` column convention that
:mod:`volpipe.pipeline` emits, so every existing model and the whole harness run
on it unchanged -- including the lookahead guards.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from volmodels import ARFIMA, ARMA, HARJ, HARQ, HARRS, LogHAR, brini_factories
from volmodels.base import NotEnoughData
from voleval import (
    BRINI_EQUITIES,
    BRINI_HORIZONS,
    BacktestConfig,
    add_losses,
    brini_protocol,
    compare_with_brini,
    mincer_zarnowitz,
    reference_table,
    run_backtest,
    summarize,
)
from voleval.backtest import walk_forward
from voleval.brini import BRINI_RATIO_EQUAL_WEIGHTED, MODEL_NAME_MAP
from volpipe.volare import (
    VolareConfig,
    VolareLoadReport,
    build_volare_dataset,
    discover_symbols,
    load_symbol,
    to_pipeline_schema,
)

from .conftest import simulate_har


# --------------------------------------------------------------------------- #
# Synthetic VOLARE data
# --------------------------------------------------------------------------- #

def make_volare_frame(n_days: int = 400, seed: int = 0, start: str = "2015-01-02") -> pd.DataFrame:
    """Build a frame with VOLARE's column names and plausible internal ratios."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)

    # Daily variance around a 20% annualised level.
    daily_vol = 0.20 / np.sqrt(252)
    rv5 = (daily_vol**2) * np.exp(rng.normal(0.0, 0.5, size=n_days))

    # Bipower variation sits just below RV (the gap is the jump component).
    bv5 = rv5 * rng.uniform(0.80, 0.99, size=n_days)
    share = rng.uniform(0.35, 0.65, size=n_days)
    return pd.DataFrame(
        {
            "date": dates,
            "rv1": rv5 * rng.uniform(0.95, 1.05, size=n_days),
            "rv5": rv5,
            "rv5_ss": rv5 * rng.uniform(0.98, 1.02, size=n_days),
            "bv5": bv5,
            "rq5": rv5**2 * 3.0 * rng.uniform(0.8, 1.2, size=n_days),
            "rsp5": rv5 * share,
            "rsn5": rv5 * (1.0 - share),
            "medrv5": rv5 * rng.uniform(0.9, 1.1, size=n_days),
            "rk": rv5 * rng.uniform(0.95, 1.05, size=n_days),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": rng.integers(1_000_000, 5_000_000, size=n_days).astype("float64"),
        }
    )


def write_native_layout(root, symbols=("AAPL", "MSFT"), n_days: int = 400) -> None:
    """Write VOLARE's native layout: <root>/<SYMBOL>/YYYY_MM_DD.parquet."""
    for i, symbol in enumerate(symbols):
        directory = root / symbol
        directory.mkdir(parents=True, exist_ok=True)
        frame = make_volare_frame(n_days=n_days, seed=i)
        for _, row in frame.iterrows():
            stamp = row["date"]
            row.to_frame().T.to_parquet(directory / f"{stamp:%Y_%m_%d}.parquet")


@pytest.fixture
def volare_root(tmp_path):
    """A small VOLARE tree in the native per-day layout."""
    root = tmp_path / "volare_raw"
    write_native_layout(root, n_days=60)
    return root


@pytest.fixture
def per_asset_root(tmp_path):
    """One concatenated parquet per asset -- the other common download shape."""
    root = tmp_path / "volare_flat"
    root.mkdir()
    for i, symbol in enumerate(("AAPL", "MSFT", "NVDA")):
        make_volare_frame(n_days=500, seed=i).to_parquet(root / f"{symbol}.parquet")
    return root


# --------------------------------------------------------------------------- #
# Layout discovery
# --------------------------------------------------------------------------- #

def test_discovers_the_native_per_day_layout(volare_root):
    found = discover_symbols(volare_root)
    assert set(found) == {"AAPL", "MSFT"}
    assert all(len(paths) == 60 for paths in found.values())


def test_discovers_the_per_asset_layout(per_asset_root):
    found = discover_symbols(per_asset_root)
    assert set(found) == {"AAPL", "MSFT", "NVDA"}
    assert all(len(paths) == 1 for paths in found.values())


def test_discovers_an_asset_class_nested_layout(tmp_path):
    root = tmp_path / "nested"
    write_native_layout(root / "stocks", symbols=("AAPL",), n_days=20)
    found = discover_symbols(root)
    assert "AAPL" in found


def test_discovers_a_combined_file(tmp_path):
    root = tmp_path / "combined"
    root.mkdir()
    frames = []
    for i, symbol in enumerate(("AAPL", "MSFT")):
        frame = make_volare_frame(n_days=300, seed=i)
        frame["symbol"] = symbol
        frames.append(frame)
    pd.concat(frames).to_parquet(root / "all_assets.parquet")

    assert "__COMBINED__" in discover_symbols(root)


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_symbols(tmp_path / "nope")


def test_empty_root_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no parquet/csv"):
        discover_symbols(empty)


# --------------------------------------------------------------------------- #
# Schema conformance -- the core contract
# --------------------------------------------------------------------------- #

def test_schema_matches_the_volpipe_contract(per_asset_root):
    found = discover_symbols(per_asset_root)
    config = VolareConfig(root=str(per_asset_root), measure="rv5")
    frame, report = load_symbol("AAPL", found["AAPL"], config)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.name == "date"
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    assert frame.index.tz is None

    # The volpipe convention: rv_* are volatilities, var_* are variances.
    assert "rv_rv5" in frame.columns
    assert "var_rv5" in frame.columns
    assert np.allclose(frame["rv_rv5"] ** 2, frame["var_rv5"], rtol=1e-10)

    for measure in ("bv", "rq", "rsp", "rsn"):
        assert f"var_{measure}" in frame.columns, measure
    assert "jump" in frame.columns
    assert report.measures_missing == [] or "minrv" in report.measures_missing


def test_target_column_is_positive_volatility(per_asset_root):
    found = discover_symbols(per_asset_root)
    config = VolareConfig(root=str(per_asset_root))
    frame, _ = load_symbol("AAPL", found["AAPL"], config)

    assert (frame["rv_rv5"] > 0).all()
    assert frame["rv_rv5"].notna().all()
    # Daily volatility of a ~20% annualised asset.
    assert 0.002 < frame["rv_rv5"].median() < 0.06


def test_jump_component_is_non_negative_and_matches_rv_minus_bv(per_asset_root):
    found = discover_symbols(per_asset_root)
    config = VolareConfig(root=str(per_asset_root))
    frame, _ = load_symbol("AAPL", found["AAPL"], config)

    assert (frame["jump"] >= 0).all()
    expected = (frame["var_rv5"] - frame["var_bv"]).clip(lower=0.0)
    np.testing.assert_allclose(frame["jump"], expected, rtol=1e-12)


def test_semivariances_sum_to_the_realized_variance(per_asset_root):
    found = discover_symbols(per_asset_root)
    config = VolareConfig(root=str(per_asset_root))
    frame, _ = load_symbol("AAPL", found["AAPL"], config)
    np.testing.assert_allclose(
        frame["var_rsp"] + frame["var_rsn"], frame["var_rv5"], rtol=1e-10
    )


def test_annualization_scales_variance_and_volatility_consistently(per_asset_root):
    found = discover_symbols(per_asset_root)
    daily = VolareConfig(root=str(per_asset_root), annualization_factor=1)
    annual = VolareConfig(root=str(per_asset_root), annualization_factor=252)

    frame_d, _ = load_symbol("AAPL", found["AAPL"], daily)
    frame_a, _ = load_symbol("AAPL", found["AAPL"], annual)

    np.testing.assert_allclose(frame_a["var_rv5"], frame_d["var_rv5"] * 252, rtol=1e-10)
    np.testing.assert_allclose(frame_a["rv_rv5"], frame_d["rv_rv5"] * np.sqrt(252), rtol=1e-10)


def test_date_range_filters_are_applied(per_asset_root):
    found = discover_symbols(per_asset_root)
    config = VolareConfig(root=str(per_asset_root), start="2015-06-01", end="2016-01-01")
    frame, report = load_symbol("AAPL", found["AAPL"], config)

    assert frame.index.min() >= pd.Timestamp("2015-06-01")
    assert frame.index.max() <= pd.Timestamp("2016-01-01")
    assert report.dropped.get("before_start", 0) > 0


def test_negative_variances_are_dropped():
    raw = make_volare_frame(n_days=50).set_index("date")
    raw.loc[raw.index[5], "rv5"] = -1.0
    raw.loc[raw.index[6], "rv5"] = np.nan

    report = VolareLoadReport(symbol="TEST")
    config = VolareConfig(root=".", min_observations=1)
    frame = to_pipeline_schema(raw, config, report)

    assert len(frame) == 48
    assert report.dropped["missing_or_nonpositive_target"] == 2


def test_missing_target_column_raises():
    raw = make_volare_frame(n_days=20).set_index("date").drop(columns=["rv5", "rv5_ss"])
    report = VolareLoadReport(symbol="TEST")
    with pytest.raises(KeyError, match="no column supplying"):
        to_pipeline_schema(raw, VolareConfig(root="."), report)


def test_unknown_measure_is_rejected():
    with pytest.raises(ValueError, match="unknown measure"):
        VolareConfig(root=".", measure="not_a_measure")


# --------------------------------------------------------------------------- #
# Dataset build + manifest
# --------------------------------------------------------------------------- #

def test_build_writes_parquet_panel_and_manifest(per_asset_root, tmp_path):
    output = tmp_path / "out"
    config = VolareConfig(
        root=str(per_asset_root), output_dir=str(output), min_observations=100
    )
    manifest = build_volare_dataset(config)

    for symbol in ("AAPL", "MSFT", "NVDA"):
        assert (output / f"{symbol}.parquet").exists()
    assert (output / "panel.parquet").exists()
    assert (output / "manifest.json").exists()

    assert set(manifest["assets"]) == {"AAPL", "MSFT", "NVDA"}
    assert manifest["panel_rows"] == 3 * 500
    assert "UNADJUSTED" in manifest["price_basis"]
    json.loads((output / "manifest.json").read_text())


def test_panel_has_the_same_multiindex_as_volpipe(per_asset_root, tmp_path):
    output = tmp_path / "out"
    build_volare_dataset(
        VolareConfig(root=str(per_asset_root), output_dir=str(output), min_observations=100)
    )
    panel = pd.read_parquet(output / "panel.parquet")

    assert list(panel.index.names) == ["date", "ticker"]
    assert panel.index.is_monotonic_increasing
    assert set(panel.index.get_level_values("ticker").unique()) == {"AAPL", "MSFT", "NVDA"}


def test_short_assets_are_skipped_not_fatal(per_asset_root, tmp_path):
    output = tmp_path / "out"
    manifest = build_volare_dataset(
        VolareConfig(root=str(per_asset_root), output_dir=str(output), min_observations=10_000)
    )
    assert manifest["assets"] == {}
    assert len(manifest["skipped"]) == 3


def test_requested_but_absent_symbols_are_reported(per_asset_root, tmp_path):
    manifest = build_volare_dataset(
        VolareConfig(
            root=str(per_asset_root), symbols=("AAPL", "NOTREAL"),
            output_dir=str(tmp_path / "out"), min_observations=100,
        )
    )
    assert "AAPL" in manifest["assets"]
    assert "NOTREAL" in manifest["skipped"]


# --------------------------------------------------------------------------- #
# The harness runs on VOLARE unchanged, including the lookahead guards
# --------------------------------------------------------------------------- #

@pytest.fixture
def volare_series(per_asset_root, tmp_path):
    """Ingested VOLARE data, loaded as (series, exog) the way the runner does."""
    output = tmp_path / "ingested"
    build_volare_dataset(
        VolareConfig(root=str(per_asset_root), output_dir=str(output), min_observations=100)
    )
    series, exog = {}, {}
    for symbol in ("AAPL", "MSFT"):
        frame = pd.read_parquet(output / f"{symbol}.parquet")
        series[symbol] = frame["rv_rv5"]
        exog[symbol] = frame[["jump", "var_rsp", "var_rsn", "var_rq", "var_bv"]]
    return series, exog


def test_existing_models_run_on_volare_unchanged(volare_series):
    series, exog = volare_series
    config = BacktestConfig(horizons=(1, 5, 22), min_train_size=250, refit_frequency=25)
    outcome = run_backtest(series, [LogHAR], config, exog_by_ticker=exog)

    assert not outcome.results.empty
    assert set(outcome.results["ticker"]) == {"AAPL", "MSFT"}
    assert outcome.results["forecast"].notna().all()
    assert (outcome.diagnostics["failed_fits"] == 0).all()


def test_exog_models_run_on_volare(volare_series):
    series, exog = volare_series
    config = BacktestConfig(horizons=(1, 22), min_train_size=250, refit_frequency=50)
    outcome = run_backtest(
        {"AAPL": series["AAPL"]}, [HARJ, HARRS, HARQ], config,
        exog_by_ticker={"AAPL": exog["AAPL"]},
    )

    assert set(outcome.results["model"]) == {"har_j", "har_rs", "harq"}
    assert outcome.results["forecast"].notna().all()
    assert (outcome.results["forecast"] > 0).all()


def test_exog_slice_never_extends_past_the_origin(volare_series):
    """The exogenous channel inherits the target's no-lookahead guarantee."""
    series, exog = volare_series
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    class ExogSpy(HARJ):
        def fit(self, history, exog_frame=None):
            if exog_frame is not None and len(exog_frame.dropna(how="all")):
                seen.append((exog_frame.dropna(how="all").index.max(), history.index.max()))
            super().fit(history, exog_frame)

    config = BacktestConfig(horizons=(1,), min_train_size=250, refit_frequency=1)
    walk_forward(series["AAPL"], [lambda: ExogSpy()], config, ticker="AAPL", exog=exog["AAPL"])

    assert seen
    for exog_max, origin in seen:
        assert exog_max <= origin, f"exog reached {exog_max}, past origin {origin}"


def test_model_rejects_exog_that_reaches_past_the_origin(volare_series):
    """A model is the last line of defence and refuses future-dated exog."""
    series, exog = volare_series
    history = series["AAPL"].iloc[:300]
    future_exog = exog["AAPL"].iloc[:400]  # extends 100 days past the history

    with pytest.raises(ValueError, match="lookahead"):
        HARJ().fit(history, future_exog)


def test_poisoning_the_future_cannot_change_past_forecasts_on_volare(volare_series):
    series, exog = volare_series
    clean_series, clean_exog = series["AAPL"], exog["AAPL"]

    cutoff = 350
    dirty_series = clean_series.copy()
    dirty_series.iloc[cutoff:] = 9.0
    dirty_exog = clean_exog.copy()
    dirty_exog.iloc[cutoff:] = 9.0

    config = BacktestConfig(horizons=(1, 5), min_train_size=250, refit_frequency=10)
    factories = [LogHAR, HARJ, HARRS, HARQ]

    clean_results, _ = walk_forward(
        clean_series, factories, config, ticker="AAPL", exog=clean_exog
    )
    dirty_results, _ = walk_forward(
        dirty_series, factories, config, ticker="AAPL", exog=dirty_exog
    )

    boundary = clean_series.index[cutoff]
    keys = ["model", "origin_date", "horizon"]
    before_clean = clean_results[clean_results["origin_date"] < boundary].set_index(keys)["forecast"]
    before_dirty = dirty_results[dirty_results["origin_date"] < boundary].set_index(keys)["forecast"]

    assert len(before_clean) > 0
    pd.testing.assert_series_equal(before_clean.sort_index(), before_dirty.sort_index())


# --------------------------------------------------------------------------- #
# Target definitions: Brini's point-in-time vs our aggregate track
# --------------------------------------------------------------------------- #

def test_point_target_is_the_value_at_t_plus_h():
    series = simulate_har(n_days=400, seed=1)
    config = BacktestConfig(horizons=(5,), min_train_size=250, target="point")
    results, _ = walk_forward(series, [LogHAR], config, ticker="X")

    positions = {date: i for i, date in enumerate(series.index)}
    row = results.iloc[0]
    assert row["realized"] == pytest.approx(float(series.iloc[positions[row["origin_date"]] + 5]))


def test_average_target_is_the_root_mean_variance_over_the_horizon():
    series = simulate_har(n_days=400, seed=1)
    config = BacktestConfig(horizons=(5,), min_train_size=250, target="average")
    results, _ = walk_forward(series, [LogHAR], config, ticker="X")

    positions = {date: i for i, date in enumerate(series.index)}
    row = results.iloc[0]
    start = positions[row["origin_date"]] + 1
    expected = np.sqrt(np.mean(series.iloc[start : start + 5].to_numpy() ** 2))
    assert row["realized"] == pytest.approx(expected)


def test_the_two_targets_agree_at_h_equals_one():
    series = simulate_har(n_days=300, seed=2)
    point = BacktestConfig(horizons=(1,), min_train_size=250, target="point")
    average = BacktestConfig(horizons=(1,), min_train_size=250, target="average")

    a, _ = walk_forward(series, [LogHAR], point, ticker="X")
    b, _ = walk_forward(series, [LogHAR], average, ticker="X")
    np.testing.assert_allclose(a["realized"], b["realized"], rtol=1e-12)


def test_average_target_is_smoother_than_the_point_target():
    """Averaging over the horizon is exactly what removes the daily noise."""
    series = simulate_har(n_days=600, seed=3)
    point = BacktestConfig(horizons=(22,), min_train_size=250, target="point")
    average = BacktestConfig(horizons=(22,), min_train_size=250, target="average")

    a, _ = walk_forward(series, [LogHAR], point, ticker="X")
    b, _ = walk_forward(series, [LogHAR], average, ticker="X")
    assert b["realized"].std() < a["realized"].std()


def test_invalid_target_is_rejected():
    with pytest.raises(ValueError, match="target must be"):
        BacktestConfig(target="median")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Brini protocol and reference tables
# --------------------------------------------------------------------------- #

def test_brini_protocol_matches_the_paper():
    config = brini_protocol()
    assert config.horizons == (1, 5, 22)
    assert config.window == "rolling"
    assert config.min_train_size == 1000
    assert config.rolling_window_size == 1000
    assert config.refit_frequency == 1
    assert config.target == "point"


def test_brini_asset_universe():
    assert len(BRINI_EQUITIES) == 40
    assert len(set(BRINI_EQUITIES)) == 40
    for expected in ("AAPL", "AMZN", "MSFT", "XOM", "NVDA"):
        assert expected in BRINI_EQUITIES


def test_reference_tables_are_complete_and_consistent():
    ratios = reference_table("ratio")
    assert set(ratios["horizon"]) == set(BRINI_HORIZONS)

    # Log-HAR is the benchmark, so its own ratio must be exactly 1 everywhere.
    benchmark = ratios[ratios["brini_model"] == "Log-HAR"]
    assert (benchmark["brini_ratio"] == 1.0).all()

    # Every econometric model we implement has a published ratio to compare to.
    for our_name, paper_name in MODEL_NAME_MAP.items():
        assert paper_name in BRINI_RATIO_EQUAL_WEIGHTED, our_name

    # TTM is the paper's headline: the only model below 1 at every horizon.
    ttm = BRINI_RATIO_EQUAL_WEIGHTED["TTM"]
    assert all(value < 1.0 for value in ttm.values())


def test_reference_table_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown reference table"):
        reference_table("nonsense")


def test_comparison_joins_our_numbers_to_the_paper(volare_series):
    series, exog = volare_series
    config = BacktestConfig(horizons=(1, 5), min_train_size=250, refit_frequency=50)
    outcome = run_backtest(series, [LogHAR, HARJ], config, exog_by_ticker=exog)
    tables = summarize(add_losses(outcome.results))

    comparison = compare_with_brini(tables["equal_weighted"], tables["pooled"])
    assert {"our_ratio", "brini_ratio", "ratio_diff"} <= set(comparison.columns)

    log_har = comparison[(comparison["model"] == "log_har") & (comparison["horizon"] == 1)]
    assert log_har["our_ratio"].iloc[0] == pytest.approx(1.0)
    assert log_har["brini_ratio"].iloc[0] == pytest.approx(1.0)


def test_comparison_keeps_models_the_paper_does_not_report():
    """Our naive floors have no Brini counterpart; they must not be dropped."""
    ours = pd.DataFrame(
        {
            "model": ["log_har", "random_walk"],
            "horizon": [1, 1],
            "qlike": [0.2, 0.3],
            "qlike_ratio": [1.0, 1.5],
        }
    )
    comparison = compare_with_brini(ours)
    assert set(comparison["model"]) == {"log_har", "random_walk"}
    assert pd.isna(comparison.loc[comparison["model"] == "random_walk", "brini_ratio"]).all()


# --------------------------------------------------------------------------- #
# Mincer-Zarnowitz
# --------------------------------------------------------------------------- #

def test_mz_on_a_perfect_forecast_gives_alpha_zero_beta_one():
    rng = np.random.default_rng(0)
    realized = rng.uniform(0.01, 0.05, size=300)
    frame = pd.DataFrame(
        {
            "model": "oracle", "ticker": "AAA", "horizon": 1,
            "forecast": realized, "realized": realized,
        }
    )
    row = mincer_zarnowitz(frame).iloc[0]
    assert row["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert row["beta"] == pytest.approx(1.0, abs=1e-12)
    assert row["r2"] == pytest.approx(1.0, abs=1e-12)


def test_mz_detects_a_systematically_over_reacting_forecast():
    """A forecast that over-reacts by 2x must produce beta ~ 0.5."""
    rng = np.random.default_rng(1)
    mean = 0.03
    realized = mean + rng.normal(0.0, 0.005, size=2000)
    forecast = mean + 2.0 * (realized - mean)

    frame = pd.DataFrame(
        {
            "model": "overreacting", "ticker": "AAA", "horizon": 1,
            "forecast": forecast, "realized": realized,
        }
    )
    row = mincer_zarnowitz(frame).iloc[0]
    assert row["beta"] == pytest.approx(0.5, abs=0.01)
    assert row["r2"] == pytest.approx(1.0, abs=0.01)


def test_mz_per_asset_averaging_differs_from_pooling():
    rng = np.random.default_rng(2)
    rows = []
    for ticker, level in (("AAA", 0.01), ("BBB", 0.08)):
        realized = level * np.exp(rng.normal(0, 0.3, size=400))
        rows.append(
            pd.DataFrame(
                {
                    "model": "m", "ticker": ticker, "horizon": 1,
                    "forecast": realized * np.exp(rng.normal(0, 0.1, size=400)),
                    "realized": realized,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)

    per_asset = mincer_zarnowitz(frame, per_asset=True).iloc[0]
    pooled = mincer_zarnowitz(frame, per_asset=False).iloc[0]
    assert per_asset["n_assets"] == 2
    assert per_asset["beta"] != pytest.approx(pooled["beta"], abs=1e-9)


def test_mz_scales():
    rng = np.random.default_rng(3)
    realized = rng.uniform(0.01, 0.05, size=200)
    frame = pd.DataFrame(
        {
            "model": "m", "ticker": "AAA", "horizon": 1,
            "forecast": realized, "realized": realized,
        }
    )
    for scale in ("volatility", "variance", "log"):
        row = mincer_zarnowitz(frame, scale=scale).iloc[0]
        assert row["beta"] == pytest.approx(1.0, abs=1e-9), scale

    with pytest.raises(ValueError, match="scale must be"):
        mincer_zarnowitz(frame, scale="nope")


# --------------------------------------------------------------------------- #
# The new econometric models
# --------------------------------------------------------------------------- #

def test_brini_factories_cover_all_eight_econometric_benchmarks():
    names = [f().name for f in brini_factories()]
    assert set(names) == {"har", "log_har", "har_j", "har_rs", "harq", "arma", "arfima", "mem"}
    assert len(names) == 8, "all eight of Brini's econometric benchmarks must be registered"


def test_exog_models_fail_cleanly_without_their_measures():
    series = simulate_har(n_days=400, seed=4)
    for model_class in (HARJ, HARRS, HARQ):
        model = model_class()
        with pytest.raises(NotEnoughData, match="exogenous|column"):
            model.fit(series)
            model.forecast(1)


def test_arma_and_arfima_forecast_sensibly():
    series = simulate_har(n_days=500, seed=5)
    for model_class in (ARMA, ARFIMA):
        model = model_class()
        model.fit(series)
        for h in BRINI_HORIZONS:
            point = model.forecast(h).point
            assert point > 0
            assert series.min() * 0.5 < point < series.max() * 2.0


@pytest.mark.parametrize("true_d", [0.0, 0.1, 0.2, 0.3])
def test_gph_recovers_a_known_memory_parameter(true_d):
    """GPH must recover ``d`` on a series with that memory built in.

    The test series is fractionally integrated white noise -- applying
    ``(1-L)^(-d)`` to iid noise -- which is ARFIMA(0, d, 0) by construction, so
    the true answer is known exactly.  Averaging over seeds controls the
    estimator's own sampling noise (sd ~ 0.07 per series).

    Values of ``d`` near 0.5 are deliberately excluded: there the estimate runs
    into the stationarity clip and the *mean* is biased down by censoring, which
    would be testing the clip rather than the estimator.
    """
    from volmodels.classical import fractional_difference, gph_estimate_d

    estimates = []
    for seed in range(12):
        noise = np.random.default_rng(seed).normal(size=4000)
        integrated = fractional_difference(noise, -true_d, truncation=1000)[1000:]
        estimates.append(gph_estimate_d(integrated))

    assert np.mean(estimates) == pytest.approx(true_d, abs=0.05), (
        f"mean estimate {np.mean(estimates):.3f} for true d={true_d}"
    )


def test_gph_clips_a_non_stationary_series_to_the_bound():
    """A random walk has d = 0.5+; the estimate must clip, not blow up."""
    from volmodels.classical import gph_estimate_d

    walk = np.cumsum(np.random.default_rng(0).normal(size=3000))
    assert gph_estimate_d(walk) == pytest.approx(0.499, abs=1e-9)


def test_arfima_estimates_a_memory_parameter():
    model = ARFIMA()
    model.fit(simulate_har(n_days=800, seed=7))
    assert model.d is not None
    assert 0.0 <= model.d <= 0.499


def test_fractional_difference_at_d_zero_is_the_identity():
    from volmodels.classical import fractional_difference

    values = np.linspace(1.0, 5.0, 100)
    np.testing.assert_allclose(fractional_difference(values, 0.0), values, rtol=1e-12)


def test_fractional_difference_at_d_one_is_the_first_difference():
    from volmodels.classical import fractional_difference

    values = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(fractional_difference(values, 1.0), expected, rtol=1e-12)


# --------------------------------------------------------------------------- #
# The insanity filter (Bollerslev, Patton & Quaedvlieg 2016)
# --------------------------------------------------------------------------- #

def test_insanity_filter_replaces_an_implausible_forecast_with_the_mean():
    """A negative or absurd point forecast falls back to the in-sample mean.

    Regression test for a real failure: in a 7,200-forecast dry run, a single
    HARQ forecast floored at 1e-8 produced a mean QLIKE of 1.0e10 -- one
    observation in ten thousand swamping the whole metric, because QLIKE
    diverges as the forecast approaches zero.
    """
    from volmodels.base import Forecast, Forecaster

    class Wild(Forecaster):
        """Returns whatever it is told to, so the filter can be tested directly."""

        def __init__(self, value: float) -> None:
            super().__init__("wild", insanity_filter=True)
            self.value = value

        def fit(self, history, exog=None):
            self._store_history(history, exog)

        def forecast(self, h: int) -> Forecast:
            return Forecast(
                point=self._floor(self._sanitize(self.value)),
                model=self.name, origin=self.origin, horizon=h,
            )

    history = pd.Series(
        np.full(100, 0.02), index=pd.bdate_range("2020-01-01", periods=100)
    )

    for bad in (-1.0, 0.0, np.nan, 1e6):
        model = Wild(bad)
        model.fit(history)
        assert model.forecast(1).point == pytest.approx(0.02), bad
        assert model.n_insane == 1

    sane = Wild(0.03)
    sane.fit(history)
    assert sane.forecast(1).point == pytest.approx(0.03)
    assert sane.n_insane == 0


def test_log_har_does_not_carry_the_insanity_filter():
    """Log-HAR cannot produce a negative forecast, so it needs no filter."""
    from volmodels import HAR

    assert LogHAR().insanity_filter is False
    assert HAR().insanity_filter is True
    for model_class in (HARJ, HARRS, HARQ):
        assert model_class().insanity_filter is True, model_class.__name__


def test_insanity_filtered_forecasts_are_counted_in_diagnostics(volare_series):
    series, exog = volare_series
    config = BacktestConfig(horizons=(1,), min_train_size=250, refit_frequency=25)
    outcome = run_backtest(
        {"AAPL": series["AAPL"]}, [HARQ], config, exog_by_ticker={"AAPL": exog["AAPL"]}
    )
    assert "insanity_filtered" in outcome.diagnostics.columns
    assert outcome.diagnostics["insanity_filtered"].iloc[0] >= 0


def test_a_single_floored_forecast_would_dominate_qlike():
    """Documents *why* the filter exists, in one assertion."""
    from voleval.metrics import qlike

    realized = np.full(1000, 0.02)
    forecast = np.full(1000, 0.02)
    assert qlike(realized, forecast).mean() == pytest.approx(0.0, abs=1e-12)

    forecast[0] = 1e-8  # one floored forecast in a thousand
    assert qlike(realized, forecast).mean() > 1e6
