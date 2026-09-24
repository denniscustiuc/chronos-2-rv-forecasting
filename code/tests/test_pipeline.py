"""Pipeline assembly, schema, manifest and lookahead tests (no network access)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from volpipe import validation
from volpipe.config import PipelineConfig
from volpipe.pipeline import PipelineResult, _build_manifest, _build_panel, build_ticker_frame

from .conftest import simulate_ohlc


@pytest.fixture
def config(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        tickers=("TEST", "OTHER"),
        start="2015-01-01",
        end="2021-01-01",
        rv_window=21,
        estimators=("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"),
        covariates=(),
        output_dir=str(tmp_path / "out"),
        write_csv=True,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return simulate_ohlc(n_days=400, seed=2)


def build(config, ohlcv, ticker="TEST"):
    return build_ticker_frame(ticker, config, ohlcv=ohlcv, fetch_covariates=False)


def test_output_schema(config, ohlcv):
    result = build(config, ohlcv)
    frame = result.frame

    expected_rv = ["rv_cc", "rv_parkinson", "rv_gk", "rv_rs", "rv_yz"]
    expected_var = ["var_cc", "var_parkinson", "var_gk", "var_rs", "var_yz"]
    assert list(frame.columns) == expected_rv + expected_var
    assert frame.index.name == "date"
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    assert result.estimator_columns["garman_klass"] == "rv_gk"


def test_daily_variance_can_be_suppressed(config, ohlcv):
    lean = PipelineConfig(**{**config.to_dict(), "tickers": ("TEST",), "include_daily_variance": False,
                             "estimators": tuple(config.estimators), "covariates": ()})
    frame = build(lean, ohlcv).frame
    assert all(c.startswith("rv_") for c in frame.columns)


def test_warmup_rows_are_dropped(config, ohlcv):
    result = build(config, ohlcv)
    assert len(result.frame) == len(ohlcv) - (config.rv_window - 1)
    assert result.frame["rv_parkinson"].notna().all()
    assert any("warm-up" in note for note in result.notes)


def test_warmup_rows_are_kept_when_disabled(config, ohlcv):
    keep = PipelineConfig(**{**config.to_dict(), "tickers": ("TEST",), "dropna_rv": False,
                             "estimators": tuple(config.estimators), "covariates": ()})
    frame = build(keep, ohlcv).frame
    assert len(frame) == len(ohlcv)
    assert frame["rv_parkinson"].iloc[0] != frame["rv_parkinson"].iloc[0]  # NaN


def test_all_rv_values_are_positive_and_plausible(config, ohlcv):
    frame = build(config, ohlcv).frame
    for col in [c for c in frame.columns if c.startswith("rv_")]:
        series = frame[col].dropna()
        assert (series > 0).all(), col
        assert series.max() < 5.0, f"{col}: implausible 500%+ annualised vol"


def test_pipeline_output_has_no_lookahead(config, ohlcv):
    """Rebuilding on a truncated history must reproduce the same values at t."""
    full = build(config, ohlcv).frame
    cutoff_date = ohlcv.index[300]
    truncated = build(config, ohlcv.loc[:cutoff_date]).frame

    overlap = truncated.index
    pd.testing.assert_frame_equal(full.loc[overlap], truncated.loc[overlap])


def test_earnings_flag_is_the_only_known_future_column(config, ohlcv, monkeypatch):
    from volpipe import covariates as cov

    index_holder = {}

    def fake_vix(index, cfg):
        index_holder["index"] = index
        return pd.Series(20.0, index=index, name="vix")

    monkeypatch.setattr(cov, "vix_covariate", fake_vix)
    monkeypatch.setattr(cov, "_fetch_earnings_dates", lambda *a, **k: pd.DatetimeIndex([ohlcv.index[100]]))

    with_cov = PipelineConfig(
        **{**config.to_dict(), "tickers": ("TEST",), "covariates": ("vix", "volume", "earnings_flag"),
           "estimators": tuple(config.estimators)}
    )
    result = build_ticker_frame("TEST", with_cov, ohlcv=ohlcv, fetch_covariates=True)

    known_future = [c for c, tag in result.covariate_tags.items() if tag == "known_future"]
    assert known_future == ["earnings_flag"]
    assert set(result.covariate_tags) == {"vix", "volume", "earnings_flag"}
    assert result.frame["earnings_flag"].sum() > 0


def test_empty_history_is_skipped_not_fatal(config):
    result = build_ticker_frame("TEST", config, ohlcv=pd.DataFrame(), fetch_covariates=False)
    assert result.frame.empty
    assert any("no usable price history" in note for note in result.notes)


def test_panel_has_date_ticker_multiindex(config, ohlcv):
    results = {
        "TEST": build(config, ohlcv, "TEST"),
        "OTHER": build(config, simulate_ohlc(n_days=400, seed=9), "OTHER"),
    }
    panel = _build_panel(results)

    assert list(panel.index.names) == ["date", "ticker"]
    assert set(panel.index.get_level_values("ticker").unique()) == {"TEST", "OTHER"}
    assert len(panel) == sum(len(r.frame) for r in results.values())
    assert panel.index.is_monotonic_increasing

    slice_test = panel.xs("TEST", level="ticker")
    pd.testing.assert_frame_equal(slice_test, results["TEST"].frame, check_names=False, check_freq=False)


def test_panel_is_empty_when_nothing_succeeded(config):
    empty = build_ticker_frame("TEST", config, ohlcv=pd.DataFrame(), fetch_covariates=False)
    assert _build_panel({"TEST": empty}).empty


def test_manifest_contents(config, ohlcv):
    results = {"TEST": build(config, ohlcv, "TEST")}
    panel = _build_panel(results)
    manifest = _build_manifest(config, results, panel)

    assert manifest["config"]["rv_window"] == 21
    assert manifest["config"]["annualization_factor"] == 252
    assert manifest["row_counts"]["TEST"] == len(results["TEST"].frame)
    assert manifest["panel_rows"] == len(panel)
    assert manifest["date_range"]["covered_start"] is not None
    assert manifest["estimators"]["garman_klass"]["column_rv"] == "rv_gk"
    assert "known_future_exception" in manifest["lookahead_policy"]
    assert "UNADJUSTED" in manifest["lookahead_policy"]["price_adjustment"]
    json.dumps(manifest, default=str)  # must stay serialisable


def test_manifest_records_dropped_gaps(config, ohlcv):
    from volpipe.ingest import DataQualityReport

    result = build(config, ohlcv)
    result.quality = DataQualityReport(ticker="TEST", rows_downloaded=401, rows_kept=400)
    result.quality.record_drop("missing_ohlc", pd.DatetimeIndex(["2016-03-04"]))

    manifest = _build_manifest(config, {"TEST": result}, _build_panel({"TEST": result}))
    assert manifest["gaps_dropped"]["TEST"]["dropped_by_reason"] == {"missing_ohlc": 1}


def test_save_writes_parquet_csv_and_manifest(config, ohlcv, tmp_path):
    results = {"TEST": build(config, ohlcv, "TEST")}
    panel = _build_panel(results)
    result = PipelineResult(
        config=config, tickers=results, panel=panel,
        manifest=_build_manifest(config, results, panel),
    )
    written = result.save()

    names = {p.split("/")[-1] for p in written}
    assert {"TEST.parquet", "TEST.csv", "panel.parquet", "manifest.json"} <= names

    round_trip = pd.read_parquet([p for p in written if p.endswith("TEST.parquet")][0])
    # check_freq=False: the synthetic fixture's bdate_range carries a freq that
    # parquet does not preserve; real yfinance indexes have no freq to begin with.
    pd.testing.assert_frame_equal(round_trip, results["TEST"].frame, check_freq=False)

    manifest_path = [p for p in written if p.endswith("manifest.json")][0]
    assert json.loads(open(manifest_path).read())["row_counts"]["TEST"] == len(results["TEST"].frame)


def test_intraday_estimator_degrades_within_the_pipeline(config, ohlcv):
    with_intraday = PipelineConfig(
        **{**config.to_dict(), "tickers": ("TEST",), "covariates": (),
           "estimators": ("parkinson", "intraday_rv")}
    )
    result = build_ticker_frame("TEST", with_intraday, ohlcv=ohlcv, intraday=None, fetch_covariates=False)

    assert "rv_intraday" in result.frame.columns
    assert result.frame["rv_intraday"].isna().all()
    assert result.frame["rv_parkinson"].notna().any()
    assert any("intraday_rv" in note for note in result.notes)


def test_validation_helpers(config, ohlcv, tmp_path):
    frame = build(config, ohlcv).frame

    corr = validation.estimator_correlation_matrix(frame)
    assert corr.shape == (5, 5)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)

    ranking = validation.estimator_noise_ranking(frame)
    assert "noise_ratio" in ranking.columns
    assert ranking["noise_ratio"].is_monotonic_decreasing

    plot_path = validation.plot_estimators(frame, "TEST", tmp_path / "plot.png")
    assert plot_path is not None and plot_path.exists()


def test_volare_comparison_stub(config, ohlcv):
    frame = build(config, ohlcv).frame
    assert validation.compare_with_volare(frame)["status"] == "stub"


def test_volare_comparison_with_supplied_series(config, ohlcv, tmp_path):
    frame = build(config, ohlcv).frame
    # Stand in for VOLARE rv5: a daily variance proxy on the same dates.
    fake_rv5 = (frame["rv_gk"] ** 2 / 252) * 1.05
    stats = validation.compare_with_volare(
        frame, fake_rv5, column="rv_gk", output_path=tmp_path / "scatter.png"
    )
    assert stats["status"] == "ok"
    assert stats["pearson"] > 0.99
    assert stats["mean_ratio_ours_over_volare"] == pytest.approx(1 / np.sqrt(1.05), rel=1e-6)
