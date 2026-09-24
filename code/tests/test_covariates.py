"""Covariate alignment, tagging and lookahead tests (network calls stubbed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volpipe import covariates as cov
from volpipe.config import PipelineConfig


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(
        tickers=("TEST",),
        start="2020-01-01",
        end="2020-04-01",
        rv_window=5,
        covariates=("vix", "volume", "earnings_flag"),
        earnings_window_days=1,
    )


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=40, name="date")


def test_align_forward_fills_and_never_back_fills(index):
    """A value observed at t must not be visible before t."""
    sparse = pd.Series(
        [10.0, 20.0],
        index=pd.DatetimeIndex([index[5], index[20]]),
    )
    aligned = cov._align(sparse, index, name="x")

    assert aligned.iloc[:5].isna().all(), "value leaked backwards before its first observation"
    assert (aligned.iloc[5:20] == 10.0).all()
    assert (aligned.iloc[20:] == 20.0).all()


def test_align_handles_empty_series(index):
    aligned = cov._align(pd.Series(dtype="float64"), index, name="x")
    assert len(aligned) == len(index)
    assert aligned.isna().all()


def test_vix_is_aligned_to_the_ticker_index(monkeypatch, config, index):
    fake = pd.Series(
        np.linspace(12.0, 40.0, len(index)),
        index=index.tz_localize("UTC"),  # exercise the tz-normalisation path
        name="^VIX",
    )
    monkeypatch.setattr(cov, "download_series_close", lambda *a, **k: fake)

    series = cov.vix_covariate(index, config)
    assert series.name == "vix"
    assert series.index.equals(index)
    assert series.notna().all()


def test_vix_missing_download_yields_all_nan(monkeypatch, config, index):
    monkeypatch.setattr(cov, "download_series_close", lambda *a, **k: pd.Series(dtype="float64"))
    series = cov.vix_covariate(index, config)
    assert series.isna().all()


def test_volume_comes_from_the_ticker_frame(index):
    ohlcv = pd.DataFrame({"volume": np.arange(len(index), dtype="float64")}, index=index)
    series = cov.volume_covariate(ohlcv, index)
    assert series.name == "volume"
    assert float(series.iloc[3]) == 3.0


def test_earnings_flag_window(monkeypatch, config, index):
    release = index[10]
    monkeypatch.setattr(cov, "_fetch_earnings_dates", lambda *a, **k: pd.DatetimeIndex([release]))

    flag = cov.earnings_flag_covariate("TEST", index, config)
    assert flag.name == "earnings_flag"
    assert flag.dtype == "int8"
    # k=1 -> the release day plus one trading day either side.
    assert flag.iloc[9:12].sum() == 3
    assert flag.sum() == 3
    assert flag.iloc[8] == 0 and flag.iloc[12] == 0


def test_earnings_flag_k_zero_flags_only_the_release_day(monkeypatch, index):
    config = PipelineConfig(
        tickers=("TEST",), start="2020-01-01", end="2020-04-01", earnings_window_days=0
    )
    monkeypatch.setattr(cov, "_fetch_earnings_dates", lambda *a, **k: pd.DatetimeIndex([index[10]]))
    flag = cov.earnings_flag_covariate("TEST", index, config)
    assert flag.sum() == 1
    assert flag.iloc[10] == 1


def test_earnings_release_on_a_non_trading_day_anchors_to_the_next_session(monkeypatch, config, index):
    """A Saturday release should flag the following Monday, not vanish."""
    saturday = index[10] + pd.Timedelta(days=(5 - index[10].weekday()) % 7 or 5)
    assert saturday not in index
    monkeypatch.setattr(cov, "_fetch_earnings_dates", lambda *a, **k: pd.DatetimeIndex([saturday]))

    flag = cov.earnings_flag_covariate("TEST", index, config)
    assert flag.sum() > 0
    anchor = index.searchsorted(saturday)
    assert flag.iloc[anchor] == 1


def test_earnings_flag_ignores_releases_past_the_sample(monkeypatch, config, index):
    future = index[-1] + pd.Timedelta(days=90)
    monkeypatch.setattr(cov, "_fetch_earnings_dates", lambda *a, **k: pd.DatetimeIndex([future]))
    flag = cov.earnings_flag_covariate("TEST", index, config)
    assert flag.sum() == 0


def test_earnings_flag_survives_a_yfinance_failure(monkeypatch, config, index):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(cov.yf, "Ticker", lambda *a, **k: type("T", (), {"get_earnings_dates": boom})())
    flag = cov.earnings_flag_covariate("TEST", index, config)
    assert (flag == 0).all()


def test_build_covariates_tags_every_column(monkeypatch, config, index):
    monkeypatch.setattr(
        cov, "download_series_close", lambda *a, **k: pd.Series(20.0, index=index, name="^VIX")
    )
    monkeypatch.setattr(cov, "_fetch_earnings_dates", lambda *a, **k: pd.DatetimeIndex([index[10]]))
    ohlcv = pd.DataFrame({"volume": np.ones(len(index))}, index=index)

    bundle = cov.build_covariates("TEST", ohlcv, index, config)
    assert list(bundle.frame.columns) == ["vix", "volume", "earnings_flag"]
    assert bundle.tags == {"vix": "past_only", "volume": "past_only", "earnings_flag": "known_future"}
    assert bundle.frame.index.equals(index)


def test_risk_free_rate_degrades_when_fred_is_unavailable(monkeypatch, config, index):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pandas_datareader"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    series = cov.risk_free_rate_covariate(index, config)
    assert len(series) == len(index)
    assert series.isna().all()
