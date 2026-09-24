"""Estimator unit tests against a synthetic series with a known volatility."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from volpipe import estimators as est
from volpipe.config import PipelineConfig
from volpipe.estimators import compute_estimator, rolling_variance, to_realized_vol

from .conftest import ANNUALIZATION, TRUE_ANNUAL_VOL, TRUE_DAILY_SIGMA, simulate_ohlc

DAILY_ESTIMATORS = [
    ("close_to_close", est.close_to_close_variance),
    ("parkinson", est.parkinson_variance),
    ("garman_klass", est.garman_klass_variance),
    ("rogers_satchell", est.rogers_satchell_variance),
]


@pytest.mark.parametrize("name,func", DAILY_ESTIMATORS)
def test_daily_variance_is_positive(name, func, synthetic_ohlc):
    values = func(synthetic_ohlc).dropna()
    assert len(values) > 1000
    assert (values >= 0).all(), f"{name} produced a negative variance"
    assert float(values.mean()) > 0.0
    # Rogers-Satchell is exactly zero on a perfectly monotone session (H==O and
    # L==C, or vice versa), which a discretely-sampled path hits a per cent or
    # two of the time -- so require "almost always positive", not "always".
    assert (values > 0).mean() > 0.95, f"{name} is degenerate (mostly zeros)"


@pytest.mark.parametrize("name,func", DAILY_ESTIMATORS)
def test_daily_variance_recovers_true_volatility(name, func, synthetic_ohlc):
    """Mean daily variance should sit near sigma^2 for a known-vol series."""
    mean_var = float(func(synthetic_ohlc).dropna().mean())
    implied_annual_vol = math.sqrt(mean_var * ANNUALIZATION)
    assert implied_annual_vol == pytest.approx(TRUE_ANNUAL_VOL, rel=0.15), (
        f"{name}: implied {implied_annual_vol:.4f} vs true {TRUE_ANNUAL_VOL:.4f}"
    )


def test_yang_zhang_recovers_true_volatility(synthetic_ohlc):
    mean_var = float(est.yang_zhang_variance(synthetic_ohlc, window=21).dropna().mean())
    implied = math.sqrt(mean_var * ANNUALIZATION)
    assert implied == pytest.approx(TRUE_ANNUAL_VOL, rel=0.15)


def test_parkinson_formula_matches_hand_computation():
    ohlc = pd.DataFrame(
        {"open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0]},
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    expected = (math.log(102.0 / 99.0) ** 2) / (4.0 * math.log(2.0))
    assert float(est.parkinson_variance(ohlc).iloc[0]) == pytest.approx(expected)


def test_garman_klass_formula_matches_hand_computation():
    ohlc = pd.DataFrame(
        {"open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0]},
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    hl, co = math.log(102.0 / 99.0), math.log(101.0 / 100.0)
    expected = 0.5 * hl**2 - (2 * math.log(2) - 1) * co**2
    assert float(est.garman_klass_variance(ohlc).iloc[0]) == pytest.approx(expected)


def test_rogers_satchell_formula_matches_hand_computation():
    ohlc = pd.DataFrame(
        {"open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0]},
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    expected = (
        math.log(102.0 / 101.0) * math.log(102.0 / 100.0)
        + math.log(99.0 / 101.0) * math.log(99.0 / 100.0)
    )
    assert float(est.rogers_satchell_variance(ohlc).iloc[0]) == pytest.approx(expected)


def test_rogers_satchell_is_drift_independent(synthetic_ohlc_drift):
    """RS stays unbiased under drift where Parkinson/GK are pulled upward."""
    rs = math.sqrt(est.rogers_satchell_variance(synthetic_ohlc_drift).dropna().mean() * ANNUALIZATION)
    pk = math.sqrt(est.parkinson_variance(synthetic_ohlc_drift).dropna().mean() * ANNUALIZATION)
    assert rs == pytest.approx(TRUE_ANNUAL_VOL, rel=0.15)
    assert abs(rs - TRUE_ANNUAL_VOL) <= abs(pk - TRUE_ANNUAL_VOL) + 0.02


def test_close_to_close_windowed_uses_sample_variance(small_ohlc):
    window = 10
    returns = np.log(small_ohlc["close"]).diff()
    windowed = rolling_variance("close_to_close", est.close_to_close_variance(small_ohlc), window, ohlc=small_ohlc)
    expected = float(returns.iloc[-window:].var(ddof=1))
    assert float(windowed.iloc[-1]) == pytest.approx(expected)


def test_rolling_mean_aggregation_for_range_estimators(small_ohlc):
    window = 10
    daily = est.parkinson_variance(small_ohlc)
    windowed = rolling_variance("parkinson", daily, window, ohlc=small_ohlc)
    assert float(windowed.iloc[-1]) == pytest.approx(float(daily.iloc[-window:].mean()))


def test_rolling_window_has_no_lookahead(synthetic_ohlc):
    """Truncating the series after date t must not change the value at t."""
    window, ann = 21, ANNUALIZATION
    cutoff = 800
    for name in ("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"):
        full = compute_estimator(name, synthetic_ohlc, window=window, annualization_factor=ann)
        truncated = compute_estimator(
            name, synthetic_ohlc.iloc[: cutoff + 1], window=window, annualization_factor=ann
        )
        at_t_full = float(full.realized_vol.iloc[cutoff])
        at_t_trunc = float(truncated.realized_vol.iloc[cutoff])
        assert at_t_full == pytest.approx(at_t_trunc, rel=1e-12), f"{name} leaks future data"


def test_warmup_rows_are_nan(synthetic_ohlc):
    window = 21
    result = compute_estimator("parkinson", synthetic_ohlc, window=window, annualization_factor=252)
    assert result.realized_vol.iloc[: window - 1].isna().all()
    assert not np.isnan(result.realized_vol.iloc[window - 1])


def test_annualization_scaling():
    variance = pd.Series([0.0004, 0.0009], index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]))
    vol = to_realized_vol(variance, 252)
    assert float(vol.iloc[0]) == pytest.approx(math.sqrt(0.0004 * 252))
    # Doubling the factor scales vol by sqrt(2).
    doubled = to_realized_vol(variance, 504)
    assert float(doubled.iloc[0]) == pytest.approx(float(vol.iloc[0]) * math.sqrt(2))


def test_realized_vol_is_in_the_right_ballpark(synthetic_ohlc):
    for name in ("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"):
        rv = compute_estimator(name, synthetic_ohlc, window=63, annualization_factor=ANNUALIZATION).realized_vol
        series = rv.dropna()
        assert (series > 0).all(), f"{name} produced a non-positive volatility"
        assert float(series.mean()) == pytest.approx(TRUE_ANNUAL_VOL, rel=0.15), name


def test_close_to_close_is_the_noisiest(synthetic_ohlc):
    """Theory: CC uses one price/day, so its rolling vol wobbles most."""
    noise = {}
    for name in ("close_to_close", "parkinson", "garman_klass", "rogers_satchell"):
        rv = compute_estimator(name, synthetic_ohlc, window=21, annualization_factor=252).realized_vol.dropna()
        noise[name] = float(rv.std() / rv.mean())
    assert noise["close_to_close"] == max(noise.values()), noise


def test_estimator_series_are_highly_correlated(synthetic_ohlc_varying_vol):
    """With a real volatility signal to track, the estimators should agree.

    Run against a *constant*-volatility path this would fail by design: there
    would be no shared signal, only each estimator's own sampling noise.
    """
    frame = pd.DataFrame(
        {
            name: compute_estimator(
                name, synthetic_ohlc_varying_vol, window=21, annualization_factor=252
            ).realized_vol
            for name in ("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang")
        }
    ).dropna()
    corr = frame.corr()
    off_diagonal = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    assert off_diagonal.min() > 0.6, corr
    # The range-based estimators see the same H/L/C and should agree very closely.
    assert corr.loc["parkinson", "garman_klass"] > 0.9, corr


def test_intraday_rv_from_synthetic_bars():
    """Sum of squared 5-minute returns should recover the daily variance."""
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2024-01-02", periods=20)
    bars_per_day = 78
    step_sigma = TRUE_DAILY_SIGMA / np.sqrt(bars_per_day)

    stamps, prices, level = [], [], 100.0
    for day in days:
        for i in range(bars_per_day):
            level *= math.exp(rng.normal(0.0, step_sigma))
            stamps.append(day + pd.Timedelta(minutes=5 * i))
            prices.append(level)

    intraday = pd.DataFrame({"close": prices}, index=pd.DatetimeIndex(stamps))
    rv = est.intraday_rv_variance(intraday, pd.DatetimeIndex(days))

    assert rv.notna().sum() == len(days)
    assert (rv.dropna() > 0).all()
    implied = math.sqrt(float(rv.mean()) * ANNUALIZATION)
    assert implied == pytest.approx(TRUE_ANNUAL_VOL, rel=0.20)


def test_intraday_rv_degrades_gracefully():
    index = pd.bdate_range("2024-01-02", periods=10)
    for missing in (None, pd.DataFrame(columns=["close"])):
        rv = est.intraday_rv_variance(missing, index)
        assert len(rv) == len(index)
        assert rv.isna().all()

    result = compute_estimator("intraday_rv", simulate_ohlc(n_days=10), window=5, annualization_factor=252, intraday=None)
    assert result.available is False
    assert result.realized_vol.isna().all()


def test_intraday_rv_excludes_overnight_gap():
    """A huge gap between sessions must not enter the sum of squared returns."""
    stamps = pd.DatetimeIndex(
        ["2024-01-02 09:30", "2024-01-02 09:35"] * 1 + ["2024-01-03 09:30", "2024-01-03 09:35"]
    )
    # Day 1 ends at 100, day 2 opens at 200: a 100% "overnight return".
    intraday = pd.DataFrame({"close": [100.0, 100.0, 200.0, 200.0]}, index=stamps)
    rv = est.intraday_rv_variance(intraday, pd.DatetimeIndex(["2024-01-02", "2024-01-03"]), min_bars_per_day=2)
    assert float(rv.iloc[1]) == pytest.approx(0.0), "overnight gap leaked into intraday RV"


def test_non_positive_prices_are_masked():
    ohlc = pd.DataFrame(
        {"open": [100.0, 0.0], "high": [101.0, 1.0], "low": [99.0, -1.0], "close": [100.5, 0.5]},
        index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]),
    )
    values = est.parkinson_variance(ohlc)
    assert not np.isnan(values.iloc[0])
    assert np.isnan(values.iloc[1])


def test_unsorted_index_is_rejected(small_ohlc):
    with pytest.raises(ValueError, match="sorted ascending"):
        est.parkinson_variance(small_ohlc.iloc[::-1])


def test_unknown_estimator_raises(small_ohlc):
    with pytest.raises(ValueError, match="unknown estimator"):
        compute_estimator("not_an_estimator", small_ohlc, window=5, annualization_factor=252)


def test_missing_column_raises(small_ohlc):
    with pytest.raises(KeyError):
        est.parkinson_variance(small_ohlc.drop(columns=["high"]))
