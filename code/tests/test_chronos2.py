"""Chronos-2 conformance, covariate handling and lookahead tests.

These run **CPU-only** on tiny synthetic series so the suite needs no GPU.  They
are skipped entirely when ``chronos-forecasting`` is not installed, so the rest
of the project stays testable without a 120M-parameter download.

The claim they defend: Chronos-2 is a *drop-in*.  It runs through the unchanged
harness, on the unchanged target, and is bound by the same no-lookahead rule as
every classical model -- with the single, deliberate known-future exception the
pipeline already documents for earnings timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volmodels.base import Forecast

chronos = pytest.importorskip("chronos", reason="chronos-forecasting is not installed")

from volmodels.chronos2 import (  # noqa: E402
    KNOWN_FUTURE_COLUMN,
    Chronos2Covariate,
    Chronos2Multivariate,
    Chronos2Univariate,
    earnings_countdown,
    load_pipeline,
    resolve_device,
)
from voleval import BacktestConfig, add_losses, run_backtest, summarize  # noqa: E402
from voleval.backtest import walk_forward  # noqa: E402

from .conftest import simulate_har  # noqa: E402

#: Everything here runs on CPU with a short context, so the suite stays fast.
CPU = {"device": "cpu", "context_length": 120, "max_horizon": 5}


@pytest.fixture(scope="module")
def cpu_pipeline():
    """Load the checkpoint once for the whole module (and prove it caches)."""
    return load_pipeline(device="cpu")


@pytest.fixture
def series() -> pd.Series:
    s = simulate_har(n_days=260, seed=11)
    s.name = "AAA"
    return s


@pytest.fixture
def exog(series) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    var = series.to_numpy() ** 2
    return pd.DataFrame(
        {
            "var_bv": var * 0.9,
            "var_rsp": var * 0.5,
            "var_rsn": var * 0.5,
            "var_rq": var**2 * 3,
            "jump": np.abs(rng.normal(0, var.mean() * 0.1, len(series))),
            "vix": 15 + 10 * rng.random(len(series)),
        },
        index=series.index,
    )


# --------------------------------------------------------------------------- #
# Device handling and checkpoint caching
# --------------------------------------------------------------------------- #

def test_resolve_device_honours_an_explicit_choice():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in ("cpu", "cuda")


def test_pipeline_is_loaded_once(cpu_pipeline):
    """A backtest makes tens of thousands of calls; reloading would dominate."""
    assert load_pipeline(device="cpu") is cpu_pipeline


# --------------------------------------------------------------------------- #
# Interface conformance
# --------------------------------------------------------------------------- #

def test_univariate_satisfies_the_forecaster_interface(cpu_pipeline, series):
    model = Chronos2Univariate(**CPU)
    model.fit(series)
    prediction = model.forecast(1)

    assert isinstance(prediction, Forecast)
    assert prediction.model == "chronos2_uni"
    assert prediction.horizon == 1
    assert prediction.origin == series.index[-1]
    assert np.isfinite(prediction.point) and prediction.point > 0


def test_fit_is_zero_shot_and_repeatable(cpu_pipeline, series):
    """No training happens, so refitting the same context must be identical."""
    model = Chronos2Univariate(**CPU)
    model.fit(series)
    first = model.forecast(1).point
    model.fit(series)
    assert model.forecast(1).point == pytest.approx(first, rel=1e-9)


def test_update_refreshes_the_context(cpu_pipeline, series):
    model = Chronos2Univariate(**CPU)
    model.fit(series.iloc[:200])
    before = model.forecast(1).point
    model.update(series.iloc[:240])
    assert model.origin == series.index[239]
    assert model.forecast(1).point != before


def test_forecast_before_fit_raises(cpu_pipeline):
    with pytest.raises(RuntimeError, match="before fit"):
        Chronos2Univariate(**CPU).forecast(1)


def test_invalid_horizon_is_rejected(cpu_pipeline, series):
    model = Chronos2Univariate(**CPU)
    model.fit(series)
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(0)


# --------------------------------------------------------------------------- #
# Native quantiles
# --------------------------------------------------------------------------- #

def test_native_quantiles_are_monotone_and_positive(cpu_pipeline, series):
    model = Chronos2Univariate(**CPU)
    model.fit(series)
    for h in (1, 5):
        q = model.forecast(h).quantiles
        levels = sorted(q)
        values = [q[level] for level in levels]
        assert levels == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        assert values == sorted(values), f"quantiles not monotone at h={h}"
        assert all(v > 0 for v in values)


def test_point_forecast_is_the_median(cpu_pipeline, series):
    """Volatility's predictive law is right-skewed; we report the median."""
    model = Chronos2Univariate(**CPU)
    model.fit(series)
    prediction = model.forecast(1)
    assert prediction.point == pytest.approx(prediction.quantiles[0.5], rel=1e-12)


def test_quantiles_come_from_the_model_not_from_residuals(cpu_pipeline, series):
    """Chronos-2 must not inherit the baselines' empirical-residual machinery."""
    from volmodels.base import EmpiricalQuantileForecaster

    model = Chronos2Univariate(**CPU)
    assert not isinstance(model, EmpiricalQuantileForecaster)
    model.fit(series)
    # Bands must widen with horizon the way a real predictive law does.
    narrow = model.forecast(1).quantiles
    wide = model.forecast(5).quantiles
    assert (wide[0.95] - wide[0.05]) > (narrow[0.95] - narrow[0.05])


def test_one_model_call_serves_every_horizon(cpu_pipeline, series, monkeypatch):
    """The path is cached per origin: the expensive call must not repeat per h."""
    model = Chronos2Univariate(**CPU)
    model.fit(series)

    calls = {"n": 0}
    original = model._compute_path

    def counting(prediction_length):
        calls["n"] += 1
        return original(prediction_length)

    monkeypatch.setattr(model, "_compute_path", counting)
    for h in (1, 2, 5):
        model.forecast(h)
    assert calls["n"] == 1

    model.update(series.iloc[:250])  # new origin invalidates the cache
    model.forecast(1)
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# Covariates
# --------------------------------------------------------------------------- #

def test_earnings_countdown_shape():
    index = pd.bdate_range("2020-01-01", periods=120, name="date")
    releases = pd.DatetimeIndex([index[30], index[90]])
    countdown = earnings_countdown(index, releases, cap=63)

    assert countdown.name == KNOWN_FUTURE_COLUMN
    assert countdown.iloc[30] == 0.0 and countdown.iloc[90] == 0.0
    assert countdown.iloc[29] == 1.0 and countdown.iloc[25] == 5.0
    # Strictly decreasing as a release approaches.
    assert (np.diff(countdown.iloc[:31].to_numpy()) == -1).all()
    # Past the last release there is nothing ahead, so it sits at the cap.
    assert countdown.iloc[-1] == 63.0


def test_earnings_countdown_with_no_releases():
    index = pd.bdate_range("2020-01-01", periods=50, name="date")
    countdown = earnings_countdown(index, pd.DatetimeIndex([]), cap=63)
    assert (countdown == 63.0).all()


def test_covariate_mode_consumes_the_exog_channel(cpu_pipeline, series, exog):
    model = Chronos2Covariate(**CPU)
    model.fit(series, exog)
    built = model._build_inputs(5)[0]

    assert "past_covariates" in built
    assert {"var_bv", "var_rsp", "var_rsn", "var_rq", "jump", "vix"} <= set(built["past_covariates"])
    for values in built["past_covariates"].values():
        assert values.shape[0] == built["target"].shape[0]
        assert np.isfinite(values).all()


def test_known_future_countdown_is_supplied_on_both_sides(cpu_pipeline, series, exog):
    """Chronos-2 requires future covariate keys to be a subset of past keys."""
    countdown = earnings_countdown(series.index, pd.DatetimeIndex([series.index[100]]))
    model = Chronos2Covariate(known_future=countdown, **CPU)
    model.fit(series.iloc[:200], exog.iloc[:200])

    built = model._build_inputs(5)[0]
    assert KNOWN_FUTURE_COLUMN in built["past_covariates"]
    assert KNOWN_FUTURE_COLUMN in built["future_covariates"]
    assert set(built["future_covariates"]) <= set(built["past_covariates"])
    assert built["future_covariates"][KNOWN_FUTURE_COLUMN].shape[0] == 5


def test_known_future_schedule_can_be_per_ticker(cpu_pipeline, series, exog):
    countdown = earnings_countdown(series.index, pd.DatetimeIndex([series.index[100]]))
    model = Chronos2Covariate(known_future={"AAA": countdown, "BBB": countdown * 0}, **CPU)
    model.fit(series.iloc[:200], exog.iloc[:200])

    assert model.series_name == "AAA"
    assert model._schedule() is countdown
    assert KNOWN_FUTURE_COLUMN in model._build_inputs(5)[0]["future_covariates"]


def test_covariate_mode_degrades_without_exog(cpu_pipeline, series):
    """No covariates supplied must still produce a valid univariate forecast."""
    model = Chronos2Covariate(**CPU)
    model.fit(series)
    built = model._build_inputs(5)[0]
    assert "past_covariates" not in built and "future_covariates" not in built
    assert model.forecast(1).point > 0


# --------------------------------------------------------------------------- #
# Lookahead -- the guard that matters
# --------------------------------------------------------------------------- #

def test_past_only_covariates_never_reach_past_the_origin(cpu_pipeline, series, exog):
    """Everything except the earnings schedule stops at the origin."""
    countdown = earnings_countdown(series.index, pd.DatetimeIndex([series.index[100]]))
    model = Chronos2Covariate(known_future=countdown, **CPU)

    origin_position = 200
    model.fit(series.iloc[:origin_position], exog.iloc[:origin_position])
    built = model._build_inputs(5)[0]
    context_length = built["target"].shape[0]

    for name, values in built["past_covariates"].items():
        assert values.shape[0] == context_length, name
        # A past covariate must equal the exog column over the context window,
        # i.e. carry no value dated after the origin.
        if name in exog.columns:
            expected = exog[name].iloc[origin_position - context_length : origin_position]
            np.testing.assert_allclose(values, expected.to_numpy(), rtol=1e-12)


def test_model_rejects_exog_that_extends_past_the_origin(cpu_pipeline, series, exog):
    with pytest.raises(ValueError, match="lookahead"):
        Chronos2Covariate(**CPU).fit(series.iloc[:150], exog.iloc[:200])


def test_poisoning_the_future_cannot_change_past_forecasts(cpu_pipeline, series, exog):
    """The tail-poison guard, applied to Chronos-2 on the real harness."""
    cutoff = 220
    dirty_series = series.copy()
    dirty_series.iloc[cutoff:] = 9.0
    dirty_exog = exog.copy()
    dirty_exog.iloc[cutoff:] = 9.0

    config = BacktestConfig(horizons=(1,), min_train_size=200, refit_frequency=1)
    factories = [lambda: Chronos2Univariate(**CPU), lambda: Chronos2Covariate(**CPU)]

    clean, _ = walk_forward(series, factories, config, ticker="AAA", exog=exog)
    dirty, _ = walk_forward(dirty_series, factories, config, ticker="AAA", exog=dirty_exog)

    boundary = series.index[cutoff]
    keys = ["model", "origin_date", "horizon"]
    before_clean = clean[clean["origin_date"] < boundary].set_index(keys)["forecast"]
    before_dirty = dirty[dirty["origin_date"] < boundary].set_index(keys)["forecast"]

    assert len(before_clean) > 0
    pd.testing.assert_series_equal(before_clean.sort_index(), before_dirty.sort_index())


def test_known_future_countdown_may_span_the_horizon(cpu_pipeline, series, exog):
    """The one permitted exception, asserted explicitly rather than assumed."""
    countdown = earnings_countdown(series.index, pd.DatetimeIndex([series.index[230]]))
    model = Chronos2Covariate(known_future=countdown, **CPU)
    model.fit(series.iloc[:200], exog.iloc[:200])

    future = model._build_inputs(5)[0]["future_covariates"][KNOWN_FUTURE_COLUMN]
    expected = countdown.iloc[200:205].to_numpy()
    np.testing.assert_allclose(future, expected, rtol=1e-12)


# --------------------------------------------------------------------------- #
# End-to-end through the UNCHANGED harness
# --------------------------------------------------------------------------- #

def test_runs_end_to_end_through_the_unchanged_harness(cpu_pipeline, series, exog):
    config = BacktestConfig(horizons=(1, 5), min_train_size=200, refit_frequency=1)
    outcome = run_backtest(
        {"AAA": series}, [lambda: Chronos2Univariate(**CPU)], config,
        exog_by_ticker={"AAA": exog},
    )

    assert not outcome.results.empty
    assert outcome.results["forecast"].notna().all()
    assert (outcome.results["forecast"] > 0).all()
    assert (outcome.diagnostics["failed_fits"] == 0).all()
    assert set(outcome.results["model"]) == {"chronos2_uni"}

    scored = add_losses(outcome.results)
    assert np.isfinite(scored["qlike"]).all()
    summarize(scored, benchmark="chronos2_uni")  # metrics need no special-casing


def test_multivariate_mode_stacks_companions(cpu_pipeline, series):
    companion = simulate_har(n_days=260, seed=12)
    companion.name = "BBB"
    model = Chronos2Multivariate(companions={"AAA": series, "BBB": companion}, **CPU)
    model.fit(series)

    target = model._build_inputs(5)[0]["target"]
    # 2-D stack, and the series must not be paired with a copy of itself.
    assert target.ndim == 2 and target.shape[0] == 2
    assert model.forecast(1).point > 0
