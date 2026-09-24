"""Model tests: coefficient recovery, forecast sanity and quantile behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volmodels import (
    EWMA,
    HAR,
    MEM,
    AMEM,
    Forecast,
    HistoricalMean,
    LogHAR,
    NotEnoughData,
    RandomWalk,
    baseline_factories,
)
from volmodels.har import har_features
from volmodels.mem import _recursion, _recursion_loop

from .conftest import TRUE_HAR_COEFFICIENTS, simulate_har


# --------------------------------------------------------------------------- #
# HAR coefficient recovery -- the core correctness test for the headline model
# --------------------------------------------------------------------------- #

def test_har_recovers_known_coefficients(har_series):
    """Fitted on data from a known HAR process, HAR must recover its parameters.

    The three cascade regressors are heavily collinear by construction -- the
    weekly average contains the daily value, the monthly contains the weekly
    (design condition number ~250).  OLS is still unbiased, but it trades the
    individual betas off against each other from sample to sample, so a single
    realisation identifies the *total persistence* and the intercept far more
    sharply than any one beta.  This test asserts what the data actually pins
    down; :func:`test_har_betas_are_unbiased_across_samples` handles the betas
    individually by averaging the noise away.
    """
    model = HAR()
    model.fit(har_series)
    estimated = model.coefficients(h=1)
    intercept, *betas = TRUE_HAR_COEFFICIENTS

    assert estimated[0] == pytest.approx(intercept, abs=0.005)
    assert sum(estimated[1:]) == pytest.approx(sum(betas), abs=0.03), (
        f"total persistence {sum(estimated[1:]):.4f} vs true {sum(betas):.4f}"
    )
    assert estimated[1:] == pytest.approx(np.array(betas), abs=0.15), (
        f"estimated {estimated.round(4)} vs true {np.array(TRUE_HAR_COEFFICIENTS)}"
    )
    # The residual scale is an independent check that the fit is right.
    residual_sd = np.sqrt(model._fit_for(1).residual_variance)
    assert residual_sd == pytest.approx(0.010, rel=0.15)


def test_har_betas_are_unbiased_across_samples():
    """Averaged over independent samples, each beta must land on its true value."""
    estimates = []
    for seed in range(24):
        model = HAR()
        model.fit(simulate_har(seed=seed))
        estimates.append(model.coefficients(h=1))

    mean_estimate = np.mean(estimates, axis=0)
    assert mean_estimate == pytest.approx(np.array(TRUE_HAR_COEFFICIENTS), abs=0.02), (
        f"mean over 24 samples {mean_estimate.round(4)} vs true {np.array(TRUE_HAR_COEFFICIENTS)}"
    )


def test_log_har_recovers_known_coefficients(log_har_series):
    """Same, for a process whose logarithm is HAR."""
    model = LogHAR()
    model.fit(log_har_series)
    estimated = model.coefficients(h=1)

    expected = np.array([-0.0805, 0.35, 0.35, 0.25])
    assert estimated == pytest.approx(expected, abs=0.06), (
        f"estimated {estimated.round(4)} vs true {expected}"
    )


def test_har_features_are_trailing_averages():
    """Row i must hold [1, x_i, mean(x_{i-4..i}), mean(x_{i-21..i})]."""
    values = np.arange(1.0, 41.0)
    design = har_features(values)

    assert design[:21, 3].tolist() == [np.nan] * 21 or np.isnan(design[:21, 3]).all()
    assert design[30, 0] == 1.0
    assert design[30, 1] == values[30]
    assert design[30, 2] == pytest.approx(values[26:31].mean())
    assert design[30, 3] == pytest.approx(values[9:31].mean())


def test_har_features_never_look_forward():
    """Changing a future value must not alter an earlier feature row."""
    values = np.arange(1.0, 61.0)
    baseline = har_features(values)
    poisoned = values.copy()
    poisoned[40:] = 1e6
    after = har_features(poisoned)

    np.testing.assert_allclose(baseline[:40], after[:40], equal_nan=True)


def test_har_direct_forecasting_uses_a_separate_fit_per_horizon(har_series):
    model = HAR()
    model.fit(har_series)
    coefficients = {h: model.coefficients(h) for h in (1, 5, 21)}

    assert not np.allclose(coefficients[1], coefficients[21]), (
        "h=1 and h=21 produced identical coefficients; direct forecasting is not happening"
    )
    # Longer horizons mean-revert: the cascade betas shrink and the intercept grows.
    assert coefficients[21][1:].sum() < coefficients[1][1:].sum()


# --------------------------------------------------------------------------- #
# Log-HAR bias correction and quantile semantics
# --------------------------------------------------------------------------- #

def test_log_har_bias_correction_raises_the_point_forecast(log_har_series):
    """exp(E[log RV]) understates E[RV]; the correction must fix that upward."""
    corrected = LogHAR(bias_correction=True)
    uncorrected = LogHAR(bias_correction=False)
    corrected.fit(log_har_series)
    uncorrected.fit(log_har_series)

    high = corrected.forecast(21).point
    low = uncorrected.forecast(21).point
    assert high > low

    # The gap is exactly exp(sigma^2 / 2).
    residual_variance = corrected._fit_for(21).residual_variance
    assert high / low == pytest.approx(np.exp(0.5 * residual_variance), rel=1e-10)


def test_log_har_median_sits_below_the_mean(log_har_series):
    """A right-skewed predictive distribution has median < mean; that is correct."""
    model = LogHAR()
    model.fit(log_har_series)
    prediction = model.forecast(21)

    assert prediction.quantiles[0.5] < prediction.point, (
        "the bias correction appears to have leaked into the quantiles"
    )


def test_quantiles_are_monotone_in_level(log_har_series):
    for factory in baseline_factories():
        model = factory()
        model.fit(log_har_series)
        prediction = model.forecast(5)
        levels = sorted(prediction.quantiles)
        values = [prediction.quantiles[level] for level in levels]
        assert values == sorted(values), f"{model.name}: quantiles are not monotone"


def test_quantiles_are_positive_and_bracket_the_point(log_har_series):
    for factory in baseline_factories():
        model = factory()
        model.fit(log_har_series)
        prediction = model.forecast(5)
        assert all(v > 0 for v in prediction.quantiles.values()), model.name
        assert prediction.quantiles[0.05] < prediction.point < prediction.quantiles[0.95], model.name


def test_quantile_grid_matches_chronos_default(log_har_series):
    model = LogHAR()
    model.fit(log_har_series)
    levels = sorted(model.forecast(1).quantiles)
    assert levels == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def test_prediction_intervals_widen_with_horizon(rv_series):
    """A longer horizon is more uncertain, so its band must be wider."""
    model = RandomWalk()
    model.fit(rv_series)
    widths = {
        h: model.forecast(h).quantiles[0.95] - model.forecast(h).quantiles[0.05]
        for h in (1, 5, 21)
    }
    assert widths[1] < widths[5] < widths[21], widths


# --------------------------------------------------------------------------- #
# Naive floors
# --------------------------------------------------------------------------- #

def test_random_walk_forecasts_the_last_value(rv_series):
    model = RandomWalk()
    model.fit(rv_series)
    for h in (1, 5, 21):
        assert model.forecast(h).point == pytest.approx(float(rv_series.iloc[-1]))


def test_historical_mean_forecasts_the_mean(rv_series):
    model = HistoricalMean()
    model.fit(rv_series)
    assert model.forecast(7).point == pytest.approx(float(rv_series.mean()))

    windowed = HistoricalMean(window=50)
    windowed.fit(rv_series)
    assert windowed.forecast(7).point == pytest.approx(float(rv_series.iloc[-50:].mean()))
    assert windowed.name == "historical_mean_50"


def test_ewma_stays_within_the_observed_range(rv_series):
    model = EWMA(lam=0.94)
    model.fit(rv_series)
    assert rv_series.min() <= model.forecast(1).point <= rv_series.max()


def test_ewma_lambda_controls_responsiveness():
    """A smaller lambda must react faster to a shock.

    Tested on a flat series with a single jump at the end, so the comparison is
    about the recursion's memory rather than about whichever way a particular
    random series happened to be drifting.
    """
    index = pd.bdate_range("2020-01-01", periods=200, name="date")
    values = np.full(200, 0.20)
    values[-1] = 0.60  # a single large shock on the last day
    series = pd.Series(values, index=index)

    responses = {}
    for lam in (0.5, 0.94):
        model = EWMA(lam=lam)
        model.fit(series)
        responses[lam] = model.forecast(1).point

    # level = lam * 0.20 + (1 - lam) * 0.60
    assert responses[0.5] == pytest.approx(0.40)
    assert responses[0.94] == pytest.approx(0.224)
    assert responses[0.5] > responses[0.94], "smaller lambda did not react faster"


def test_ewma_rejects_an_invalid_lambda():
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="lam"):
            EWMA(lam=bad)


# --------------------------------------------------------------------------- #
# MEM
# --------------------------------------------------------------------------- #

def test_mem_recursion_matches_the_reference_loop():
    """The lfilter fast path must reproduce the literal recursion exactly."""
    rng = np.random.default_rng(3)
    values = np.abs(rng.normal(0.2, 0.05, size=500)) + 0.01
    for omega, alpha, beta in [(0.02, 0.3, 0.6), (0.001, 0.05, 0.9), (0.1, 0.0, 0.0)]:
        fast = _recursion(values, omega, alpha, beta)
        slow = _recursion_loop(values, omega, alpha, beta)
        np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-14)


def test_mem_fits_and_forecasts(rv_series):
    model = MEM()
    model.fit(rv_series)

    assert 0.0 < model.persistence < 1.0
    assert model.omega > 0.0
    for h in (1, 5, 21):
        point = model.forecast(h).point
        assert point > 0.0
        assert 0.5 * rv_series.min() < point < 2.0 * rv_series.max()


def test_mem_multi_step_reverts_towards_the_unconditional_mean(rv_series):
    model = MEM()
    model.fit(rv_series)
    long_run = model.omega / (1.0 - model.persistence)

    near, far = model.forecast(1).point, model.forecast(250).point
    assert abs(far - long_run) < abs(near - long_run) + 1e-12


def test_mem_rejects_a_non_positive_series(rv_series):
    broken = rv_series.copy()
    broken.iloc[10] = -1.0
    with pytest.raises(NotEnoughData, match="strictly positive"):
        MEM().fit(broken)


def test_amem_is_an_explicit_stub():
    with pytest.raises(NotImplementedError, match="signed returns"):
        AMEM()


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #

def test_every_baseline_satisfies_the_interface(rv_series):
    for factory in baseline_factories(include_mem=True):
        model = factory()
        model.fit(rv_series)
        prediction = model.forecast(5)

        assert isinstance(prediction, Forecast)
        assert prediction.model == model.name
        assert prediction.horizon == 5
        assert prediction.origin == rv_series.index[-1]
        assert np.isfinite(prediction.point) and prediction.point > 0


def test_update_conditions_on_the_latest_data_without_refitting(har_series):
    """update() must move the forecast forward even though betas are unchanged."""
    model = HAR()
    model.fit(har_series.iloc[:500])
    before = model.forecast(1).point
    coefficients_before = model.coefficients(1)

    model.update(har_series.iloc[:600])
    after = model.forecast(1).point

    np.testing.assert_allclose(model.coefficients(1), coefficients_before, rtol=1e-12)
    assert model.origin == har_series.index[599]
    assert before != after, "update() did not refresh the conditioning data"


def test_forecast_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        RandomWalk().forecast(1)


def test_short_history_raises_not_enough_data():
    """HAR estimates lazily per horizon, so a short history fails at forecast()."""
    short = simulate_har(n_days=25, seed=1)
    model = HAR()
    model.fit(short)  # building the design matrix succeeds
    with pytest.raises(NotEnoughData, match="rows for horizon"):
        model.forecast(1)  # the per-horizon regression is what runs out of data


def test_empty_history_raises_not_enough_data():
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    with pytest.raises(NotEnoughData, match="empty"):
        RandomWalk().fit(empty)


def test_unsorted_history_is_rejected(rv_series):
    with pytest.raises(ValueError, match="sorted ascending"):
        RandomWalk().fit(rv_series.iloc[::-1])


def test_invalid_horizon_is_rejected(rv_series):
    model = RandomWalk()
    model.fit(rv_series)
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(0)


def test_levels_har_can_floor_negative_forecasts():
    """A levels-space HAR on a near-zero series predicts below zero; count it."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2015-01-01", periods=400, name="date")
    noisy = pd.Series(np.abs(rng.normal(0.01, 0.05, size=400)) + 1e-6, index=index)

    model = HAR()
    model.fit(noisy)
    for h in (1, 5, 21):
        assert model.forecast(h).point > 0.0
    # Not asserting a floor occurred (it is data-dependent), only that the
    # counter exists and the output stayed positive either way.
    assert model.n_floored >= 0


def test_reset_clears_state(rv_series):
    model = LogHAR()
    model.fit(rv_series)
    model.forecast(1)
    model.reset()

    assert model.n_floored == 0
    with pytest.raises(RuntimeError):
        model.forecast(1)
