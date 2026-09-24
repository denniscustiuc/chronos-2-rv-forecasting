"""Significance tests: Diebold-Mariano and the Model Confidence Set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voleval.metrics import add_losses
from voleval.significance import (
    _circular_block_indices,
    _hac_variance,
    diebold_mariano,
    dm_against_benchmark,
    loss_matrix,
    model_confidence_set,
    per_ticker_dm,
)


def make_results(
    model_noise: dict[str, float],
    *,
    n: int = 400,
    tickers: tuple[str, ...] = ("AAA",),
    horizon: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic backtest results where each model's accuracy is set by hand."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n, name="date")
    rows = []
    for ticker in tickers:
        realized = 0.2 * np.exp(rng.normal(0.0, 0.3, size=n))
        for model, noise in model_noise.items():
            errors = rng.normal(0.0, noise, size=n)
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "model": model,
                        "ticker": ticker,
                        "origin_date": date,
                        "target_date": date,
                        "horizon": horizon,
                        "forecast": float(np.clip(realized[i] + errors[i], 1e-4, None)),
                        "realized": float(realized[i]),
                    }
                )
    return add_losses(pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# HAC variance
# --------------------------------------------------------------------------- #

def test_hac_variance_matches_the_plain_variance_at_zero_lags():
    rng = np.random.default_rng(0)
    d = rng.normal(size=500)
    assert _hac_variance(d, lags=0) == pytest.approx(float(d.var()) / d.size, rel=1e-12)


def test_hac_variance_grows_with_positive_autocorrelation():
    """Overlapping forecasts induce positive serial correlation; HAC must notice."""
    rng = np.random.default_rng(1)
    shocks = rng.normal(size=2000)
    correlated = shocks[5:] + shocks[4:-1] + shocks[3:-2] + shocks[2:-3] + shocks[1:-4]

    naive = float(correlated.var()) / correlated.size
    hac = _hac_variance(correlated, lags=4)
    assert hac > 1.5 * naive, (hac, naive)


def test_hac_variance_never_returns_a_non_positive_value():
    alternating = np.array([1.0, -1.0] * 100)
    assert _hac_variance(alternating, lags=5) > 0.0


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #

def test_dm_detects_a_clearly_better_model():
    results = make_results({"good": 0.01, "bad": 0.10}, n=500, seed=2)
    table = dm_against_benchmark(results, horizon=1, benchmark="bad")

    row = table.iloc[0]
    assert row["model_a"] == "good"
    assert row["mean_diff"] < 0, "the better model should have the lower loss"
    assert row["dm_stat"] < 0
    assert row["p_value"] < 0.01
    assert row["better"] == "good"


def test_dm_reports_a_tie_for_identical_models():
    results = make_results({"a": 0.05, "b": 0.05}, n=500, seed=3)
    table = dm_against_benchmark(results, horizon=1, benchmark="b")
    assert table.iloc[0]["better"] == "tie"
    assert table.iloc[0]["p_value"] > 0.05


def test_dm_on_identical_loss_series_is_exactly_a_tie():
    losses = np.random.default_rng(4).uniform(0.1, 1.0, size=200)
    outcome = diebold_mariano(losses, losses.copy(), horizon=1)
    assert outcome.mean_difference == pytest.approx(0.0)
    assert outcome.p_value == pytest.approx(1.0)
    assert outcome.better == "tie"


def test_dm_is_antisymmetric():
    rng = np.random.default_rng(5)
    a, b = rng.uniform(0.1, 1.0, size=300), rng.uniform(0.1, 1.0, size=300)
    forward = diebold_mariano(a, b, horizon=1)
    reverse = diebold_mariano(b, a, horizon=1)

    assert forward.mean_difference == pytest.approx(-reverse.mean_difference)
    assert forward.statistic == pytest.approx(-reverse.statistic)
    assert forward.p_value == pytest.approx(reverse.p_value)


def test_dm_hln_correction_shrinks_the_statistic_at_longer_horizons():
    """Multi-step overlap must widen the standard error, not narrow it."""
    rng = np.random.default_rng(6)
    a = rng.normal(1.0, 0.1, size=300)
    b = rng.normal(1.05, 0.1, size=300)

    short = diebold_mariano(a, b, horizon=1)
    long = diebold_mariano(a, b, horizon=21)
    assert abs(long.statistic) < abs(short.statistic)
    assert long.p_value > short.p_value
    assert long.lags == 20


def test_dm_requires_enough_observations():
    with pytest.raises(ValueError, match="at least 3"):
        diebold_mariano([0.1, 0.2], [0.1, 0.3], horizon=1)


def test_dm_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        diebold_mariano([0.1, 0.2, 0.3], [0.1, 0.2], horizon=1)


def test_per_ticker_dm_splits_by_asset():
    results = make_results({"good": 0.01, "log_har": 0.06}, n=300, tickers=("AAA", "BBB"), seed=7)
    table = per_ticker_dm(results, horizon=1, benchmark="log_har")

    assert set(table["ticker"]) == {"AAA", "BBB"}
    assert (table["mean_diff"] < 0).all()
    assert len(table) == 2


def test_loss_matrix_keeps_only_common_observations():
    results = make_results({"a": 0.02, "b": 0.03}, n=200, seed=8)
    results.loc[(results["model"] == "a") & (results.index % 5 == 0), "qlike"] = np.nan

    matrix = loss_matrix(results, horizon=1)
    assert list(matrix.columns) == ["a", "b"]
    assert matrix.notna().all().all()
    assert len(matrix) < 200


def test_loss_matrix_requires_the_loss_column():
    frame = pd.DataFrame({"model": ["a"], "ticker": ["AAA"], "horizon": [1], "forecast": [0.1]})
    with pytest.raises(KeyError, match="add_losses"):
        loss_matrix(frame, horizon=1)


# --------------------------------------------------------------------------- #
# Model Confidence Set
# --------------------------------------------------------------------------- #

def test_block_bootstrap_indices_are_in_range_and_contiguous():
    rng = np.random.default_rng(0)
    indices = _circular_block_indices(n=50, block_size=5, n_bootstrap=20, rng=rng)

    assert indices.shape == (20, 50)
    assert indices.min() >= 0 and indices.max() < 50
    # Within a block, successive indices advance by one (mod n).
    first_block = indices[0, :5]
    assert np.all((np.diff(first_block) % 50) == 1)


def test_mcs_eliminates_a_clearly_inferior_model():
    results = make_results({"good": 0.01, "medium": 0.02, "terrible": 0.30}, n=500, seed=9)
    outcome = model_confidence_set(results, horizon=1, n_bootstrap=400, seed=0)

    assert "terrible" not in outcome.included
    assert "good" in outcome.included
    terrible = outcome.table.set_index("model").loc["terrible"]
    assert terrible["included"] is np.False_ or terrible["included"] is False
    assert terrible["mcs_pvalue"] < 0.10


def test_mcs_keeps_indistinguishable_models():
    results = make_results({"a": 0.03, "b": 0.03, "c": 0.03}, n=400, seed=10)
    outcome = model_confidence_set(results, horizon=1, n_bootstrap=400, seed=0)
    assert set(outcome.included) == {"a", "b", "c"}


def test_mcs_table_is_ordered_by_mean_loss():
    results = make_results({"good": 0.01, "medium": 0.05, "bad": 0.15}, n=400, seed=11)
    outcome = model_confidence_set(results, horizon=1, n_bootstrap=300, seed=0)

    assert outcome.table["mean_loss"].is_monotonic_increasing
    assert outcome.table.iloc[0]["model"] == "good"
    assert set(outcome.table.columns) == {
        "model", "mcs_pvalue", "included", "eliminated_at", "mean_loss"
    }


def test_mcs_is_reproducible():
    results = make_results({"a": 0.02, "b": 0.04, "c": 0.09}, n=300, seed=12)
    first = model_confidence_set(results, horizon=1, n_bootstrap=300, seed=42)
    second = model_confidence_set(results, horizon=1, n_bootstrap=300, seed=42)
    pd.testing.assert_frame_equal(first.table, second.table)


def test_mcs_degrades_gracefully_with_one_model():
    results = make_results({"only": 0.02}, n=200, seed=13)
    outcome = model_confidence_set(results, horizon=1, n_bootstrap=100)
    assert outcome.included == ["only"]
    assert outcome.table["mcs_pvalue"].iloc[0] == 1.0


def test_mcs_degrades_gracefully_with_too_few_observations():
    results = make_results({"a": 0.02, "b": 0.05}, n=5, seed=14)
    outcome = model_confidence_set(results, horizon=1, n_bootstrap=100)
    assert set(outcome.included) == {"a", "b"}
    assert outcome.n_bootstrap == 0


def test_mcs_block_size_defaults_to_the_horizon():
    results = make_results({"a": 0.02, "b": 0.05}, n=300, horizon=21, seed=15)
    outcome = model_confidence_set(results, horizon=21, n_bootstrap=200, seed=0)
    assert outcome.block_size == 21
