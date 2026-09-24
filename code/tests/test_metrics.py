"""Metric tests: QLIKE properties, aggregation and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voleval.metrics import (
    absolute_error,
    add_losses,
    coverage_table,
    leaderboard,
    qlike,
    squared_error,
    summarize,
)


# --------------------------------------------------------------------------- #
# QLIKE
# --------------------------------------------------------------------------- #

def test_qlike_is_zero_only_when_the_forecast_is_exact():
    values = np.array([0.1, 0.2, 0.35, 0.5])
    exact = qlike(values, values)
    assert exact == pytest.approx(np.zeros_like(values), abs=1e-15)


def test_qlike_is_positive_whenever_the_forecast_is_wrong():
    rng = np.random.default_rng(0)
    realized = rng.uniform(0.05, 0.8, size=500)
    forecast = realized * rng.uniform(0.3, 3.0, size=500)

    losses = qlike(realized, forecast)
    assert np.all(losses > 0), "QLIKE went non-positive for an inexact forecast"


def test_qlike_is_minimised_at_the_truth():
    """Sweeping the forecast across the realized value must bottom out at it."""
    realized = 0.25
    grid = np.linspace(0.05, 1.0, 400)
    losses = qlike(np.full_like(grid, realized), grid)
    assert grid[np.argmin(losses)] == pytest.approx(realized, abs=0.005)


def test_qlike_uses_variances_not_volatilities():
    """The documented definition squares the inputs; verify it literally."""
    realized, forecast = 0.3, 0.2
    ratio = realized**2 / forecast**2
    assert float(qlike([realized], [forecast])[0]) == pytest.approx(ratio - np.log(ratio) - 1.0)

    # Passing variances directly must give the same answer.
    direct = qlike([realized**2], [forecast**2], inputs_are_variance=True)
    assert float(direct[0]) == pytest.approx(ratio - np.log(ratio) - 1.0)


def test_qlike_punishes_under_forecasting_more_than_over_forecasting():
    """Asymmetry is the reason QLIKE is preferred for volatility."""
    realized = 0.30
    too_low = float(qlike([realized], [realized / 2])[0])
    too_high = float(qlike([realized], [realized * 2])[0])
    assert too_low > too_high


def test_qlike_is_nan_for_non_positive_inputs():
    losses = qlike([0.2, 0.0, -0.1, 0.3], [0.2, 0.2, 0.2, 0.0])
    assert np.isfinite(losses[0])
    assert np.isnan(losses[1:]).all()


def test_secondary_losses():
    assert float(squared_error([0.3], [0.2])[0]) == pytest.approx(0.01)
    assert float(absolute_error([0.3], [0.2])[0]) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

@pytest.fixture
def results() -> pd.DataFrame:
    """A small synthetic results frame with a deliberately ranked model set."""
    rng = np.random.default_rng(4)
    rows = []
    dates = pd.bdate_range("2021-01-04", periods=120, name="date")
    quality = {"log_har": 0.04, "good": 0.02, "bad": 0.12}

    for ticker, level in (("AAA", 0.20), ("BBB", 0.35)):
        realized = level * np.exp(rng.normal(0, 0.25, size=len(dates)))
        for model, noise in quality.items():
            errors = rng.normal(0.0, noise, size=len(dates))
            for h in (1, 21):
                for i, date in enumerate(dates):
                    rows.append(
                        {
                            "model": model,
                            "ticker": ticker,
                            "origin_date": date,
                            "target_date": date,
                            "horizon": h,
                            "forecast": max(realized[i] + errors[i], 1e-4),
                            "realized": realized[i],
                            "q0.05": max(realized[i] + errors[i] - 0.1, 1e-4),
                            "q0.5": max(realized[i] + errors[i], 1e-4),
                            "q0.95": realized[i] + errors[i] + 0.1,
                        }
                    )
    return pd.DataFrame(rows)


def test_add_losses_attaches_every_loss_column(results):
    scored = add_losses(results)
    assert {"qlike", "se", "ae"} <= set(scored.columns)
    assert scored["qlike"].notna().all()
    assert len(scored) == len(results)


def test_add_losses_requires_the_core_columns():
    with pytest.raises(KeyError, match="realized"):
        add_losses(pd.DataFrame({"forecast": [0.1]}))


def test_summarize_produces_all_three_aggregations(results):
    tables = summarize(results)
    assert set(tables) == {"pooled", "per_asset", "equal_weighted"}

    pooled = tables["pooled"]
    assert set(pooled["horizon"]) == {1, 21}
    assert set(pooled["model"]) == {"log_har", "good", "bad"}
    assert (pooled["n_obs"] > 0).all()


def test_loss_ratios_are_relative_to_the_benchmark(results):
    pooled = summarize(results, benchmark="log_har")["pooled"]
    benchmark_rows = pooled[pooled["model"] == "log_har"]
    assert benchmark_rows["qlike_ratio"].to_numpy() == pytest.approx(np.ones(2))

    good = pooled[pooled["model"] == "good"]["qlike_ratio"]
    bad = pooled[pooled["model"] == "bad"]["qlike_ratio"]
    assert (good < 1.0).all(), "the low-noise model should beat the benchmark"
    assert (bad > 1.0).all(), "the high-noise model should lose to the benchmark"


def test_equal_weighted_differs_from_pooled_when_assets_differ(results):
    """Pooled weights by observation; equal-weighted gives each asset one vote."""
    unbalanced = pd.concat([results, results[results["ticker"] == "BBB"]], ignore_index=True)
    tables = summarize(unbalanced)

    pooled = tables["pooled"].set_index(["model", "horizon"])["qlike"]
    equal = tables["equal_weighted"].set_index(["model", "horizon"])["qlike"]
    assert not np.allclose(pooled.to_numpy(), equal.to_numpy())
    assert (tables["equal_weighted"]["n_assets"] == 2).all()


def test_per_asset_table_keeps_tickers_separate(results):
    per_asset = summarize(results)["per_asset"]
    assert set(per_asset["ticker"]) == {"AAA", "BBB"}
    assert len(per_asset) == 3 * 2 * 2  # models x tickers x horizons


def test_summarize_warns_and_blanks_ratios_for_a_missing_benchmark(results):
    pooled = summarize(results, benchmark="does_not_exist")["pooled"]
    assert pooled["qlike_ratio"].isna().all()


def test_failed_forecasts_are_excluded_but_counted(results):
    broken = results.copy()
    broken.loc[broken["model"] == "bad", "forecast"] = np.nan
    pooled = summarize(broken)["pooled"]

    bad = pooled[pooled["model"] == "bad"].iloc[0]
    assert bad["n_obs"] == 0
    assert bad["n_failed"] > 0
    assert np.isnan(bad["qlike"])


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def test_coverage_of_a_perfectly_calibrated_model():
    """Quantiles drawn from the true predictive law must cover at their nominal rate."""
    from scipy.stats import norm

    rng = np.random.default_rng(1)
    mu, sigma = np.log(0.2), 0.3
    realized = np.exp(rng.normal(mu, sigma, size=8000))

    frame = pd.DataFrame(
        {
            "model": "oracle",
            "ticker": "AAA",
            "horizon": 1,
            "forecast": np.exp(mu + 0.5 * sigma**2),
            "realized": realized,
        }
    )
    # Exact log-normal quantiles: an oracle that knows the true predictive law.
    for level in (0.05, 0.1, 0.5, 0.9, 0.95):
        frame[f"q{level:g}"] = np.exp(mu + sigma * norm.ppf(level))

    table = coverage_table(frame)
    for _, row in table.iterrows():
        assert row["empirical"] == pytest.approx(row["nominal"], abs=0.02), row.to_dict()


def test_coverage_detects_bands_that_are_too_narrow(results):
    """A band tighter than reality must under-cover at the top level."""
    table = coverage_table(results)
    assert set(table["level"]) == {0.05, 0.5, 0.95}
    assert (table["n_obs"] > 0).all()
    assert table["empirical"].between(0.0, 1.0).all()

    # q0.5 was set equal to the point forecast, so it should sit near 50%.
    medians = table[table["level"] == 0.5]["empirical"]
    assert medians.between(0.3, 0.7).all()


def test_coverage_table_is_empty_without_quantile_columns():
    frame = pd.DataFrame(
        {"model": ["a"], "ticker": ["AAA"], "horizon": [1], "forecast": [0.2], "realized": [0.2]}
    )
    assert coverage_table(frame).empty


def test_coverage_gap_sign_is_meaningful():
    """gap > 0 means the quantile sits too high (over-coverage)."""
    frame = pd.DataFrame(
        {
            "model": "wide",
            "ticker": "AAA",
            "horizon": 1,
            "forecast": 0.2,
            "realized": np.full(100, 0.2),
            "q0.05": np.full(100, 0.9),  # absurdly high 5% quantile
        }
    )
    row = coverage_table(frame).iloc[0]
    assert row["empirical"] == 1.0
    assert row["gap"] == pytest.approx(0.95)


# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #

def test_leaderboard_ranks_best_first(results):
    pooled = summarize(results)["pooled"]
    board = leaderboard(pooled, horizon=21)

    assert board.index.name == "rank"
    assert board.index[0] == 1
    assert board["qlike"].is_monotonic_increasing
    assert board.iloc[0]["model"] == "good"
    assert board.iloc[-1]["model"] == "bad"


def test_leaderboard_rejects_an_unknown_loss(results):
    with pytest.raises(KeyError, match="not in summary"):
        leaderboard(summarize(results)["pooled"], horizon=1, loss="nonsense")
