"""Harness tests.

The lookahead guards are the point of this file.  Everything else in the
project can be wrong in ways that produce a bad forecast; a lookahead leak
produces a *good-looking* forecast, which is far more dangerous.  Two
independent tests cover it:

* :func:`test_fit_never_sees_the_future` -- a spy model records every date the
  harness ever showed it and asserts none post-dates the origin.
* :func:`test_poisoning_the_future_cannot_change_past_forecasts` -- a
  behavioural check that corrupting the tail of the series leaves earlier
  forecasts bit-identical.  This one catches leaks the spy would miss, because
  it tests consequences rather than inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volmodels import LogHAR, RandomWalk, baseline_factories
from volmodels.base import Forecast, Forecaster
from voleval.backtest import (
    RESULT_COLUMNS,
    BacktestConfig,
    _origin_positions,
    _training_slice,
    run_backtest,
    walk_forward,
)

from .conftest import simulate_har


class SpyForecaster(Forecaster):
    """Records every history the harness hands it, for the lookahead audit.

    Pass it to the harness as ``[lambda: spy]`` rather than ``[spy]``: a bare
    instance is deep-copied so state cannot leak between tickers, and the copy
    would record the calls where the test cannot see them.
    """

    def __init__(self) -> None:
        super().__init__("spy")
        self.fit_origins: list[pd.Timestamp] = []
        self.update_origins: list[pd.Timestamp] = []
        self.seen_max: list[pd.Timestamp] = []
        self.seen_lengths: list[int] = []
        self.seen_exog_max: list[pd.Timestamp] = []

    def _record(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        self.seen_max.append(history.index.max())
        self.seen_lengths.append(len(history))
        if exog is not None and len(exog):
            self.seen_exog_max.append(exog.index.max())

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        self._store_history(history, exog)
        self.fit_origins.append(history.index.max())
        self._record(history, exog)

    def update(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        self._store_history(history, exog)
        self.update_origins.append(history.index.max())
        self._record(history, exog)

    def forecast(self, h: int) -> Forecast:
        return Forecast(
            point=float(self._history.iloc[-1]),
            model=self.name,
            origin=self.origin,
            horizon=h,
        )


@pytest.fixture
def series() -> pd.Series:
    return simulate_har(n_days=400, seed=42)


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(horizons=(1, 5, 21), min_train_size=100, refit_frequency=1)


# --------------------------------------------------------------------------- #
# Lookahead guards -- the most important tests in the project
# --------------------------------------------------------------------------- #

def test_fit_never_sees_the_future(series, config):
    """Every training slice must end exactly at its origin, never later."""
    spy = SpyForecaster()
    results, _ = walk_forward(series, [lambda: spy], config, ticker="TEST")

    expected_origins = [series.index[i] for i in _origin_positions(series, config)]
    assert spy.seen_max, "the harness never called fit()"
    assert len(spy.seen_max) == len(expected_origins)

    for seen, origin in zip(spy.seen_max, expected_origins):
        assert seen <= origin, f"model saw {seen}, which is after origin {origin}"
        assert seen == origin, f"model saw only up to {seen} at origin {origin}"

    # And the recorded origins in the results frame agree with what the spy saw.
    recorded = sorted(results["origin_date"].unique())
    assert recorded == expected_origins


def test_fit_never_sees_the_future_under_every_protocol(series):
    """The guarantee must hold for rolling windows and lazy refits too."""
    protocols = [
        BacktestConfig(horizons=(1,), min_train_size=60, window="expanding", refit_frequency=1),
        BacktestConfig(horizons=(1, 21), min_train_size=60, window="expanding", refit_frequency=7),
        BacktestConfig(
            horizons=(1, 5), min_train_size=60, window="rolling",
            rolling_window_size=80, refit_frequency=3,
        ),
    ]
    for protocol in protocols:
        spy = SpyForecaster()
        walk_forward(series, [lambda: spy], protocol, ticker="TEST")
        expected = [series.index[i] for i in _origin_positions(series, protocol)]

        assert len(spy.seen_max) == len(expected), protocol
        for seen, origin in zip(spy.seen_max, expected):
            assert seen == origin, f"{protocol.window}/refit={protocol.refit_frequency}: {seen} != {origin}"


def test_poisoning_the_future_cannot_change_past_forecasts(series, config):
    """Corrupt the tail of the series; earlier forecasts must be unchanged.

    A behavioural leak detector.  If any model or the harness peeked beyond the
    origin -- through a full-sample mean, a centred rolling window, a scaler fit
    on everything -- these two runs would diverge.
    """
    cutoff = 300
    poisoned = series.copy()
    poisoned.iloc[cutoff:] = 99.0

    factories = baseline_factories(include_mem=True)
    clean_results, _ = walk_forward(series, factories, config, ticker="TEST")
    dirty_results, _ = walk_forward(poisoned, factories, config, ticker="TEST")

    boundary = series.index[cutoff]
    keys = ["model", "origin_date", "horizon"]
    clean = clean_results[clean_results["origin_date"] < boundary].set_index(keys)["forecast"]
    dirty = dirty_results[dirty_results["origin_date"] < boundary].set_index(keys)["forecast"]

    assert len(clean) > 0
    pd.testing.assert_series_equal(clean.sort_index(), dirty.sort_index())


def test_training_slice_ends_at_the_origin(series):
    """Unit-level check of the single expression the guarantee rests on."""
    for window, size in (("expanding", None), ("rolling", 50)):
        config = BacktestConfig(window=window, rolling_window_size=size, min_train_size=60)
        for position in (60, 150, len(series) - 2):
            train = _training_slice(series, position, config)
            assert train.index[-1] == series.index[position]
            assert train.index.max() <= series.index[position]
            if window == "rolling":
                assert len(train) <= 50


def test_target_is_the_value_h_steps_after_the_origin(series, config):
    """The target legitimately uses future data -- verify it is the right one."""
    results, _ = walk_forward(series, [RandomWalk()], config, ticker="TEST")
    positions = {date: i for i, date in enumerate(series.index)}

    sample = results.sample(min(50, len(results)), random_state=0)
    for row in sample.itertuples():
        origin_position = positions[row.origin_date]
        expected_target = series.index[origin_position + row.horizon]
        assert row.target_date == expected_target
        assert row.realized == pytest.approx(float(series.iloc[origin_position + row.horizon]))
        assert row.target_date > row.origin_date


# --------------------------------------------------------------------------- #
# Protocol mechanics
# --------------------------------------------------------------------------- #

def test_refit_frequency_controls_fits_not_conditioning(series):
    """With refit_frequency=k the harness fits every k-th origin and updates otherwise.

    Crucially, ``update`` still receives the full history through the current
    origin -- stale *parameters* are the point, stale *data* would be a bug.
    """
    config = BacktestConfig(horizons=(1,), min_train_size=100, refit_frequency=10)
    spy = SpyForecaster()
    walk_forward(series, [lambda: spy], config, ticker="TEST")

    n_origins = len(_origin_positions(series, config))
    assert len(spy.fit_origins) == int(np.ceil(n_origins / 10))
    assert len(spy.update_origins) == n_origins - len(spy.fit_origins)

    # Every slice, fit or update, grew by exactly one observation.
    assert spy.seen_lengths == list(range(100, 100 + n_origins))


def test_expanding_and_rolling_windows_differ(series):
    expanding = BacktestConfig(horizons=(1,), min_train_size=100, window="expanding")
    rolling = BacktestConfig(
        horizons=(1,), min_train_size=100, window="rolling", rolling_window_size=100
    )

    spy_e, spy_r = SpyForecaster(), SpyForecaster()
    walk_forward(series, [lambda: spy_e], expanding, ticker="TEST")
    walk_forward(series, [lambda: spy_r], rolling, ticker="TEST")

    assert spy_e.seen_lengths[-1] > spy_e.seen_lengths[0]
    assert set(spy_r.seen_lengths) == {100}


def test_min_train_size_sets_the_first_origin(series):
    config = BacktestConfig(horizons=(1,), min_train_size=250)
    results, _ = walk_forward(series, [RandomWalk()], config, ticker="TEST")
    assert results["origin_date"].min() == series.index[249]


def test_origin_bounds_are_respected(series):
    config = BacktestConfig(
        horizons=(1,), min_train_size=100,
        start_origin=str(series.index[200].date()), end_origin=str(series.index[250].date()),
    )
    results, _ = walk_forward(series, [RandomWalk()], config, ticker="TEST")
    assert results["origin_date"].min() >= series.index[200]
    assert results["origin_date"].max() <= series.index[250]


def test_max_origins_caps_the_run(series):
    config = BacktestConfig(horizons=(1,), min_train_size=100, max_origins=12)
    results, _ = walk_forward(series, [RandomWalk()], config, ticker="TEST")
    assert results["origin_date"].nunique() == 12


def test_no_row_is_emitted_without_a_realisation(series, config):
    """The last few origins have no t+21 value; they must be dropped, not faked."""
    results, _ = walk_forward(series, [RandomWalk()], config, ticker="TEST")
    assert results["realized"].notna().all()

    long_horizon = results[results["horizon"] == 21]
    assert long_horizon["origin_date"].max() == series.index[len(series) - 1 - 21]


def test_too_short_a_series_yields_no_origins():
    short = simulate_har(n_days=30, seed=1)
    config = BacktestConfig(min_train_size=100)
    results, diagnostics = walk_forward(short, [RandomWalk()], config, ticker="TEST")
    assert results.empty
    assert diagnostics == []


# --------------------------------------------------------------------------- #
# Results schema and multi-ticker runs
# --------------------------------------------------------------------------- #

def test_results_schema(series, config):
    results, _ = walk_forward(series, [LogHAR()], config, ticker="TEST")

    assert list(results.columns[: len(RESULT_COLUMNS)]) == list(RESULT_COLUMNS)
    quantile_columns = list(results.columns[len(RESULT_COLUMNS):])
    assert quantile_columns == [
        "q0.05", "q0.1", "q0.2", "q0.3", "q0.4", "q0.5", "q0.6", "q0.7", "q0.8", "q0.9", "q0.95",
    ]
    assert set(results["horizon"]) == {1, 5, 21}
    assert results["ticker"].unique().tolist() == ["TEST"]
    assert pd.api.types.is_datetime64_any_dtype(results["origin_date"])


def test_run_backtest_across_tickers(series, config):
    panel = {"AAA": series, "BBB": simulate_har(n_days=400, seed=7)}
    outcome = run_backtest(panel, baseline_factories(), config)

    assert set(outcome.results["ticker"]) == {"AAA", "BBB"}
    assert outcome.models == ["ewma_0.94", "har", "historical_mean", "log_har", "random_walk"]
    assert not outcome.diagnostics.empty
    assert outcome.elapsed_seconds > 0

    counts = outcome.results.groupby(["ticker", "model"], observed=True).size().unstack()
    assert (counts.nunique(axis=1) == 1).all(), "models saw different numbers of origins"


def test_models_do_not_leak_state_between_tickers(config):
    """Each ticker gets a fresh model, so a second asset cannot inherit a fit."""
    flat = pd.Series(
        np.full(400, 0.2), index=pd.bdate_range("2015-01-01", periods=400, name="date")
    )
    spiky = simulate_har(n_days=400, seed=3) * 3.0

    outcome = run_backtest({"FLAT": flat, "SPIKY": spiky}, [RandomWalk], config)
    by_ticker = outcome.results.groupby("ticker", observed=True)["forecast"].mean()

    assert by_ticker["FLAT"] == pytest.approx(0.2)
    assert by_ticker["SPIKY"] > 0.4


def test_a_failing_model_does_not_abort_the_run(series, config):
    """One broken model must not take the whole backtest down with it."""

    class Broken(Forecaster):
        def __init__(self) -> None:
            super().__init__("broken")

        def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
            self._store_history(history, exog)

        def forecast(self, h: int) -> Forecast:
            raise ValueError("this model is broken on purpose")

    results, diagnostics = walk_forward(series, [Broken(), RandomWalk()], config, ticker="TEST")

    broken_rows = results[results["model"] == "broken"]
    assert len(broken_rows) > 0
    assert broken_rows["forecast"].isna().all()
    assert results[results["model"] == "random_walk"]["forecast"].notna().all()

    broken_diagnostics = next(d for d in diagnostics if d["model"] == "broken")
    assert broken_diagnostics["failed_forecasts"] > 0


def test_config_validation():
    with pytest.raises(ValueError, match="horizons must not be empty"):
        BacktestConfig(horizons=())
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        BacktestConfig(horizons=(0,))
    with pytest.raises(ValueError, match="window must be"):
        BacktestConfig(window="sliding")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refit_frequency"):
        BacktestConfig(refit_frequency=0)
    with pytest.raises(ValueError, match="min_train_size"):
        BacktestConfig(min_train_size=1)


def test_harness_accepts_instances_and_factories(series, config):
    results, _ = walk_forward(series, [RandomWalk(), RandomWalk], config, ticker="TEST")
    assert set(results["model"]) == {"random_walk"}


def test_harness_rejects_a_non_forecaster(series, config):
    with pytest.raises(TypeError, match="Forecaster"):
        walk_forward(series, ["not a model"], config, ticker="TEST")


# --------------------------------------------------------------------------- #
# End-to-end sanity: the fitted benchmark should clear the naive floors
# --------------------------------------------------------------------------- #

def test_log_har_beats_the_naive_floors_on_har_data():
    """On data generated by a HAR process, Log-HAR must clear the naive floors.

    This is the sanity check that the whole stack -- forecaster, harness,
    metrics -- is wired up the right way round.  It is asserted rather than
    merely printed because the data-generating process is HAR by construction,
    so a correctly-specified model genuinely should win here.  On *real* RV
    series the same comparison is an open empirical question, not a guarantee;
    at horizons beyond the RV window the random walk is a serious opponent.
    """
    from voleval.metrics import summarize

    series = {
        "AAA": simulate_har(n_days=900, seed=101),
        "BBB": simulate_har(n_days=900, seed=202),
    }
    config = BacktestConfig(horizons=(1, 5), min_train_size=250, refit_frequency=5)
    outcome = run_backtest(series, baseline_factories(), config)

    pooled = summarize(outcome.results)["pooled"].set_index(["model", "horizon"])
    for horizon in (1, 5):
        log_har = pooled.loc[("log_har", horizon), "qlike"]
        for floor in ("random_walk", "historical_mean"):
            assert log_har < pooled.loc[(floor, horizon), "qlike"], (
                f"log_har lost to {floor} at h={horizon} on HAR-generated data"
            )
        assert pooled.loc[("log_har", horizon), "qlike_ratio"] == pytest.approx(1.0)


def test_every_model_runs_end_to_end_without_failures():
    """No model may silently fail its fits on a clean, well-behaved series."""
    series = {"AAA": simulate_har(n_days=700, seed=303)}
    config = BacktestConfig(horizons=(1, 21), min_train_size=250, refit_frequency=10)
    outcome = run_backtest(series, baseline_factories(include_mem=True), config)

    assert (outcome.diagnostics["failed_fits"] == 0).all(), outcome.diagnostics
    assert (outcome.diagnostics["failed_forecasts"] == 0).all(), outcome.diagnostics
    assert outcome.results["forecast"].notna().all()
    assert (outcome.results["forecast"] > 0).all()
