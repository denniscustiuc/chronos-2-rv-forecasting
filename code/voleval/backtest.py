"""Walk-forward backtest harness.

One harness drives every model.  It knows nothing about how the realized-
volatility series was computed -- a VOLARE ``rv5`` series and a yfinance
``rv_gk`` series are interchangeable inputs -- and nothing about which model it
is driving beyond the :class:`~volmodels.base.Forecaster` interface.  Adding
Chronos-2 or Kronos later means writing one more ``Forecaster`` subclass and
registering it; the harness, the metrics and the significance tests do not
change.

Two rules govern the loop, and they are not the same rule:

**Model inputs may never see the future.**  At origin ``t`` the slice handed to
``fit``/``update`` ends at ``t`` inclusive.  This is enforced structurally (the
slice is built by position) and verified by
``tests/test_backtest.py::test_fit_never_sees_the_future``, which spies on every
call the harness makes.

**The target legitimately does.**  The target for origin ``t`` and horizon ``h``
is the series value at ``t + h``.  That is future data by definition -- it is
what makes the exercise a forecast rather than a fit.  The lookahead rule
constrains inputs, not targets.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from volmodels.base import Forecaster, NotEnoughData, quantile_column

logger = logging.getLogger(__name__)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "RESULT_COLUMNS",
    "walk_forward",
    "run_backtest",
]

WindowType = Literal["expanding", "rolling"]
TargetType = Literal["point", "average"]

#: Fixed leading columns of the tidy results frame.  Quantile columns
#: (``q0.05`` ...) follow, in ascending level order.
RESULT_COLUMNS = (
    "model",
    "ticker",
    "origin_date",
    "target_date",
    "horizon",
    "forecast",
    "realized",
)


@dataclass(frozen=True)
class BacktestConfig:
    """Protocol for one walk-forward run.

    Attributes:
        horizons: Forecast horizons in trading days.  Note that on a ``w``-day
            rolling RV series any horizon below ``w`` compares a forecast
            against a target that shares input days with it, which inflates
            apparent skill; ``h >= w`` is the honest comparison.
        window: ``"expanding"`` grows the training set from the start of the
            sample; ``"rolling"`` keeps a fixed-length trailing window.
        min_train_size: Observations required before the first origin.
        rolling_window_size: Length of the trailing window when
            ``window="rolling"``; defaults to ``min_train_size``.
        refit_frequency: Re-estimate parameters every ``k`` origins.  Between
            refits the harness calls :meth:`~volmodels.base.Forecaster.update`,
            which refreshes the conditioning data without re-estimating -- so
            forecasts stay conditioned on data current as of the origin even
            when the coefficients are a few days stale.  ``1`` refits at every
            origin.
        start_origin: Skip origins before this date.
        end_origin: Skip origins after this date.
        max_origins: Cap the number of origins (for smoke runs).
        target: Which quantity the forecast is scored against.

            ``"point"`` -- the value at ``t+h`` exactly.  **This is Brini
            (2026)'s definition** and the default: successive targets do not
            overlap, so the target series keeps the daily dynamics intact and
            nothing about the evaluation is inflated by shared observations.

            ``"average"`` -- ``sqrt(mean(RV_{t+1..t+h}))``, the average variance
            *over* the horizon.  This is the economically relevant quantity for
            an h-day risk or option horizon, and it is the target our own
            methodological track prefers.  It is kept strictly separate from the
            Brini replication because its targets overlap across origins, which
            is exactly what Brini's design avoids -- comparing our "average"
            numbers to the paper's would not be like-for-like.
    """

    horizons: tuple[int, ...] = (1, 5, 21)
    window: WindowType = "expanding"
    min_train_size: int = 250
    rolling_window_size: int | None = None
    refit_frequency: int = 1
    start_origin: str | None = None
    end_origin: str | None = None
    max_origins: int | None = None
    target: TargetType = "point"

    def __post_init__(self) -> None:
        if self.target not in ("point", "average"):
            raise ValueError(f"target must be 'point' or 'average', got {self.target!r}")
        if not self.horizons:
            raise ValueError("horizons must not be empty")
        if any(h < 1 for h in self.horizons):
            raise ValueError("every horizon must be >= 1")
        if self.window not in ("expanding", "rolling"):
            raise ValueError(f"window must be 'expanding' or 'rolling', got {self.window!r}")
        if self.min_train_size < 2:
            raise ValueError("min_train_size must be >= 2")
        if self.refit_frequency < 1:
            raise ValueError("refit_frequency must be >= 1")
        if self.window == "rolling" and self.rolling_window_size is not None:
            if self.rolling_window_size < 2:
                raise ValueError("rolling_window_size must be >= 2")

    @property
    def effective_window_size(self) -> int:
        """Trailing window length actually used when ``window="rolling"``."""
        return self.rolling_window_size or self.min_train_size

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view, for the run manifest."""
        return {
            "horizons": list(self.horizons),
            "window": self.window,
            "min_train_size": self.min_train_size,
            "rolling_window_size": self.rolling_window_size,
            "refit_frequency": self.refit_frequency,
            "start_origin": self.start_origin,
            "end_origin": self.end_origin,
            "max_origins": self.max_origins,
            "target": self.target,
        }


@dataclass
class BacktestResult:
    """Output of a backtest run.

    Attributes:
        results: Tidy frame, one row per (model, ticker, origin, horizon).
        config: The protocol that produced it.
        diagnostics: Per (model, ticker) counts of failed fits, failed
            forecasts and floored point forecasts.
        elapsed_seconds: Wall-clock runtime.
    """

    results: pd.DataFrame
    config: BacktestConfig
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    elapsed_seconds: float = 0.0

    @property
    def models(self) -> list[str]:
        """Names of the models present in the results."""
        return sorted(self.results["model"].unique()) if not self.results.empty else []

    def to_parquet(self, path) -> None:
        """Write the results frame to parquet."""
        self.results.to_parquet(path)


def _normalise_forecasters(
    forecasters: Iterable[Forecaster | Callable[[], Forecaster]],
) -> list[Callable[[], Forecaster]]:
    """Turn a mixed iterable of instances and factories into factories.

    Each ticker gets a *fresh* model, so state cannot leak between assets.  A
    bare instance is deep-copied rather than reused for exactly that reason.
    """
    factories: list[Callable[[], Forecaster]] = []
    for item in forecasters:
        if isinstance(item, Forecaster):
            factories.append(lambda template=item: copy.deepcopy(template))
        elif callable(item):
            factories.append(item)
        else:
            raise TypeError(f"expected a Forecaster or a zero-argument factory, got {item!r}")
    return factories


def _origin_positions(values: pd.Series, config: BacktestConfig) -> list[int]:
    """Positions in ``values`` that serve as forecast origins.

    An origin needs ``min_train_size`` observations behind it (inclusive) and at
    least one observation ahead of it to score against.
    """
    n = len(values)
    first = config.min_train_size - 1
    last = n - 2  # need values[i + 1] to exist for the shortest horizon
    if first > last:
        return []

    positions = list(range(first, last + 1))
    index = values.index

    if config.start_origin is not None:
        start = pd.Timestamp(config.start_origin)
        positions = [i for i in positions if index[i] >= start]
    if config.end_origin is not None:
        end = pd.Timestamp(config.end_origin)
        positions = [i for i in positions if index[i] <= end]
    if config.max_origins is not None:
        positions = positions[: config.max_origins]
    return positions


def _realized_target(values: pd.Series, position: int, h: int, config: BacktestConfig) -> float:
    """The realisation an origin-``position``, horizon-``h`` forecast is scored against.

    Deliberately reaches *forward* of the origin -- that is what makes this a
    forecast target rather than a fit.  The no-lookahead rule constrains model
    inputs, never the target.

    ``"point"`` returns the value at ``t+h``.  ``"average"`` returns the
    root-mean variance over ``t+1..t+h``, i.e. the volatility of the whole
    horizon rather than of its final day.
    """
    if config.target == "point":
        return float(values.iloc[position + h])
    window = values.iloc[position + 1 : position + h + 1].to_numpy(dtype="float64")
    return float(np.sqrt(np.mean(window**2)))


def _training_slice(values: pd.Series, position: int, config: BacktestConfig) -> pd.Series:
    """Build the training slice for the origin at ``position``.

    The slice ends at ``position`` **inclusive** and never extends beyond it.
    This single expression is the structural guarantee of no input lookahead --
    everything else in the harness is bookkeeping around it.
    """
    if config.window == "rolling":
        start = max(0, position + 1 - config.effective_window_size)
        return values.iloc[start : position + 1]
    return values.iloc[: position + 1]


def walk_forward(
    series: pd.Series,
    forecasters: Iterable[Forecaster | Callable[[], Forecaster]],
    config: BacktestConfig | None = None,
    *,
    ticker: str = "SERIES",
    exog: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Run the walk-forward loop for one series.

    Args:
        series: Realized-volatility series indexed by trading date.  Source-
            agnostic: anything with a ``DatetimeIndex`` and positive values.
        forecasters: Model instances or zero-argument factories.
        config: Backtest protocol; defaults apply if ``None``.
        ticker: Label written into the ``ticker`` column.
        exog: Optional companion measures (bipower variation, quarticity,
            semivariances, covariates) indexed by the same dates.  It is sliced
            to the training window exactly as ``series`` is, so it is subject
            to the identical no-lookahead guarantee -- models that ignore it are
            unaffected.

    Returns:
        ``(results_frame, diagnostics)``.
    """
    config = config or BacktestConfig()
    values = series.dropna().astype("float64").sort_index()
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("series must be indexed by a DatetimeIndex")

    if exog is not None:
        if not isinstance(exog.index, pd.DatetimeIndex):
            raise TypeError("exog must be indexed by a DatetimeIndex")
        exog = exog.sort_index()

    positions = _origin_positions(values, config)
    if not positions:
        logger.warning(
            "%s: no valid origins (%d observations, min_train_size=%d)",
            ticker, len(values), config.min_train_size,
        )
        return pd.DataFrame(), []

    n = len(values)
    index = values.index
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for factory in _normalise_forecasters(forecasters):
        model = factory()
        model.reset()
        name = model.name
        failed_fits = failed_forecasts = 0
        started = time.perf_counter()

        for step, position in enumerate(positions):
            train = _training_slice(values, position, config)
            origin = index[position]
            # Reindexing onto the training slice is what makes the exogenous
            # channel share the target's no-lookahead guarantee: it cannot
            # reach a date the training slice does not contain.
            train_exog = None if exog is None else exog.reindex(train.index)

            try:
                if step % config.refit_frequency == 0:
                    model.fit(train, train_exog)
                else:
                    model.update(train, train_exog)
                fitted = True
            except (NotEnoughData, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                fitted = False
                failed_fits += 1
                if failed_fits == 1:
                    logger.warning("%s/%s: fit failed at %s (%s)", ticker, name, origin.date(), exc)

            for h in config.horizons:
                target_position = position + h
                if target_position >= n:
                    continue  # no realisation to score against

                point = np.nan
                quantiles: Mapping[float, float] = {}
                if fitted:
                    try:
                        prediction = model.forecast(h)
                        point = prediction.point
                        quantiles = prediction.quantiles
                        if prediction.origin != origin:
                            raise RuntimeError(
                                f"{name} reported origin {prediction.origin} at origin {origin}"
                            )
                    except (NotEnoughData, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        failed_forecasts += 1
                        if failed_forecasts == 1:
                            logger.warning(
                                "%s/%s: forecast(h=%d) failed at %s (%s)",
                                ticker, name, h, origin.date(), exc,
                            )

                row: dict[str, object] = {
                    "model": name,
                    "ticker": ticker,
                    "origin_date": origin,
                    "target_date": index[target_position],
                    "horizon": h,
                    "forecast": point,
                    "realized": _realized_target(values, position, h, config),
                }
                row.update({quantile_column(lvl): val for lvl, val in quantiles.items()})
                rows.append(row)

        elapsed = time.perf_counter() - started
        diagnostics.append(
            {
                "model": name,
                "ticker": ticker,
                "origins": len(positions),
                "failed_fits": failed_fits,
                "failed_forecasts": failed_forecasts,
                "floored_forecasts": model.n_floored,
                "insanity_filtered": model.n_insane,
                "seconds": round(elapsed, 3),
            }
        )
        if model.n_insane:
            logger.warning(
                "%s/%s: insanity filter replaced %d implausible forecast(s) with the in-sample mean",
                ticker, name, model.n_insane,
            )
        logger.info(
            "%s/%s: %d origins in %.2fs (%d failed fits, %d floored, %d insanity-filtered)",
            ticker, name, len(positions), elapsed, failed_fits, model.n_floored, model.n_insane,
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        quantile_columns = sorted(
            (c for c in frame.columns if c not in RESULT_COLUMNS),
            key=lambda c: float(c[1:]),
        )
        frame = frame[list(RESULT_COLUMNS) + quantile_columns]
    return frame, diagnostics


def run_backtest(
    series_by_ticker: Mapping[str, pd.Series],
    forecasters: Iterable[Forecaster | Callable[[], Forecaster]],
    config: BacktestConfig | None = None,
    *,
    exog_by_ticker: Mapping[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """Run the walk-forward backtest across several tickers.

    Args:
        series_by_ticker: ``{ticker: rv_series}``.  Each series is handled
            independently with a fresh model instance.
        forecasters: Model instances or zero-argument factories.
        config: Backtest protocol; defaults apply if ``None``.
        exog_by_ticker: Optional ``{ticker: exog_frame}`` of companion
            measures.  Tickers absent from the mapping simply get ``None``.

    Returns:
        A :class:`BacktestResult` holding the pooled tidy results frame.
    """
    config = config or BacktestConfig()
    factories = _normalise_forecasters(forecasters)
    started = time.perf_counter()

    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for ticker, series in series_by_ticker.items():
        ticker_exog = None if exog_by_ticker is None else exog_by_ticker.get(ticker)
        frame, ticker_diagnostics = walk_forward(
            series, factories, config, ticker=ticker, exog=ticker_exog
        )
        if not frame.empty:
            frames.append(frame)
        diagnostics.extend(ticker_diagnostics)

    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elapsed = time.perf_counter() - started
    logger.info(
        "backtest complete: %d rows across %d ticker(s) in %.1fs",
        len(results), len(series_by_ticker), elapsed,
    )
    return BacktestResult(
        results=results,
        config=config,
        diagnostics=pd.DataFrame(diagnostics),
        elapsed_seconds=elapsed,
    )
