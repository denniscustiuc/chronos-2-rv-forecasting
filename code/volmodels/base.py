"""The shared forecaster interface every model in this project conforms to.

This is the glue of the whole evaluation stack.  The baselines here, and
Chronos-2 / Kronos later, all implement :class:`Forecaster` and all return
:class:`Forecast` objects, so :mod:`voleval.backtest` never needs to know which
model it is driving and no model ever needs its own harness.

The contract, in three rules:

1. ``fit(history)`` receives the realized-volatility series **up to and
   including the forecast origin ``t``, and nothing after it**.  The harness
   enforces this; see ``tests/test_backtest.py::test_fit_never_sees_the_future``.
2. ``forecast(h)`` returns a point forecast for the value at ``t + h`` steps,
   plus optional quantiles.  It must not consult anything the last ``fit`` /
   ``update`` call did not provide.
3. Everything is expressed in the units of the input series.  We feed it
   realized **volatility**, so forecasts come back as volatility; the QLIKE
   metric squares them internally.

A model that needs periodic rather than per-origin re-estimation overrides
:meth:`Forecaster.update`, which refreshes the conditioning data without
re-estimating parameters.  That is what makes ``refit_frequency`` both fast and
honest: parameters may be stale by design, but the data the forecast is
conditioned on is always current as of ``t``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_QUANTILE_LEVELS",
    "Forecast",
    "Forecaster",
    "EmpiricalQuantileForecaster",
    "NotEnoughData",
    "quantile_column",
    "parse_quantile_column",
]

#: Quantile grid matching Chronos-2's default output, so baseline prediction
#: intervals and Chronos-2's native quantiles are directly comparable in the
#: calibration table without any regridding.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (
    0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
)

#: Forecasts are floored here.  A volatility forecast of zero or below is
#: meaningless and makes QLIKE undefined (it takes a log of the ratio), but a
#: levels-space HAR genuinely can predict a negative value in a calm regime.
#: Flooring keeps the backtest running and the event is counted in
#: :attr:`Forecaster.n_floored` so it shows up as a finding rather than a
#: silent repair.
MIN_FORECAST = 1e-8


class NotEnoughData(ValueError):
    """Raised by ``fit`` when the history is too short to estimate the model.

    The backtest harness catches this, records a ``NaN`` forecast for that
    origin and carries on, rather than aborting the whole run.
    """


def quantile_column(level: float) -> str:
    """Return the results-frame column name for a quantile ``level``.

    ``0.05 -> "q0.05"``.  Kept as a function so the results schema has exactly
    one definition shared by the forecasters, the harness and the metrics.
    """
    return f"q{level:g}"


def parse_quantile_column(column: str) -> float | None:
    """Inverse of :func:`quantile_column`; ``None`` if ``column`` is not one."""
    if not column.startswith("q"):
        return None
    try:
        return float(column[1:])
    except ValueError:
        return None


@dataclass(frozen=True)
class Forecast:
    """One model's prediction for one (origin, horizon) pair.

    Attributes:
        point: Point forecast in the units of the input series (volatility).
        model: Name of the model that produced it.
        origin: The forecast origin ``t`` -- the last date the model was
            allowed to see.
        horizon: Steps ahead; the forecast targets the series value at ``t+h``.
        quantiles: Optional ``{level: value}`` predictive quantiles.  Baselines
            fill this from empirical residuals; Chronos-2 will fill it natively.
    """

    point: float
    model: str
    origin: pd.Timestamp
    horizon: int
    quantiles: Mapping[float, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

    def as_row(self) -> dict[str, float | str | pd.Timestamp | int]:
        """Flatten to a results-frame row fragment (no ticker/realized yet)."""
        row: dict[str, float | str | pd.Timestamp | int] = {
            "model": self.model,
            "origin_date": self.origin,
            "horizon": self.horizon,
            "forecast": self.point,
        }
        row.update({quantile_column(level): value for level, value in self.quantiles.items()})
        return row


class Forecaster(ABC):
    """Abstract base every forecasting model implements.

    Subclasses must implement :meth:`fit` and :meth:`forecast`.  Optionally
    override :meth:`update` when re-estimating parameters is expensive but
    refreshing the conditioning data is cheap.

    Attributes:
        name: Short identifier used in the results frame and leaderboards.
        quantile_levels: Levels this model reports quantiles for.
        insanity_filter: Whether to replace implausible point forecasts with
            the in-sample mean (see :meth:`_sanitize`).
        insanity_multiple: A forecast above this multiple of the in-sample
            maximum counts as implausible.
        n_floored: How many forecasts have been clipped at :data:`MIN_FORECAST`
            over this object's lifetime.
        n_insane: How many forecasts the insanity filter replaced.  This is a
            headline diagnostic for the levels-space HAR family, not a footnote.
    """

    def __init__(
        self,
        name: str,
        *,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        insanity_filter: bool = False,
        insanity_multiple: float = 10.0,
    ) -> None:
        self.name = name
        self.quantile_levels = tuple(float(level) for level in quantile_levels)
        if any(not 0.0 < level < 1.0 for level in self.quantile_levels):
            raise ValueError("quantile levels must lie strictly between 0 and 1")
        self.insanity_filter = insanity_filter
        self.insanity_multiple = float(insanity_multiple)
        self.n_floored = 0
        self.n_insane = 0
        self._history: pd.Series | None = None
        self._exog: pd.DataFrame | None = None
        self._sane_fallback: float | None = None
        self._sane_upper: float | None = None

    # -- interface -------------------------------------------------------- #

    @abstractmethod
    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Estimate the model on ``history``.

        Args:
            history: Realized-volatility series indexed by date, running up to
                and including the forecast origin.  It never contains a value
                dated after the origin -- that is the harness's guarantee and
                the thing the lookahead test enforces.
            exog: Optional companion measures aligned to the same index
                (bipower variation, quarticity, semivariances, covariates...).
                Most models ignore it; the HAR-J / HAR-RS / HARQ
                specifications require it, and a foundation model with
                covariate support will consume it here.  It is subject to the
                same no-lookahead rule as ``history``.

        Raises:
            NotEnoughData: If the history is too short to estimate the model.
        """

    @abstractmethod
    def forecast(self, h: int) -> Forecast:
        """Forecast the series value ``h`` steps after the origin.

        Args:
            h: Horizon in steps (trading days), ``>= 1``.

        Returns:
            A :class:`Forecast` whose ``origin`` is the last date of the
            history most recently passed to :meth:`fit` or :meth:`update`.
        """

    def update(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Refresh the conditioning data without necessarily re-estimating.

        The default re-fits, which is always correct.  Models whose parameter
        estimation is expensive (HAR's per-horizon OLS, and later Chronos-2's
        forward pass) override this to reuse cached parameters while still
        conditioning on data current as of the new origin.  This is what
        ``refit_frequency`` in the backtest config exploits.

        Args:
            history: Series up to and including the new origin.
            exog: Companion measures for the same span, if the model uses them.
        """
        self.fit(history, exog)

    # -- helpers for subclasses ------------------------------------------- #

    @property
    def origin(self) -> pd.Timestamp:
        """The last date of the fitted history (the forecast origin ``t``)."""
        if self._history is None or self._history.empty:
            raise RuntimeError(f"{self.name}: forecast() called before fit()")
        return self._history.index[-1]

    def _store_history(
        self,
        history: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Validate, clean and remember ``history`` (and any ``exog``).

        Also enforces the no-lookahead rule *inside the model*: exogenous data
        dated after the last history observation is rejected outright rather
        than quietly ignored.  The harness already guarantees this, but a model
        is the last place a leak can be caught, so it checks too.
        """
        if not isinstance(history, pd.Series):
            raise TypeError("history must be a pandas Series")
        if not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("history must be indexed by a DatetimeIndex")
        if not history.index.is_monotonic_increasing:
            raise ValueError("history must be sorted ascending")

        clean = history.dropna().astype("float64")
        if clean.empty:
            raise NotEnoughData(f"{self.name}: history is empty after dropping NaNs")
        self._history = clean
        self._set_sanity_bounds(clean.to_numpy())

        if exog is None:
            self._exog = None
        else:
            if not isinstance(exog, pd.DataFrame):
                raise TypeError("exog must be a pandas DataFrame")
            if not isinstance(exog.index, pd.DatetimeIndex):
                raise TypeError("exog must be indexed by a DatetimeIndex")
            origin = clean.index[-1]
            if len(exog) and exog.index.max() > origin:
                raise ValueError(
                    f"{self.name}: exog extends to {exog.index.max():%Y-%m-%d}, "
                    f"past the origin {origin:%Y-%m-%d} -- that would be lookahead"
                )
            self._exog = exog.reindex(clean.index)

        return clean

    def _exog_column(self, name: str) -> np.ndarray:
        """Return one exogenous column aligned to the stored history.

        Raises:
            NotEnoughData: If no exog was supplied or the column is absent --
                phrased as a data problem so the harness records a ``NaN``
                forecast and continues rather than aborting the run.
        """
        if self._exog is None:
            raise NotEnoughData(f"{self.name}: requires exogenous measures but none were supplied")
        if name not in self._exog.columns:
            raise NotEnoughData(
                f"{self.name}: requires the {name!r} column; "
                f"exog has {list(self._exog.columns)}"
            )
        return self._exog[name].to_numpy(dtype="float64")

    def _set_sanity_bounds(self, values: np.ndarray) -> None:
        """Record the in-sample mean and maximum the insanity filter uses."""
        finite = values[np.isfinite(values)]
        if finite.size:
            self._sane_fallback = float(finite.mean())
            self._sane_upper = float(finite.max()) * self.insanity_multiple

    def _sanitize(self, value: float) -> float:
        """Bollerslev-Patton-Quaedvlieg (2016) "insanity filter".

        A levels-space HAR can produce a wildly implausible point forecast --
        negative, or orders of magnitude above anything ever observed -- when
        its regressors take an extreme value.  HARQ is the notorious case: its
        quarticity interaction occasionally drives the prediction through zero.

        The standard remedy in this literature is not to clip such a forecast to
        a tiny epsilon but to **replace it with the in-sample unconditional
        mean**, on the grounds that a model announcing something impossible has
        told you nothing and the sample average is the honest fallback.

        This matters enormously under QLIKE, which diverges as the forecast
        approaches zero.  In a 7,200-forecast run, a single forecast floored at
        ``1e-8`` produced a mean QLIKE of 1.0e10 -- one observation in ten
        thousand dominating the entire metric.  Without this filter our HARQ
        numbers are not comparable to any published HARQ result.

        Returns:
            The original value, or the in-sample mean if it was implausible.
        """
        if not self.insanity_filter or self._sane_fallback is None:
            return value
        implausible = (
            not np.isfinite(value)
            or value <= 0.0
            or (self._sane_upper is not None and value > self._sane_upper)
        )
        if implausible:
            self.n_insane += 1
            return self._sane_fallback
        return value

    def _floor(self, value: float) -> float:
        """Clip a *point* forecast to stay strictly positive, counting the event."""
        if not np.isfinite(value) or value < MIN_FORECAST:
            self.n_floored += 1
            return MIN_FORECAST
        return float(value)

    @staticmethod
    def _clip(value: float) -> float:
        """Clip a quantile to stay strictly positive, without counting it.

        A low quantile of a levels-space model routinely lands below zero; that
        is a property of the symmetric residual distribution, not a defect worth
        flagging the way a floored point forecast is.
        """
        if not np.isfinite(value) or value < MIN_FORECAST:
            return MIN_FORECAST
        return float(value)

    def reset(self) -> None:
        """Forget all fitted state.  Called by the harness between tickers."""
        self._history = None
        self._exog = None
        self._sane_fallback = None
        self._sane_upper = None
        self.n_floored = 0
        self.n_insane = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


class EmpiricalQuantileForecaster(Forecaster):
    """A :class:`Forecaster` whose quantiles come from its in-sample residuals.

    This is the VOLARE paper's approach to giving a point-forecast model a
    predictive distribution: take the empirical distribution of the model's own
    ``h``-step in-sample residuals and centre it on the point forecast.  It
    costs nothing extra to estimate and gives the baselines calibrated-ish
    bands that sit in the same calibration table as Chronos-2's native
    quantiles.

    Subclasses implement three hooks:

    * :meth:`_residual_space_point` -- the prediction in whatever space the
      residuals live in (levels for HAR, logs for Log-HAR).
    * :meth:`_in_sample_residuals` -- the ``h``-step residuals in that space.
    * :meth:`_inverse_transform` -- map that space back to volatility units
      (identity by default; ``exp`` for Log-HAR).

    The point forecast defaults to the inverse-transformed residual-space
    prediction, but :meth:`_point_forecast` can be overridden when the two
    differ -- which they do for Log-HAR, where the point forecast carries a
    Jensen bias correction that the quantiles must *not* have.
    """

    @abstractmethod
    def _residual_space_point(self, h: int) -> float:
        """Prediction for horizon ``h`` in the space the residuals live in."""

    @abstractmethod
    def _in_sample_residuals(self, h: int) -> np.ndarray:
        """In-sample ``h``-step residuals, in the residual space."""

    def _inverse_transform(self, value: float | np.ndarray) -> float | np.ndarray:
        """Map a residual-space value back to volatility units."""
        return value

    def _point_forecast(self, h: int) -> float:
        """The reported point forecast, in volatility units."""
        return float(self._inverse_transform(self._residual_space_point(h)))

    def forecast(self, h: int) -> Forecast:
        """Point forecast plus empirical-residual quantiles for horizon ``h``."""
        if h < 1:
            raise ValueError(f"horizon must be >= 1, got {h}")

        # Resolve the origin first: it raises a clear "forecast() before fit()"
        # error rather than letting subclasses fail on an empty internal array.
        origin = self.origin
        point = self._floor(self._sanitize(self._point_forecast(h)))
        quantiles: dict[float, float] = {}

        residuals = np.asarray(self._in_sample_residuals(h), dtype="float64")
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size >= 2:
            centre = self._residual_space_point(h)
            # Quantiles are built in residual space and then inverse-transformed.
            # For a monotone transform (exp) that is exact: the alpha-quantile of
            # log(RV) maps to the alpha-quantile of RV.  Deliberately no bias
            # correction here -- that correction targets the *mean*, and applying
            # it to quantiles would shift the whole predictive distribution.
            offsets = np.quantile(residuals, self.quantile_levels)
            for level, offset in zip(self.quantile_levels, offsets):
                quantiles[level] = self._clip(float(self._inverse_transform(centre + offset)))

        return Forecast(
            point=point,
            model=self.name,
            origin=origin,
            horizon=h,
            quantiles=quantiles,
        )
