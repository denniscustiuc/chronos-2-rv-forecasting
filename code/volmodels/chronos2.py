"""Chronos-2 as a drop-in :class:`~volmodels.base.Forecaster`.

Chronos-2 (Amazon Science, ``amazon/chronos-2``) is a ~120M-parameter
encoder-only time-series foundation model with group attention, native
quantile output, and native support for past-only and known-future covariates.
It is **zero-shot**: there is no training step, so ``fit`` does nothing but set
the context the next forecast is conditioned on.

Nothing in the harness, the metrics, the significance tests or the classical
models changes to accommodate it.  That is the point: Chronos-2 is scored on
exactly the footing our Brini replication validated -- same VOLARE ``rv5``
series, same rolling 1,000-day window, same point-in-time target
``sigma_{t+h}``, same QLIKE.

Three registered modes:

``chronos2_uni``
    Target series only.  The minimal comparison against Log-HAR.
``chronos2_cov``
    Adds covariates through the ``exog`` channel: VOLARE's own realized
    measures and VIX as **past-only**, and days-until-next-earnings as a
    **known-future** covariate.
``chronos2_multi``
    Several tickers forecast jointly in one call.  Exploratory -- the Chronos-2
    report is explicit that cross-learning does not always help, so this is a
    test rather than an assumed win.

**Native quantiles, not residual bands.**  The classical baselines synthesise
prediction intervals from their own in-sample residuals; Chronos-2 emits
quantiles directly.  Using its real quantiles is the whole point of the
calibration comparison, so this class deliberately does *not* inherit from
:class:`~volmodels.base.EmpiricalQuantileForecaster`.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .base import DEFAULT_QUANTILE_LEVELS, Forecast, Forecaster, NotEnoughData

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CHECKPOINT",
    "Chronos2Univariate",
    "Chronos2Covariate",
    "Chronos2Multivariate",
    "load_pipeline",
    "clear_pipeline_cache",
    "resolve_device",
    "earnings_countdown",
    "PAST_COVARIATE_COLUMNS",
]

DEFAULT_CHECKPOINT = "amazon/chronos-2"

#: Past-only covariate columns we pass when present.  These are VOLARE's own
#: realized measures plus VIX; all are observable at the origin and never
#: supplied beyond it.
PAST_COVARIATE_COLUMNS: tuple[str, ...] = ("var_bv", "var_rsp", "var_rsn", "var_rq", "jump", "vix")

#: The one known-future covariate: days until the next scheduled earnings date.
KNOWN_FUTURE_COLUMN = "days_to_earnings"

#: Loaded pipelines, keyed by (checkpoint, device).  The checkpoint is loaded
#: once per process and reused across every origin, ticker and mode.
_PIPELINE_CACHE: dict[tuple[str, str], object] = {}


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to ``"cuda"`` when a GPU is present, else ``"cpu"``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency of chronos
        return "cpu"


def load_pipeline(checkpoint: str = DEFAULT_CHECKPOINT, device: str = "auto"):
    """Load (and cache) a Chronos-2 pipeline.

    The checkpoint is loaded lazily on first use and kept for the lifetime of
    the process.  A backtest makes tens of thousands of forecast calls, so
    reloading per call -- or even per ticker -- would dominate the runtime.

    Args:
        checkpoint: Hugging Face model id.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or any ``device_map`` value.

    Returns:
        A ``Chronos2Pipeline``.

    Raises:
        ImportError: If ``chronos-forecasting>=2.0`` is not installed.
    """
    resolved = resolve_device(device)
    key = (checkpoint, resolved)
    if key not in _PIPELINE_CACHE:
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "Chronos-2 needs the official package: pip install 'chronos-forecasting>=2.0'"
            ) from exc

        logger.info("loading %s on %s", checkpoint, resolved)
        _PIPELINE_CACHE[key] = Chronos2Pipeline.from_pretrained(checkpoint, device_map=resolved)
    return _PIPELINE_CACHE[key]


def clear_pipeline_cache() -> None:
    """Drop cached pipelines (used by tests to free GPU memory)."""
    _PIPELINE_CACHE.clear()


def earnings_countdown(
    index: pd.DatetimeIndex,
    release_dates: pd.DatetimeIndex,
    *,
    cap: int = 63,
) -> pd.Series:
    """Trading days until the next scheduled earnings release.

    This replaces the binary ``earnings_flag`` the pipeline originally used.
    That flag carried essentially no signal at h=22 (correlation −0.03 to +0.02
    with future log-RV) because it was on for only ~5% of days and marked a
    *short-horizon* event.  A countdown is continuous, is informative on every
    day rather than a handful, and lets the model learn the whole run-up and
    decay around a release.

    It is a **known-future** covariate: the earnings calendar is published weeks
    ahead, so its values over ``t+1..t+h`` are legitimately known at ``t``.
    This is the same deliberate exception the pipeline already documents, and
    the only input in this module permitted to extend past the origin.

    Args:
        index: Trading dates to compute the countdown over.
        release_dates: Scheduled and historical release dates.
        cap: Values are clipped here, so a long gap between releases does not
            dominate the covariate's scale.

    Returns:
        Float series named :data:`KNOWN_FUTURE_COLUMN`, counting trading-day
        positions (not calendar days) to the next release at or after each date.
    """
    out = np.full(len(index), float(cap))
    if len(release_dates):
        releases = pd.DatetimeIndex(sorted(set(release_dates)))
        # Position in `index` of the first trading day on or after each release.
        positions = np.unique(index.searchsorted(releases, side="left"))
        positions = positions[positions < len(index)]
        if positions.size:
            # For each date, the distance to the next release position ahead.
            ahead = np.searchsorted(positions, np.arange(len(index)), side="left")
            valid = ahead < positions.size
            out[valid] = positions[ahead[valid]] - np.arange(len(index))[valid]
    return pd.Series(
        np.clip(out, 0.0, cap), index=index, name=KNOWN_FUTURE_COLUMN, dtype="float64"
    )


class _Chronos2Base(Forecaster):
    """Shared lifecycle for the Chronos-2 modes.

    Attributes:
        checkpoint: Hugging Face model id.
        device: Resolved compute device.
        context_length: Most recent observations handed to the model.
        max_horizon: Prediction length requested per call.  One call per origin
            covers every horizon, and :meth:`forecast` reads step ``h`` out of
            the cached path -- the model call is the expensive part and does not
            depend on ``h``.
    """

    def __init__(
        self,
        name: str,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "auto",
        context_length: int = 1000,
        max_horizon: int = 22,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        seed: int = 0,
    ) -> None:
        super().__init__(name, quantile_levels=quantile_levels)
        self.checkpoint = checkpoint
        self.device = resolve_device(device)
        self.context_length = int(context_length)
        self.max_horizon = int(max_horizon)
        self.seed = int(seed)
        self._path: np.ndarray | None = None   # (prediction_length, n_levels)
        self._mean_path: np.ndarray | None = None

    # -- lifecycle --------------------------------------------------------- #

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Set the context.  Chronos-2 is zero-shot -- nothing is estimated.

        Args:
            history: RV/volatility series up to and including the origin.
            exog: Past-only covariates, already truncated at the origin by the
                harness.  Ignored in univariate mode.
        """
        self._store_history(history, exog)
        self._path = None  # the context moved, so the cached forecast is stale

    def update(self, history: pd.Series, exog: pd.DataFrame | None = None) -> None:
        """Identical to :meth:`fit` -- there are no parameters to re-estimate."""
        self.fit(history, exog)

    # -- inputs ------------------------------------------------------------ #

    @property
    def series_name(self) -> str | None:
        """Identifier of the series being forecast, if the history carries one.

        The harness names each series after its ticker, which lets a single
        factory serve every asset while still selecting per-ticker inputs (an
        earnings schedule, a companion set) -- without the harness needing to
        pass the ticker explicitly.
        """
        if self._history is None:
            return None
        return None if self._history.name is None else str(self._history.name)

    def _context(self) -> np.ndarray:
        """The most recent ``context_length`` observations."""
        values = self._history.to_numpy(dtype="float64")
        return values[-self.context_length :]

    def _build_inputs(self, prediction_length: int) -> list[dict]:
        """Build the ``predict_quantiles`` input list.  Overridden per mode."""
        return [{"target": self._context()}]

    # -- forecasting ------------------------------------------------------- #

    def _compute_path(self, prediction_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Call the model once and return ``(quantiles, mean)`` for the path."""
        import torch

        pipeline = load_pipeline(self.checkpoint, self.device)
        torch.manual_seed(self.seed)  # deterministic given the same context

        with torch.inference_mode():
            quantiles, mean = pipeline.predict_quantiles(
                self._build_inputs(prediction_length),
                prediction_length=prediction_length,
                quantile_levels=list(self.quantile_levels),
            )
        # Each element is (n_variates, prediction_length, n_levels); this class
        # forecasts one target series, so take variate 0 of element 0.
        return (
            quantiles[0][0].float().cpu().numpy(),
            mean[0][0].float().cpu().numpy(),
        )

    def forecast(self, h: int) -> Forecast:
        """Forecast ``sigma_{t+h}`` with Chronos-2's native quantiles.

        The point forecast is the **median** (``q0.5``), not the mean: the
        predictive distribution of volatility is right-skewed, and the median is
        the quantile the model reports most directly.

        Args:
            h: Horizon in trading days.

        Returns:
            A :class:`~volmodels.base.Forecast` carrying the full native
            quantile dict.
        """
        if h < 1:
            raise ValueError(f"horizon must be >= 1, got {h}")
        origin = self.origin

        needed = max(h, self.max_horizon)
        if self._path is None or self._path.shape[0] < h:
            self._path, self._mean_path = self._compute_path(needed)

        row = self._path[h - 1]
        quantiles = {
            level: self._clip(float(value))
            for level, value in zip(self.quantile_levels, row)
        }

        median_index = (
            self.quantile_levels.index(0.5) if 0.5 in self.quantile_levels else len(row) // 2
        )
        point = self._floor(float(row[median_index]))

        return Forecast(
            point=point, model=self.name, origin=origin, horizon=h, quantiles=quantiles
        )

    def reset(self) -> None:
        """Clear per-series state; the loaded checkpoint is deliberately kept."""
        super().reset()
        self._path = None
        self._mean_path = None


class Chronos2Univariate(_Chronos2Base):
    """Chronos-2 on the target series alone.

    The minimal drop-in: the same RV series Log-HAR sees, nothing else.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "chronos2_uni")
        super().__init__(kwargs.pop("name"), **kwargs)


class Chronos2Covariate(_Chronos2Base):
    """Chronos-2 with past-only and known-future covariates.

    Past-only covariates arrive through the harness's ``exog`` channel and are
    truncated at the origin exactly as the target is.  The known-future
    countdown is supplied at construction, because the harness deliberately
    truncates ``exog`` at ``t`` and a genuinely-known-future schedule needs
    values over ``t+1..t+h``.  That schedule is the *only* input here allowed
    past the origin, mirroring the ``earnings_flag`` exception the pipeline
    already documents.

    Chronos-2 requires every key in ``future_covariates`` to also appear in
    ``past_covariates``, so the countdown is supplied on both sides.

    Attributes:
        past_columns: Columns taken from ``exog`` when present.
        known_future: The countdown schedule.  Either one series (used for
            every asset) or a ``{ticker: series}`` mapping selected by
            :attr:`series_name`, so one factory serves the whole panel.
            ``None`` runs with past-only covariates.
    """

    def __init__(
        self,
        *,
        past_columns: Sequence[str] = PAST_COVARIATE_COLUMNS,
        known_future: pd.Series | Mapping[str, pd.Series] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("name", "chronos2_cov")
        super().__init__(kwargs.pop("name"), **kwargs)
        self.past_columns = tuple(past_columns)
        self.known_future = known_future

    def _schedule(self) -> pd.Series | None:
        """The known-future countdown for the series currently being forecast."""
        if self.known_future is None:
            return None
        if isinstance(self.known_future, pd.Series):
            return self.known_future
        return self.known_future.get(self.series_name)

    def _build_inputs(self, prediction_length: int) -> list[dict]:
        context = self._context()
        length = context.shape[0]
        history_index = self._history.index[-length:]

        past: dict[str, np.ndarray] = {}
        if self._exog is not None:
            for column in self.past_columns:
                if column not in self._exog.columns:
                    continue
                values = self._exog[column].to_numpy(dtype="float64")[-length:]
                if not np.isfinite(values).any():
                    continue
                # Chronos-2 handles NaN, but a constant-fill on a fully-missing
                # tail is cheaper than letting the scaler see an all-NaN series.
                past[column] = np.nan_to_num(values, nan=float(np.nanmedian(values)))

        entry: dict[str, object] = {"target": context}

        schedule = self._schedule()
        if schedule is not None:
            past_countdown = schedule.reindex(history_index).to_numpy(dtype="float64")
            # The forward slice is what makes this a *known-future* covariate.
            after = schedule.loc[schedule.index > history_index[-1]]
            future_countdown = after.to_numpy(dtype="float64")[:prediction_length]
            if (
                np.isfinite(past_countdown).all()
                and future_countdown.shape[0] == prediction_length
            ):
                past[KNOWN_FUTURE_COLUMN] = past_countdown
                entry["future_covariates"] = {KNOWN_FUTURE_COLUMN: future_countdown}

        if past:
            entry["past_covariates"] = past
        elif "future_covariates" in entry:
            # A future covariate with no past counterpart is rejected by the API.
            entry.pop("future_covariates")

        return [entry]


class Chronos2Multivariate(_Chronos2Base):
    """Chronos-2 forecasting several series jointly -- **exploratory**.

    The target is stacked with a set of companion series into one multivariate
    item, so group attention can share information across them.  The Chronos-2
    report is explicit that cross-learning "doesn't always improve forecast
    accuracy and must be tested for individual use cases", so this mode is a
    test, not an assumed win.  Expect modest effects at best.

    Attributes:
        companions: ``{name: series}`` stacked alongside the target.  Each is
            truncated at the origin like the target.
    """

    def __init__(self, *, companions: Mapping[str, pd.Series] | None = None, **kwargs) -> None:
        kwargs.setdefault("name", "chronos2_multi")
        super().__init__(kwargs.pop("name"), **kwargs)
        self.companions = dict(companions or {})

    def _build_inputs(self, prediction_length: int) -> list[dict]:
        context = self._context()
        length = context.shape[0]
        index = self._history.index[-length:]

        rows = [context]
        own = self.series_name
        for name, series in self.companions.items():
            if own is not None and name == own:
                continue  # never stack a series with a copy of itself
            aligned = series.reindex(index).to_numpy(dtype="float64")
            if np.isfinite(aligned).sum() < length // 2:
                continue  # too sparse over this window to be worth a variate
            rows.append(np.nan_to_num(aligned, nan=float(np.nanmedian(aligned))))

        if len(rows) == 1:
            return [{"target": context}]
        return [{"target": np.vstack(rows)}]

    def _compute_path(self, prediction_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Same as the base call, but keep variate 0 -- our ticker -- from the stack."""
        return super()._compute_path(prediction_length)
