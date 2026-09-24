"""Daily realized-variance estimators and their rolling realized-volatility series.

Conventions used throughout this module:

* ``ohlc`` is a ``DataFrame`` indexed by trading date with (at least) the
  columns ``open``, ``high``, ``low``, ``close``, all *adjusted consistently*
  (see :mod:`volpipe.ingest`).
* Every ``*_variance`` function returns a **per-day variance** in squared daily
  log-return units -- not annualised, not a standard deviation.
* Every rolling series at date ``t`` is computed from the window *ending at and
  including* ``t``.  Nothing after ``t`` is ever touched, so the output carries
  no lookahead bias.

Note on comparability with VOLARE: VOLARE's realized measures are built from
UNADJUSTED intraday prices, whereas everything here is derived from
split/dividend-adjusted OHLC.  The two therefore will not agree exactly,
especially across split and ex-dividend dates.  That divergence is expected.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ESTIMATOR_COLUMN_SUFFIX

logger = logging.getLogger(__name__)

__all__ = [
    "EstimatorResult",
    "close_to_close_variance",
    "parkinson_variance",
    "garman_klass_variance",
    "rogers_satchell_variance",
    "yang_zhang_variance",
    "intraday_rv_variance",
    "rolling_variance",
    "annualize_variance",
    "to_realized_vol",
    "compute_estimator",
    "WINDOW_DEPENDENT_ESTIMATORS",
]

_REQUIRED_COLUMNS = ("open", "high", "low", "close")

#: Estimators whose rolling variance is *not* simply the mean of the per-day
#: variances (they are defined directly on the window).
WINDOW_DEPENDENT_ESTIMATORS = frozenset({"close_to_close", "yang_zhang"})


@dataclass(frozen=True)
class EstimatorResult:
    """The two series produced for each estimator.

    Attributes:
        name: Estimator name, e.g. ``"garman_klass"``.
        daily_variance: Per-day variance estimate (daily units).
        windowed_variance: Rolling window variance ending at each date (daily
            units), ``NaN`` until a full window is available.
        realized_vol: ``sqrt(annualization_factor * windowed_variance)``.
        available: ``False`` when the inputs were missing entirely (e.g. no
            intraday history), in which case the series are all-``NaN``.
    """

    name: str
    daily_variance: pd.Series
    windowed_variance: pd.Series
    realized_vol: pd.Series
    available: bool = True

    @property
    def suffix(self) -> str:
        """Short column suffix for this estimator (``"gk"``, ``"cc"``, ...)."""
        return ESTIMATOR_COLUMN_SUFFIX[self.name]


def _validate_ohlc(ohlc: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in ohlc.columns]
    if missing:
        raise KeyError(f"ohlc frame is missing required column(s): {missing}")
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise TypeError("ohlc must be indexed by a DatetimeIndex of trading dates")
    if not ohlc.index.is_monotonic_increasing:
        raise ValueError("ohlc index must be sorted ascending (no lookahead otherwise)")


def _logs(ohlc: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return ``(log O, log H, log L, log C)`` with non-positive prices masked."""
    _validate_ohlc(ohlc)
    frame = ohlc[list(_REQUIRED_COLUMNS)].astype("float64")
    frame = frame.where(frame > 0.0)  # log of a non-positive price is undefined
    logged = np.log(frame)
    return logged["open"], logged["high"], logged["low"], logged["close"]


# --------------------------------------------------------------------------- #
# Per-day variance estimators
# --------------------------------------------------------------------------- #

def close_to_close_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Squared close-to-close log return, ``r_t^2`` with ``r_t = ln(C_t/C_{t-1})``.

    This is the zero-mean per-day variance proxy.  The *windowed* close-to-close
    variance is the sample variance of ``r`` (see :func:`rolling_variance`),
    which demeans and is what the spec calls for.  Closes only, so this is the
    noisiest of the estimators here.

    Args:
        ohlc: Adjusted OHLC frame indexed by trading date.

    Returns:
        Per-day variance series; ``NaN`` on the first row.
    """
    _, _, _, log_c = _logs(ohlc)
    returns = log_c.diff()
    return (returns**2).rename("close_to_close")


def close_to_close_returns(ohlc: pd.DataFrame) -> pd.Series:
    """Close-to-close log returns ``ln(C_t / C_{t-1})``."""
    _, _, _, log_c = _logs(ohlc)
    return log_c.diff().rename("r_cc")


def parkinson_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Parkinson (1980) range estimator: ``(1/(4 ln 2)) * ln(H/L)^2``.

    Roughly five times more efficient than close-to-close, but it ignores the
    overnight gap and assumes zero drift.
    """
    _, log_h, log_l, _ = _logs(ohlc)
    return (1.0 / (4.0 * math.log(2.0)) * (log_h - log_l) ** 2).rename("parkinson")


def garman_klass_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Garman-Klass (1980): ``0.5*ln(H/L)^2 - (2 ln2 - 1)*ln(C/O)^2``.

    Uses the full OHLC bar and is more efficient than Parkinson.  Still assumes
    zero drift, and like Parkinson it measures the open-to-close session only.
    The result is non-negative because ``|ln(C/O)| <= ln(H/L)`` and
    ``0.5 - (2 ln2 - 1) > 0``.
    """
    log_o, log_h, log_l, log_c = _logs(ohlc)
    hl = log_h - log_l
    co = log_c - log_o
    return (0.5 * hl**2 - (2.0 * math.log(2.0) - 1.0) * co**2).rename("garman_klass")


def rogers_satchell_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Rogers-Satchell (1991): ``ln(H/C)ln(H/O) + ln(L/C)ln(L/O)``.

    Drift-independent -- it stays unbiased when the price has a non-zero trend,
    which is where Parkinson and Garman-Klass break down.  Both products are
    non-negative by construction, so the estimate is too.
    """
    log_o, log_h, log_l, log_c = _logs(ohlc)
    return (
        (log_h - log_c) * (log_h - log_o) + (log_l - log_c) * (log_l - log_o)
    ).rename("rogers_satchell")


def _yang_zhang_k(window: int) -> float:
    """Yang-Zhang weighting constant ``k`` minimising estimator variance."""
    if window < 2:
        raise ValueError("yang_zhang requires window >= 2")
    return 0.34 / (1.34 + (window + 1.0) / (window - 1.0))


def yang_zhang_variance(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Yang-Zhang (2000) per-day variance *proxy*.

    The true Yang-Zhang estimator is defined on a window (it uses the sample
    variance of the overnight and open-to-close returns); this function returns
    the analogous per-day decomposition

    ``o_t^2 + k*c_t^2 + (1-k)*rs_t``

    with ``o_t = ln(O_t/C_{t-1})``, ``c_t = ln(C_t/O_t)`` and ``rs_t`` the
    Rogers-Satchell term.  It is a legitimate per-day variance estimate (the
    zero-mean version of the window formula) and is what lands in ``var_yz``;
    the rolling ``rv_yz`` series uses the proper windowed form from
    :func:`rolling_variance`.

    Args:
        ohlc: Adjusted OHLC frame indexed by trading date.
        window: Window length, needed only for the weight ``k``.
    """
    log_o, _, _, log_c = _logs(ohlc)
    overnight = log_o - log_c.shift(1)
    open_to_close = log_c - log_o
    k = _yang_zhang_k(window)
    rs = rogers_satchell_variance(ohlc)
    return (overnight**2 + k * open_to_close**2 + (1.0 - k) * rs).rename("yang_zhang")


def intraday_rv_variance(
    intraday: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    *,
    price_column: str = "close",
    min_bars_per_day: int = 10,
) -> pd.Series:
    """Intraday realized variance ``RV_t = sum_i r_{t,i}^2`` from intraday bars.

    Returns are taken *within* each session only -- the overnight gap between
    the last bar of day ``t-1`` and the first bar of day ``t`` is excluded, so
    this measures the open-to-close session like Parkinson/Garman-Klass do.

    yfinance only serves roughly 60 days of intraday history, so on a multi-year
    sample this series is ``NaN`` almost everywhere.  That is expected: the
    historical intraday path is VOLARE's job.  If ``intraday`` is ``None`` or
    empty the function degrades to an all-``NaN`` series aligned to ``index``.

    Args:
        intraday: Intraday bars indexed by timestamp, or ``None``.
        index: Daily trading-date index to align the output to.
        price_column: Column of ``intraday`` holding the bar price.
        min_bars_per_day: Sessions with fewer bars are discarded as partial.

    Returns:
        Per-day variance series reindexed onto ``index``.
    """
    empty = pd.Series(np.nan, index=index, name="intraday_rv", dtype="float64")
    if intraday is None or len(intraday) == 0:
        logger.info("intraday_rv: no intraday data supplied; emitting all-NaN series")
        return empty
    if price_column not in intraday.columns:
        raise KeyError(f"intraday frame has no {price_column!r} column")

    prices = intraday[price_column].astype("float64")
    prices = prices.where(prices > 0.0).dropna()
    if prices.empty:
        return empty

    stamps = pd.DatetimeIndex(prices.index)
    if stamps.tz is not None:
        # Bars come back in exchange-local time; drop the tz so the session date
        # matches the tz-naive daily index we align against.
        stamps = stamps.tz_localize(None)
    session = pd.Index(stamps.normalize(), name="date")

    log_p = pd.Series(np.log(prices.to_numpy()), index=session)
    grouped = log_p.groupby(level=0, sort=True)
    # diff() within each session drops the first bar, so the overnight gap never
    # contributes -- exactly the intraday-only sum we want.
    rv = grouped.apply(lambda s: float(np.nansum(np.diff(s.to_numpy()) ** 2)))
    counts = grouped.size()
    rv = rv.where(counts >= min_bars_per_day)

    rv.index = pd.DatetimeIndex(rv.index)
    out = rv.reindex(index)
    out.name = "intraday_rv"
    return out.astype("float64")


# --------------------------------------------------------------------------- #
# Windowing / annualisation
# --------------------------------------------------------------------------- #

def rolling_variance(
    name: str,
    daily_variance: pd.Series,
    window: int,
    *,
    ohlc: pd.DataFrame | None = None,
) -> pd.Series:
    """Aggregate a per-day variance series into a rolling window variance.

    The value at date ``t`` uses only the ``window`` observations ending at
    ``t`` (``min_periods=window``, so partial leading windows stay ``NaN``).

    Most estimators aggregate as the window mean of the daily variances.  Two
    are special-cased:

    * ``close_to_close`` -- the sample variance (``ddof=1``) of the log returns
      over the window, as specified, rather than the mean of ``r^2``.
    * ``yang_zhang`` -- the proper windowed form
      ``V_overnight + k*V_open_to_close + (1-k)*V_rogers_satchell``, where the
      first two terms are window sample variances.

    Args:
        name: Estimator name.
        daily_variance: Per-day variance series.
        window: Window length in trading days.
        ohlc: Required for the window-dependent estimators.

    Returns:
        Rolling variance in daily units, aligned to ``daily_variance.index``.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    if name in WINDOW_DEPENDENT_ESTIMATORS:
        if ohlc is None:
            raise ValueError(f"estimator {name!r} needs the ohlc frame to build its window")
        if name == "close_to_close":
            returns = close_to_close_returns(ohlc)
            out = returns.rolling(window, min_periods=window).var(ddof=1)
        else:  # yang_zhang
            log_o, _, _, log_c = _logs(ohlc)
            overnight = log_o - log_c.shift(1)
            open_to_close = log_c - log_o
            k = _yang_zhang_k(window)
            v_o = overnight.rolling(window, min_periods=window).var(ddof=1)
            v_c = open_to_close.rolling(window, min_periods=window).var(ddof=1)
            v_rs = rogers_satchell_variance(ohlc).rolling(window, min_periods=window).mean()
            out = v_o + k * v_c + (1.0 - k) * v_rs
    else:
        out = daily_variance.rolling(window, min_periods=window).mean()

    return out.rename(f"{name}_windowed_variance")


def annualize_variance(variance: pd.Series, annualization_factor: int) -> pd.Series:
    """Scale a daily variance to annual units: ``variance * factor``."""
    return variance * float(annualization_factor)


def to_realized_vol(windowed_variance: pd.Series, annualization_factor: int) -> pd.Series:
    """Convert a windowed daily variance to annualised volatility.

    ``sqrt(annualization_factor * windowed_variance)``.  Negative variances
    (numerically possible only for Yang-Zhang, and then only from floating-point
    noise) are masked to ``NaN`` rather than silently producing a ``NaN`` sqrt
    warning.
    """
    safe = windowed_variance.where(windowed_variance >= 0.0)
    return np.sqrt(annualize_variance(safe, annualization_factor))


def compute_estimator(
    name: str,
    ohlc: pd.DataFrame,
    *,
    window: int,
    annualization_factor: int,
    intraday: pd.DataFrame | None = None,
) -> EstimatorResult:
    """Compute the per-day variance and rolling realized-volatility for one estimator.

    Args:
        name: One of the keys of
            :data:`volpipe.config.ESTIMATOR_COLUMN_SUFFIX`.
        ohlc: Adjusted OHLC frame indexed by trading date.
        window: Rolling window length in trading days.
        annualization_factor: Trading days per year.
        intraday: Intraday bars, only used by ``intraday_rv``.

    Returns:
        An :class:`EstimatorResult`.

    Raises:
        ValueError: If ``name`` is not a known estimator.
    """
    _validate_ohlc(ohlc)

    if name == "close_to_close":
        daily = close_to_close_variance(ohlc)
    elif name == "parkinson":
        daily = parkinson_variance(ohlc)
    elif name == "garman_klass":
        daily = garman_klass_variance(ohlc)
    elif name == "rogers_satchell":
        daily = rogers_satchell_variance(ohlc)
    elif name == "yang_zhang":
        daily = yang_zhang_variance(ohlc, window)
    elif name == "intraday_rv":
        daily = intraday_rv_variance(intraday, ohlc.index)
    else:
        raise ValueError(f"unknown estimator: {name!r}")

    available = bool(daily.notna().any())
    windowed = rolling_variance(name, daily, window, ohlc=ohlc)
    rv = to_realized_vol(windowed, annualization_factor)

    return EstimatorResult(
        name=name,
        daily_variance=daily.rename(f"var_{ESTIMATOR_COLUMN_SUFFIX[name]}"),
        windowed_variance=windowed,
        realized_vol=rv.rename(f"rv_{ESTIMATOR_COLUMN_SUFFIX[name]}"),
        available=available,
    )
