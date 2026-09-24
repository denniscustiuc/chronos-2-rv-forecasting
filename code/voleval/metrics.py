"""Loss functions, aggregation and calibration for volatility forecasts.

**QLIKE is the primary metric.**  Realized volatility is not observed -- it is
*estimated*, with noise -- so a loss function evaluated against a noisy proxy
can rank models differently from how it would rank them against the truth.
Patton (2011) shows only a narrow class of losses is robust to that, and QLIKE
and MSE are the two commonly used members.  QLIKE is preferred for volatility
because it is scale-free and penalises under-prediction far more harshly than
over-prediction, which matches how a volatility forecast is actually used: a
risk system that under-states vol is dangerous, one that over-states it is
merely expensive.

Everything here operates on the tidy results frame produced by
:mod:`voleval.backtest`, so metrics never need to know which model or which
data source produced a row.
"""

from __future__ import annotations

import logging
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from volmodels.base import parse_quantile_column

logger = logging.getLogger(__name__)

__all__ = [
    "qlike",
    "squared_error",
    "absolute_error",
    "LOSS_FUNCTIONS",
    "PRIMARY_LOSS",
    "add_losses",
    "summarize",
    "coverage_table",
    "leaderboard",
    "mincer_zarnowitz",
    "subsample_summaries",
    "COVID_BOUNDARY",
]

#: The loss every headline table is ranked by.
PRIMARY_LOSS = "qlike"


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype="float64")


def qlike(
    realized,
    forecast,
    *,
    inputs_are_variance: bool = False,
) -> np.ndarray:
    """QLIKE loss, computed on **variances**.

    ``L = (r / f) - ln(r / f) - 1`` where ``r`` and ``f`` are the realized and
    forecast *variances*.  Inputs default to volatilities and are squared
    internally, which is the convention throughout this project.

    The loss is zero when ``f == r`` and strictly positive otherwise (it is a
    Bregman divergence), so it is minimised by a perfect forecast.  It is also
    asymmetric: halving the forecast hurts roughly twice as much as doubling it.

    Args:
        realized: Realized values (volatility unless ``inputs_are_variance``).
        forecast: Forecast values, same units as ``realized``.
        inputs_are_variance: Set ``True`` if the inputs are already variances.

    Returns:
        Per-observation losses; ``NaN`` where either input is non-positive or
        non-finite (a log of the ratio is undefined there).
    """
    r, f = _as_array(realized), _as_array(forecast)

    # Validate BEFORE squaring.  A negative volatility is meaningless, but
    # squaring it would turn it into a perfectly valid-looking variance and
    # yield a finite loss for nonsense input.
    valid = np.isfinite(r) & np.isfinite(f) & (r > 0.0) & (f > 0.0)
    if inputs_are_variance:
        variance_r, variance_f = r, f
    else:
        variance_r, variance_f = r**2, f**2

    out = np.full(r.shape, np.nan, dtype="float64")
    ratio = np.divide(variance_r, variance_f, out=np.ones_like(variance_r), where=valid)
    out[valid] = ratio[valid] - np.log(ratio[valid]) - 1.0
    return out


def squared_error(realized, forecast) -> np.ndarray:
    """Squared error on the volatility scale."""
    return (_as_array(realized) - _as_array(forecast)) ** 2


def absolute_error(realized, forecast) -> np.ndarray:
    """Absolute error on the volatility scale."""
    return np.abs(_as_array(realized) - _as_array(forecast))


#: Per-observation loss functions, keyed by the column name they produce.
LOSS_FUNCTIONS: Mapping[str, Callable[..., np.ndarray]] = {
    "qlike": qlike,
    "se": squared_error,
    "ae": absolute_error,
}


def add_losses(results: pd.DataFrame) -> pd.DataFrame:
    """Attach per-observation loss columns to a results frame.

    Args:
        results: Tidy backtest results with ``forecast`` and ``realized``.

    Returns:
        A copy with ``qlike``, ``se`` and ``ae`` columns added.  Rows where the
        forecast failed (``NaN``) propagate as ``NaN`` losses and are excluded
        from every aggregate downstream.
    """
    for column in ("forecast", "realized"):
        if column not in results.columns:
            raise KeyError(f"results frame is missing the {column!r} column")

    out = results.copy()
    for name, function in LOSS_FUNCTIONS.items():
        out[name] = function(out["realized"], out["forecast"])
    return out


def _aggregate(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Mean loss and observation count over ``keys``."""
    loss_columns = [c for c in LOSS_FUNCTIONS if c in frame.columns]
    grouped = frame.groupby(list(keys), observed=True)
    summary = grouped[loss_columns].mean()
    summary["rmse"] = np.sqrt(grouped["se"].mean())
    summary["n_obs"] = grouped["forecast"].count()
    summary["n_failed"] = grouped["forecast"].apply(lambda s: int(s.isna().sum()))
    return summary.reset_index()


def _add_loss_ratios(summary: pd.DataFrame, benchmark: str, keys: Sequence[str]) -> pd.DataFrame:
    """Add ``<loss>_ratio`` columns relative to ``benchmark`` within ``keys``.

    A ratio below 1 means the model beats the benchmark on that loss.
    """
    out = summary.copy()
    loss_columns = [c for c in (*LOSS_FUNCTIONS, "rmse") if c in out.columns]

    if benchmark not in set(out["model"]):
        logger.warning("benchmark %r not among the evaluated models; skipping loss ratios", benchmark)
        for column in loss_columns:
            out[f"{column}_ratio"] = np.nan
        return out

    benchmark_rows = out[out["model"] == benchmark]
    # Map each row's key onto the benchmark's loss for that key.  Index.map
    # preserves row order, where dividing two differently-indexed Series would
    # silently realign and scramble the result.
    row_keys = pd.MultiIndex.from_frame(out[list(keys)]) if len(keys) > 1 else pd.Index(out[keys[0]])
    for column in loss_columns:
        reference = benchmark_rows.set_index(list(keys))[column]
        if reference.index.has_duplicates:
            raise ValueError(f"benchmark {benchmark!r} appears more than once per {list(keys)}")
        denominator = np.asarray(row_keys.map(reference), dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{column}_ratio"] = out[column].to_numpy() / denominator
    return out


def summarize(
    results: pd.DataFrame,
    *,
    benchmark: str = "log_har",
) -> dict[str, pd.DataFrame]:
    """Aggregate backtest results into the headline metric tables.

    Produces both aggregations the evaluation plan calls for, because they
    answer different questions and can disagree:

    ``pooled``
        Every (origin, ticker) observation weighted equally.  A ticker with more
        origins, or one in a turbulent regime with larger losses, pulls the
        average around.
    ``per_asset``
        Mean loss within each ticker, kept separate.
    ``equal_weighted``
        The per-asset means averaged across tickers, so each asset counts once
        regardless of sample length. This is the fairer headline number when
        assets have unequal histories.

    Args:
        results: Tidy backtest results, with or without loss columns.
        benchmark: Model whose loss forms the denominator of the ratio columns.

    Returns:
        ``{"pooled": ..., "per_asset": ..., "equal_weighted": ...}``, each a
        frame keyed by model and horizon with mean losses, ``n_obs`` and
        ``<loss>_ratio`` columns.
    """
    frame = results if "qlike" in results.columns else add_losses(results)

    pooled = _add_loss_ratios(_aggregate(frame, ["model", "horizon"]), benchmark, ["horizon"])
    per_asset = _add_loss_ratios(
        _aggregate(frame, ["model", "ticker", "horizon"]), benchmark, ["ticker", "horizon"]
    )

    loss_columns = [c for c in (*LOSS_FUNCTIONS, "rmse") if c in per_asset.columns]
    equal_weighted = (
        per_asset.groupby(["model", "horizon"], observed=True)[loss_columns]
        .mean()
        .reset_index()
    )
    equal_weighted["n_assets"] = (
        per_asset.groupby(["model", "horizon"], observed=True)["ticker"].nunique().to_numpy()
    )
    equal_weighted = _add_loss_ratios(equal_weighted, benchmark, ["horizon"])

    return {"pooled": pooled, "per_asset": per_asset, "equal_weighted": equal_weighted}


def coverage_table(results: pd.DataFrame) -> pd.DataFrame:
    """Empirical vs nominal coverage of each quantile level.

    For quantile level ``alpha`` the empirical coverage is the fraction of
    observations with ``realized <= q_alpha``.  A well-calibrated model puts
    that at ``alpha``: the 0.9 quantile should sit above the realized value 90%
    of the time.

    Args:
        results: Tidy backtest results including ``q<level>`` columns.

    Returns:
        Long frame with ``model``, ``horizon``, ``level``, ``nominal``,
        ``empirical``, ``gap`` (empirical - nominal) and ``n_obs``.  Empty if
        the results carry no quantile columns.
    """
    levels = {
        column: level
        for column in results.columns
        if (level := parse_quantile_column(column)) is not None
    }
    if not levels:
        return pd.DataFrame(columns=["model", "horizon", "level", "nominal", "empirical", "gap", "n_obs"])

    rows = []
    for (model, horizon), group in results.groupby(["model", "horizon"], observed=True):
        for column, level in sorted(levels.items(), key=lambda item: item[1]):
            pair = group[["realized", column]].dropna()
            if pair.empty:
                continue
            empirical = float((pair["realized"] <= pair[column]).mean())
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "level": level,
                    "nominal": level,
                    "empirical": empirical,
                    "gap": empirical - level,
                    "n_obs": int(len(pair)),
                }
            )
    return pd.DataFrame(rows)


def leaderboard(
    summary: pd.DataFrame,
    horizon: int,
    *,
    loss: str = PRIMARY_LOSS,
) -> pd.DataFrame:
    """Rank models at one horizon by ``loss``, best first.

    Args:
        summary: One of the frames returned by :func:`summarize`.
        horizon: Horizon to slice.
        loss: Loss column to rank by.

    Returns:
        Frame indexed by rank with the model, its loss, the ratio versus the
        benchmark and the observation count.
    """
    if loss not in summary.columns:
        raise KeyError(f"{loss!r} not in summary columns: {list(summary.columns)}")

    slice_ = summary[summary["horizon"] == horizon].sort_values(loss).reset_index(drop=True)
    columns = ["model", loss, f"{loss}_ratio", "mae" if loss != "ae" else "ae"]
    columns = [c for c in dict.fromkeys(columns) if c in slice_.columns]
    for extra in ("ae", "rmse", "n_obs", "n_assets"):
        if extra in slice_.columns and extra not in columns:
            columns.append(extra)

    ranked = slice_[columns].copy()
    ranked.index = pd.RangeIndex(1, len(ranked) + 1, name="rank")
    return ranked


# --------------------------------------------------------------------------- #
# Mincer-Zarnowitz forecast-efficiency regressions
# --------------------------------------------------------------------------- #

#: Default boundary separating the pre- and post-COVID subsamples.  The WHO
#: pandemic declaration and the US equity drawdown both land in this window, and
#: the volatility regime shift around it is the sharpest in the modern sample.
COVID_BOUNDARY = "2020-03-01"


def mincer_zarnowitz(
    results: pd.DataFrame,
    *,
    scale: str = "volatility",
    per_asset: bool = True,
) -> pd.DataFrame:
    """Mincer-Zarnowitz forecast-efficiency regressions.

    Regresses the realisation on the forecast::

        realized_t = alpha + beta * forecast_t + e_t

    A forecast that is *efficient* -- using its information optimally and
    needing no recalibration -- has ``alpha = 0`` and ``beta = 1``.  ``beta < 1``
    means the forecast over-reacts and should be shrunk toward its mean;
    ``beta > 1`` means it under-reacts.  ``R^2`` measures how much of the
    variation in realized volatility the forecast explains at all.

    This is the diagnostic that separates two very different ways of winning on
    QLIKE: a model can have genuinely more *information* (higher ``R^2``) or it
    can simply be better *calibrated* (``beta`` nearer 1) while carrying the same
    information. Brini uses exactly this split to argue that the foundation
    models' edge is largely the latter.

    Args:
        results: Tidy backtest results with ``forecast`` and ``realized``.
        scale: ``"volatility"`` (as stored), ``"variance"`` (square both) or
            ``"log"`` (log both).  Brini reports on the volatility scale.
        per_asset: Fit one regression per (model, ticker, horizon) and average
            the coefficients across assets -- matching Brini's "average over
            assets" tables.  ``False`` pools every observation into one
            regression per (model, horizon).

    Returns:
        Frame with ``model``, ``horizon``, ``alpha``, ``beta``, ``r2``,
        ``n_obs`` and, when ``per_asset``, ``n_assets``.
    """
    if scale not in ("volatility", "variance", "log"):
        raise ValueError(f"scale must be volatility, variance or log; got {scale!r}")

    frame = results[["model", "ticker", "horizon", "forecast", "realized"]].dropna()
    if frame.empty:
        return pd.DataFrame(columns=["model", "horizon", "alpha", "beta", "r2", "n_obs"])

    if scale == "variance":
        frame = frame.assign(forecast=frame["forecast"] ** 2, realized=frame["realized"] ** 2)
    elif scale == "log":
        positive = (frame["forecast"] > 0) & (frame["realized"] > 0)
        frame = frame[positive]
        frame = frame.assign(
            forecast=np.log(frame["forecast"]), realized=np.log(frame["realized"])
        )

    def _regress(group: pd.DataFrame) -> dict[str, float]:
        x = group["forecast"].to_numpy(dtype="float64")
        y = group["realized"].to_numpy(dtype="float64")
        if x.size < 3 or np.allclose(x, x[0]):
            return {"alpha": np.nan, "beta": np.nan, "r2": np.nan, "n_obs": float(x.size)}
        design = np.column_stack([np.ones(x.size), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        total = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(residuals @ residuals) / total if total > 0 else np.nan
        return {
            "alpha": float(coefficients[0]),
            "beta": float(coefficients[1]),
            "r2": r2,
            "n_obs": float(x.size),
        }

    keys = ["model", "ticker", "horizon"] if per_asset else ["model", "horizon"]
    rows = []
    for key, group in frame.groupby(keys, observed=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update(_regress(group))
        rows.append(row)

    table = pd.DataFrame(rows)
    if not per_asset:
        return table.sort_values(["horizon", "model"]).reset_index(drop=True)

    averaged = (
        table.groupby(["model", "horizon"], observed=True)[["alpha", "beta", "r2"]]
        .mean()
        .reset_index()
    )
    averaged["n_assets"] = (
        table.groupby(["model", "horizon"], observed=True)["ticker"].nunique().to_numpy()
    )
    averaged["n_obs"] = (
        table.groupby(["model", "horizon"], observed=True)["n_obs"].sum().to_numpy()
    )
    return averaged.sort_values(["horizon", "model"]).reset_index(drop=True)


def subsample_summaries(
    results: pd.DataFrame,
    *,
    boundary: str = COVID_BOUNDARY,
    benchmark: str = "log_har",
    date_column: str = "target_date",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Split the results at ``boundary`` and summarise each period separately.

    Volatility regimes are not stationary, and a model that wins on average can
    owe that entirely to one turbulent stretch.  Splitting at the COVID shock is
    the standard robustness check in this literature, and it is the cheapest way
    to see whether a ranking is stable or an artefact of one regime.

    Args:
        results: Tidy backtest results.
        boundary: ISO date; rows strictly before it are ``"pre"``.
        benchmark: Model for the loss-ratio denominators.
        date_column: Which date to split on -- ``target_date`` (when the
            realisation happened) rather than ``origin_date``, so a forecast is
            attributed to the regime it was actually scored in.

    Returns:
        ``{"pre": tables, "post": tables, "full": tables}`` where each value is
        the dict :func:`summarize` returns.  Periods with no rows are omitted.
    """
    frame = results if "qlike" in results.columns else add_losses(results)
    cut = pd.Timestamp(boundary)

    periods = {
        "pre": frame[frame[date_column] < cut],
        "post": frame[frame[date_column] >= cut],
        "full": frame,
    }
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for label, subset in periods.items():
        if subset.empty:
            logger.info("subsample %r is empty at boundary %s; skipping", label, boundary)
            continue
        out[label] = summarize(subset, benchmark=benchmark)
    return out
