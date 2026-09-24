"""Tests of whether a difference in forecast accuracy is real or noise.

Two tools, both operating on per-observation loss series so they work for any
model registered in the harness:

* :func:`diebold_mariano` -- pairwise, "is A better than B?"
* :func:`model_confidence_set` -- joint, "which models can we not rule out as
  best?"  The MCS is the more useful of the two once several models are in play,
  because running many pairwise tests inflates the false-positive rate.  It is
  built now so that adding Chronos-2 and Kronos needs no new evaluation code.

A caveat that matters for this project: both tests assume the loss
differentials are a single, reasonably well-behaved time series.  Pooling three
mega-cap tech tickers violates that -- their log-RV series correlate 0.71-0.85,
so the effective sample is materially smaller than the row count suggests.
Prefer :func:`per_ticker_dm` for headline claims and read the pooled test as
indicative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "DMResult",
    "MCSResult",
    "loss_matrix",
    "diebold_mariano",
    "dm_against_benchmark",
    "per_ticker_dm",
    "model_confidence_set",
    "mcs_inclusion_rates",
]

Kernel = Literal["bartlett", "rectangular"]


@dataclass(frozen=True)
class DMResult:
    """Outcome of one Diebold-Mariano test.

    Attributes:
        model_a, model_b: The models compared; the differential is
            ``loss(a) - loss(b)``.
        mean_difference: Mean loss differential.  Negative favours ``model_a``.
        statistic: HLN-corrected DM statistic.
        p_value: Two-sided p-value against a ``t(n-1)`` distribution.
        n_obs: Paired observations used.
        lags: HAC truncation lag (``horizon - 1``).
        better: Name of the lower-loss model, or ``"tie"`` if the difference is
            not significant at 5%.
    """

    model_a: str
    model_b: str
    mean_difference: float
    statistic: float
    p_value: float
    n_obs: int
    lags: int
    better: str

    def as_row(self) -> dict[str, object]:
        """Flatten to a table row."""
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "mean_diff": self.mean_difference,
            "dm_stat": self.statistic,
            "p_value": self.p_value,
            "n_obs": self.n_obs,
            "better": self.better,
        }


@dataclass
class MCSResult:
    """Outcome of a Model Confidence Set procedure.

    Attributes:
        table: One row per model with ``mcs_pvalue``, ``included`` and
            ``eliminated_at`` (elimination order, ``NaN`` for survivors).
        alpha: Confidence level used.
        included: Models in the surviving set.
        n_bootstrap: Bootstrap replications.
        block_size: Block length used by the circular block bootstrap.
    """

    table: pd.DataFrame
    alpha: float
    included: list[str] = field(default_factory=list)
    n_bootstrap: int = 0
    block_size: int = 1

    def __str__(self) -> str:  # pragma: no cover - presentation only
        kept = ", ".join(self.included) if self.included else "(none)"
        return f"MCS(alpha={self.alpha}) survivors: {kept}"


def loss_matrix(
    results: pd.DataFrame,
    horizon: int,
    *,
    loss: str = "qlike",
    ticker: str | None = None,
) -> pd.DataFrame:
    """Pivot backtest results into a ``(observation x model)`` loss matrix.

    Only observations where *every* model produced a forecast survive, so the
    comparison is like-for-like.

    Args:
        results: Tidy results frame carrying the ``loss`` column (run
            :func:`voleval.metrics.add_losses` first if absent).
        horizon: Horizon to slice.
        loss: Loss column to extract.
        ticker: Restrict to one ticker; ``None`` pools all of them.

    Returns:
        Frame indexed by ``(ticker, origin_date)`` with one column per model,
        sorted by origin so the HAC lags mean what they should.
    """
    if loss not in results.columns:
        raise KeyError(f"{loss!r} not in results; call voleval.metrics.add_losses first")

    frame = results[results["horizon"] == horizon]
    if ticker is not None:
        frame = frame[frame["ticker"] == ticker]
    if frame.empty:
        return pd.DataFrame()

    matrix = frame.pivot_table(
        index=["ticker", "origin_date"], columns="model", values=loss, observed=True
    )
    return matrix.dropna(how="any").sort_index(level=["ticker", "origin_date"])


def _hac_variance(d: np.ndarray, lags: int, kernel: Kernel = "bartlett") -> float:
    """Long-run variance of the mean of ``d``, robust to serial correlation.

    Direct ``h``-step forecast errors overlap, making the loss differential an
    MA(h-1) process, so the naive ``var / n`` understates uncertainty and
    inflates the test statistic.  Truncating at ``h - 1`` lags is the standard
    Diebold-Mariano choice.  Bartlett weights are the default because they
    guarantee a non-negative estimate; the rectangular kernel of the original
    paper can go negative in small samples.
    """
    n = d.size
    centred = d - d.mean()
    gamma0 = float(centred @ centred / n)
    total = gamma0

    for k in range(1, max(0, lags) + 1):
        if k >= n:
            break
        gamma_k = float(centred[k:] @ centred[:-k] / n)
        weight = 1.0 - k / (lags + 1.0) if kernel == "bartlett" else 1.0
        total += 2.0 * weight * gamma_k

    if total <= 0.0:
        logger.warning("HAC variance was non-positive; falling back to the lag-0 estimate")
        total = gamma0
    return total / n


def diebold_mariano(
    loss_a,
    loss_b,
    *,
    horizon: int = 1,
    model_a: str = "a",
    model_b: str = "b",
    kernel: Kernel = "bartlett",
    alpha: float = 0.05,
) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests ``H0: E[loss(a) - loss(b)] = 0`` against a two-sided alternative,
    using a HAC variance truncated at ``horizon - 1`` lags and the
    Harvey-Leybourne-Newbold small-sample correction (which rescales the
    statistic and compares it against ``t(n-1)`` rather than a normal --
    without it the test over-rejects badly at long horizons).

    Args:
        loss_a, loss_b: Paired per-observation losses, same length and ordering.
        horizon: Forecast horizon, setting the HAC truncation.
        model_a, model_b: Names for the result object.
        kernel: HAC kernel.
        alpha: Significance level for the ``better`` verdict.

    Returns:
        A :class:`DMResult`.  A **negative** statistic favours ``model_a``.

    Raises:
        ValueError: If fewer than three paired finite observations remain.
    """
    a, b = np.asarray(loss_a, dtype="float64"), np.asarray(loss_b, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"loss arrays must have the same shape, got {a.shape} and {b.shape}")

    d = a - b
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        raise ValueError(f"need at least 3 paired observations, got {n}")

    lags = max(0, horizon - 1)
    variance = _hac_variance(d, lags, kernel)
    mean_difference = float(d.mean())

    if variance <= 0.0 or not np.isfinite(variance):
        statistic, p_value = 0.0, 1.0
    else:
        statistic = mean_difference / np.sqrt(variance)
        # Harvey-Leybourne-Newbold small-sample correction.
        h = horizon
        adjustment = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
        statistic *= np.sqrt(max(adjustment, 1e-12))
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1)))

    if p_value < alpha:
        better = model_a if mean_difference < 0 else model_b
    else:
        better = "tie"

    return DMResult(
        model_a=model_a,
        model_b=model_b,
        mean_difference=mean_difference,
        statistic=float(statistic),
        p_value=p_value,
        n_obs=n,
        lags=lags,
        better=better,
    )


def dm_against_benchmark(
    results: pd.DataFrame,
    horizon: int,
    *,
    benchmark: str = "log_har",
    loss: str = "qlike",
    ticker: str | None = None,
) -> pd.DataFrame:
    """Run Diebold-Mariano for every model against ``benchmark`` at one horizon.

    Returns:
        Table of :meth:`DMResult.as_row` rows, sorted by mean differential so
        the models that beat the benchmark appear first.
    """
    matrix = loss_matrix(results, horizon, loss=loss, ticker=ticker)
    if matrix.empty or benchmark not in matrix.columns:
        logger.warning("no comparable observations for benchmark %r at h=%d", benchmark, horizon)
        return pd.DataFrame()

    rows = []
    for model in matrix.columns:
        if model == benchmark:
            continue
        outcome = diebold_mariano(
            matrix[model].to_numpy(),
            matrix[benchmark].to_numpy(),
            horizon=horizon,
            model_a=model,
            model_b=benchmark,
        )
        rows.append(outcome.as_row())

    table = pd.DataFrame(rows)
    return table.sort_values("mean_diff").reset_index(drop=True) if not table.empty else table


def per_ticker_dm(
    results: pd.DataFrame,
    horizon: int,
    *,
    benchmark: str = "log_har",
    loss: str = "qlike",
) -> pd.DataFrame:
    """Diebold-Mariano per ticker, avoiding the pooled-independence problem.

    Pooling correlated assets into one loss series overstates the effective
    sample size.  Running the test asset by asset and reading the pattern across
    assets is the more defensible claim.
    """
    frames = []
    for ticker in sorted(results["ticker"].unique()):
        table = dm_against_benchmark(
            results, horizon, benchmark=benchmark, loss=loss, ticker=ticker
        )
        if not table.empty:
            table.insert(0, "ticker", ticker)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _circular_block_indices(
    n: int, block_size: int, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular block bootstrap indices, shape ``(n_bootstrap, n)``.

    Resampling contiguous blocks rather than individual observations preserves
    the serial dependence in the loss differentials, which overlapping
    multi-step forecasts guarantee is present.
    """
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return indices.reshape(n_bootstrap, -1)[:, :n]


def model_confidence_set(
    results: pd.DataFrame,
    horizon: int,
    *,
    loss: str = "qlike",
    alpha: float = 0.10,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    ticker: str | None = None,
    seed: int = 0,
) -> MCSResult:
    """Hansen, Lunde & Nason (2011) Model Confidence Set, range-statistic form.

    Iteratively tests the null that every surviving model has equal predictive
    accuracy.  If rejected, the worst model is eliminated and the test repeats.
    What survives is the set of models that cannot be distinguished from the
    best at confidence ``1 - alpha`` -- the honest answer when several models
    are close, and far better behaved than running every pairwise test.

    The procedure uses the range statistic
    ``T_R = max_ij |dbar_ij| / sqrt(var(dbar_ij))`` with the variance and the
    null distribution both estimated by circular block bootstrap.

    Args:
        results: Tidy results frame carrying the ``loss`` column.
        horizon: Horizon to slice.
        loss: Loss column to compare on.
        alpha: Significance level; survivors form the ``(1-alpha)`` MCS.
        n_bootstrap: Bootstrap replications.
        block_size: Block length; defaults to ``max(horizon, 2)`` so blocks span
            the MA(h-1) dependence induced by overlapping forecasts.
        ticker: Restrict to one ticker; ``None`` pools all of them.
        seed: RNG seed, so the procedure is reproducible.

    Returns:
        An :class:`MCSResult`.  With fewer than two models, or too few
        observations, it returns every model as included with a p-value of 1.
    """
    matrix = loss_matrix(results, horizon, loss=loss, ticker=ticker)
    models = list(matrix.columns)

    def _trivial(reason: str) -> MCSResult:
        logger.warning("MCS at h=%d: %s", horizon, reason)
        table = pd.DataFrame(
            {"model": models, "mcs_pvalue": 1.0, "included": True, "eliminated_at": np.nan}
        )
        return MCSResult(table=table, alpha=alpha, included=list(models), n_bootstrap=0)

    if len(models) < 2:
        return _trivial("fewer than two models to compare")
    if len(matrix) < 10:
        return _trivial(f"only {len(matrix)} comparable observations")

    block = block_size or max(horizon, 2)
    rng = np.random.default_rng(seed)
    losses = matrix.to_numpy()
    n = losses.shape[0]

    indices = _circular_block_indices(n, block, n_bootstrap, rng)
    # (n_bootstrap, n_models) bootstrap replicates of each model's mean loss.
    boot_means = losses[indices].mean(axis=1)

    alive = list(range(len(models)))
    eliminated: list[tuple[int, float]] = []
    running_pvalue = 0.0

    while len(alive) > 1:
        sub_mean = losses[:, alive].mean(axis=0)
        sub_boot = boot_means[:, alive]

        # Pairwise mean differentials and their bootstrap standard errors.
        dbar = sub_mean[:, None] - sub_mean[None, :]
        boot_dbar = sub_boot[:, :, None] - sub_boot[:, None, :]
        variance = ((boot_dbar - dbar[None, :, :]) ** 2).mean(axis=0)
        scale = np.sqrt(np.where(variance > 0.0, variance, np.nan))

        t_stats = np.divide(dbar, scale, out=np.zeros_like(dbar), where=np.isfinite(scale))
        boot_t = np.divide(
            np.abs(boot_dbar - dbar[None, :, :]),
            scale[None, :, :],
            out=np.zeros_like(boot_dbar),
            where=np.isfinite(scale)[None, :, :],
        )

        observed = float(np.nanmax(np.abs(t_stats)))
        null_distribution = np.nanmax(boot_t, axis=(1, 2))
        p_value = float(np.mean(null_distribution >= observed))

        running_pvalue = max(running_pvalue, p_value)
        if p_value >= alpha:
            break

        # Eliminate the model whose loss most exceeds the others'.
        worst_local = int(np.nanargmax(np.nanmax(t_stats, axis=1)))
        eliminated.append((alive[worst_local], running_pvalue))
        alive.pop(worst_local)

    survivors = {models[i] for i in alive}
    eliminated_pvalues = {models[i]: p for i, p in eliminated}
    eliminated_order = {models[i]: rank for rank, (i, _) in enumerate(eliminated, start=1)}

    table = pd.DataFrame(
        {
            "model": models,
            "mcs_pvalue": [
                eliminated_pvalues.get(m, max(running_pvalue, alpha)) for m in models
            ],
            "included": [m in survivors for m in models],
            "eliminated_at": [eliminated_order.get(m, np.nan) for m in models],
            "mean_loss": matrix.mean().to_numpy(),
        }
    ).sort_values("mean_loss").reset_index(drop=True)

    return MCSResult(
        table=table,
        alpha=alpha,
        included=[m for m in table["model"] if m in survivors],
        n_bootstrap=n_bootstrap,
        block_size=block,
    )


def mcs_inclusion_rates(
    results: pd.DataFrame,
    horizon: int,
    *,
    loss: str = "qlike",
    alpha: float = 0.10,
    n_bootstrap: int = 500,
    block_size: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the MCS **per asset** and report each model's inclusion rate.

    This is Brini's Table 8 protocol, and it is the right one for a
    cross-section: a single MCS over a pooled loss series silently assumes the
    assets are independent draws, which correlated equities are not.  Running
    the procedure asset by asset and counting how often a model survives gives a
    statement that does not lean on that assumption -- "this model is
    indistinguishable from the best on 90% of assets" -- and it degrades
    gracefully when one asset behaves oddly.

    Args:
        results: Tidy results frame carrying the ``loss`` column.
        horizon: Horizon to slice.
        loss: Loss column to compare on.
        alpha: MCS significance level.
        n_bootstrap: Bootstrap replications per asset.
        block_size: Block length; defaults to ``max(horizon, 2)``.
        seed: Base RNG seed; each asset uses a distinct derived seed.

    Returns:
        Frame with ``model``, ``n_assets``, ``n_included`` and
        ``inclusion_pct``, sorted by inclusion rate descending.
    """
    tickers = sorted(results["ticker"].unique())
    counts: dict[str, int] = {}
    assessed = 0

    for offset, ticker in enumerate(tickers):
        outcome = model_confidence_set(
            results, horizon, loss=loss, alpha=alpha, n_bootstrap=n_bootstrap,
            block_size=block_size, ticker=ticker, seed=seed + offset,
        )
        if outcome.table.empty:
            continue
        assessed += 1
        for model in outcome.table["model"]:
            counts.setdefault(model, 0)
        for model in outcome.included:
            counts[model] += 1

    if not assessed:
        return pd.DataFrame(columns=["model", "n_assets", "n_included", "inclusion_pct"])

    table = pd.DataFrame(
        {
            "model": list(counts),
            "n_assets": assessed,
            "n_included": [counts[m] for m in counts],
        }
    )
    table["inclusion_pct"] = 100.0 * table["n_included"] / table["n_assets"]
    return table.sort_values("inclusion_pct", ascending=False).reset_index(drop=True)
