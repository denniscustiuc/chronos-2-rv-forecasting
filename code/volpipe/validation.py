"""Diagnostics for the estimator output: correlations, overlay plots, VOLARE check.

Nothing here is part of the production path -- these are the sanity checks you
run after a pipeline run to confirm the estimators behave the way theory says
they should (highly correlated with each other, close-to-close the noisiest).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "estimator_correlation_matrix",
    "estimator_noise_ranking",
    "plot_estimators",
    "compare_with_volare",
]


def estimator_correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix of the ``rv_*`` columns.

    Args:
        frame: A per-ticker output frame.
        method: Any method accepted by :meth:`pandas.DataFrame.corr`.

    Returns:
        Square correlation matrix over the ``rv_*`` columns that have data.
    """
    rv_cols = [c for c in frame.columns if c.startswith("rv_") and frame[c].notna().any()]
    if not rv_cols:
        return pd.DataFrame()
    return frame[rv_cols].corr(method=method)


def estimator_noise_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank estimators by how noisy their volatility series is.

    Noise proxy: the standard deviation of the *daily change* in the estimator's
    realized-volatility series, scaled by its mean level.  Close-to-close is
    expected to come out on top -- it uses one price per day and throws away the
    whole intraday range.

    Returns:
        Frame indexed by column with ``mean_vol``, ``std_vol`` and
        ``noise_ratio``, sorted noisiest first.
    """
    rv_cols = [c for c in frame.columns if c.startswith("rv_") and frame[c].notna().any()]
    rows = []
    for col in rv_cols:
        series = frame[col].dropna()
        if series.empty:
            continue
        mean_vol = float(series.mean())
        rows.append(
            {
                "column": col,
                "mean_vol": mean_vol,
                "std_vol": float(series.std()),
                "noise_ratio": float(series.diff().std() / mean_vol) if mean_vol else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("column").sort_values("noise_ratio", ascending=False)


def plot_estimators(
    frame: pd.DataFrame,
    ticker: str,
    output_path: str | Path | None = None,
    *,
    show: bool = False,
) -> Path | None:
    """Overlay every ``rv_*`` series on one axis.

    Args:
        frame: A per-ticker output frame.
        ticker: Used in the plot title.
        output_path: Where to save the PNG; skipped when ``None`` and
            ``show`` is ``False``.
        show: Call ``plt.show()`` instead of/as well as saving.

    Returns:
        The path written, or ``None`` if nothing was plotted.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot")
        return None

    rv_cols = [c for c in frame.columns if c.startswith("rv_") and frame[c].notna().any()]
    if not rv_cols:
        logger.warning("%s: no rv_* columns with data; skipping plot", ticker)
        return None

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in rv_cols:
        series = frame[col].dropna()
        # Sparse series (intraday_rv) read better as points than as a line.
        style = {"linewidth": 1.1} if len(series) > 60 else {"linewidth": 0, "marker": ".", "markersize": 3}
        ax.plot(series.index, series.to_numpy(), label=col, **style)

    ax.set_title(f"{ticker} - annualised realized volatility by estimator")
    ax.set_ylabel("annualised volatility")
    ax.set_xlabel("date")
    ax.grid(alpha=0.25)
    ax.legend(ncol=min(3, len(rv_cols)), fontsize=9)
    fig.tight_layout()

    written: Path | None = None
    if output_path is not None:
        written = Path(output_path)
        written.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(written, dpi=130)
        logger.info("%s: wrote plot to %s", ticker, written)
    if show:
        plt.show()
    plt.close(fig)
    return written


def compare_with_volare(
    frame: pd.DataFrame,
    volare_rv5: pd.Series | None = None,
    *,
    column: str = "rv_gk",
    annualization_factor: int = 252,
    volare_is_variance: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, float | str]:
    """OPTIONAL check of our estimates against VOLARE's 5-minute ``rv5``.

    VOLARE is not wired up yet, so ``volare_rv5=None`` returns a clearly-marked
    stub result.  Once the VOLARE loader exists, pass its ``rv5`` series (daily
    variance by default) for one overlapping ticker and date range.

    Two caveats, both expected and neither a bug:

    1. VOLARE builds its measures from UNADJUSTED prices; ours come from
       split/dividend-adjusted OHLC, so the series diverge around corporate
       actions.
    2. ``rv5`` is a 5-minute intraday measure of the *open-to-close* session,
       while ``rv_gk``/``rv_parkinson`` are daily-OHLC approximations of the
       same session and ``rv_cc``/``rv_yz`` also carry the overnight gap.
       Correlation should be high; levels need not match one-for-one.

    Args:
        frame: A per-ticker output frame.
        volare_rv5: VOLARE's ``rv5`` series indexed by date, or ``None``.
        column: Which of our ``rv_*`` columns to compare against.
        annualization_factor: Used to annualise ``volare_rv5``.
        volare_is_variance: ``True`` if ``volare_rv5`` is a daily variance (the
            usual convention); ``False`` if already a volatility.
        output_path: Optional PNG path for the scatter plot.

    Returns:
        A dict with ``status`` plus, when data was supplied, the overlap size,
        Pearson/Spearman correlations and the mean ratio of the two levels.
    """
    if volare_rv5 is None:
        logger.info("compare_with_volare: no VOLARE series supplied - returning stub")
        return {
            "status": "stub",
            "detail": (
                "VOLARE not wired up. Pass volare_rv5 (daily rv5 variance indexed by date) "
                "for an overlapping ticker/date range to run the comparison."
            ),
        }

    if column not in frame.columns:
        raise KeyError(f"{column!r} not in frame; available: {[c for c in frame.columns if c.startswith('rv_')]}")

    ours = frame[column].dropna()
    theirs = volare_rv5.dropna().copy()
    theirs.index = pd.DatetimeIndex(pd.to_datetime(theirs.index)).normalize()
    if theirs.index.tz is not None:
        theirs.index = theirs.index.tz_localize(None)
    if volare_is_variance:
        theirs = np.sqrt(theirs * float(annualization_factor))

    joined = pd.concat([ours.rename("ours"), theirs.rename("volare")], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return {"status": "insufficient_overlap", "overlap_days": float(len(joined))}

    # Spearman computed as Pearson on ranks, so scipy stays an optional dependency.
    ranked = joined.rank()
    stats: dict[str, float | str] = {
        "status": "ok",
        "column": column,
        "overlap_days": float(len(joined)),
        "pearson": float(joined["ours"].corr(joined["volare"])),
        "spearman": float(ranked["ours"].corr(ranked["volare"])),
        "mean_ratio_ours_over_volare": float((joined["ours"] / joined["volare"]).mean()),
    }

    if output_path is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(joined["volare"], joined["ours"], s=10, alpha=0.6)
            lim = [0, float(max(joined.max())) * 1.05]
            ax.plot(lim, lim, linestyle="--", linewidth=1, color="grey", label="45 degrees")
            ax.set_xlabel("VOLARE rv5 (annualised vol)")
            ax.set_ylabel(f"ours: {column}")
            ax.set_title(f"{column} vs VOLARE rv5 (r={stats['pearson']:.3f})")
            ax.legend()
            fig.tight_layout()
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=130)
            plt.close(fig)
            stats["plot"] = str(path)
        except ImportError:
            logger.warning("matplotlib not installed; skipping VOLARE scatter plot")

    return stats
