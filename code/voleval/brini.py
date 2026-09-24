"""Brini (2026) replication target: the paper's protocol and its published numbers.

Reference
---------
Alessio Brini, *Forecasting Realized Volatility with Time Series Foundation
Models: A Comparison with Econometric Benchmarks*, arXiv:2607.05291 (6 July
2026).  Benchmarks nine zero-shot foundation models against eight econometric
specifications on the VOLARE dataset, 50 assets.  Headline result: only Tiny
Time Mixers beats Log-HAR at every horizon, and only narrowly; the econometric
benchmarks stay competitive throughout.

No code or data repository is linked from the arXiv page, so this module encodes
the paper's **published tables** as reference constants and compares our re-run
against them directly.  Every constant below is transcribed from the paper and
carries its table number; nothing here is inferred or recomputed.

What "success" means
--------------------
Reproducing a paper's *absolute* numbers from its description alone is hard --
preprocessing, estimation-window edges and library differences all move the
third decimal.  The defensible claim is a faithful **protocol** replication
whose **relative** results line up: loss ratios against Log-HAR, the ordering of
the econometric benchmarks, and MCS membership.  If the absolute QLIKE levels
also land close, that is strong corroboration; if they do not, the relative
comparison still stands on its own.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "BRINI_EQUITIES",
    "BRINI_FX",
    "BRINI_FUTURES",
    "BRINI_HORIZONS",
    "BRINI_CONTEXT_LENGTH",
    "BRINI_EQUITY_START",
    "BRINI_EQUITY_END",
    "BRINI_QLIKE_EQUITIES",
    "BRINI_RATIO_EQUAL_WEIGHTED",
    "BRINI_MCS_INCLUSION",
    "BRINI_MINCER_ZARNOWITZ",
    "BRINI_ECONOMETRIC_MODELS",
    "MODEL_NAME_MAP",
    "brini_protocol",
    "reference_table",
    "compare_with_brini",
    "format_comparison",
]

#: The 40 US equities (paper, Appendix A), spanning all 11 GICS sectors.
BRINI_EQUITIES: tuple[str, ...] = (
    "AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO",
    "CVX", "DIS", "GE", "GOOGL", "GS", "HD", "HON", "IBM", "JNJ", "JPM",
    "KO", "MCD", "META", "MMM", "MRK", "MSFT", "NFLX", "NKE", "NVDA", "ORCL",
    "PG", "PM", "SHW", "TRV", "TSLA", "UNH", "V", "VZ", "WMT", "XOM",
)

#: The 5 major currency pairs.
BRINI_FX: tuple[str, ...] = ("AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDJPY")

#: The 5 futures contracts (Corn, Crude Oil, E-mini S&P 500, Gold, Natural Gas).
BRINI_FUTURES: tuple[str, ...] = ("C", "CL", "ES", "GC", "NG")

#: Forecast horizons in trading days -- note h=22, not our earlier h=21.
BRINI_HORIZONS: tuple[int, ...] = (1, 5, 22)

#: Estimation/context window: 1,000 daily observations (~4 years).
BRINI_CONTEXT_LENGTH: int = 1000

#: Equity sample bounds.  2,786 trading days per stock, 1,764 forecasts at h=1.
BRINI_EQUITY_START: str = "2015-01-02"
BRINI_EQUITY_END: str = "2026-01-30"

#: The eight econometric specifications the paper benchmarks.
BRINI_ECONOMETRIC_MODELS: tuple[str, ...] = (
    "HAR", "Log-HAR", "HAR-J", "HAR-RS", "HARQ", "ARFIMA", "ARMA", "MEM",
)

#: Our model names -> the paper's labels, so the comparison can join on them.
MODEL_NAME_MAP: dict[str, str] = {
    "har": "HAR",
    "log_har": "Log-HAR",
    "har_j": "HAR-J",
    "har_rs": "HAR-RS",
    "harq": "HARQ",
    "arfima": "ARFIMA",
    "arma": "ARMA",
    "mem": "MEM",
}

#: **Table 4** -- pooled QLIKE loss, 40 equities, by model and horizon.
#: A ``*`` in the paper marks Model Confidence Set membership; that information
#: lives in :data:`BRINI_MCS_INCLUSION` instead.
BRINI_QLIKE_EQUITIES: dict[str, dict[int, float]] = {
    "Log-HAR": {1: 0.198, 5: 0.301, 22: 0.538},
    "HAR": {1: 0.199, 5: 0.304, 22: 0.582},
    "HAR-J": {1: 0.217, 5: 0.313, 22: 0.570},
    "HAR-RS": {1: 0.264, 5: 0.412, 22: 0.639},
    "HARQ": {1: 0.636, 5: 0.793, 22: 0.691},
    "ARFIMA": {1: 0.199, 5: 0.313, 22: 0.583},
    "ARMA": {1: 0.198, 5: 0.312, 22: 0.601},
    "MEM": {1: 0.200, 5: 0.310, 22: 0.606},
    "Chronos-Bolt-S": {1: 0.222, 5: 0.361, 22: 0.710},
    "Chronos-Bolt-B": {1: 0.224, 5: 0.357, 22: 0.709},
    "Moirai-2.0-S": {1: 0.207, 5: 0.336, 22: 0.678},
    "Moirai-MoE-S": {1: 0.215, 5: 0.327, 22: 0.595},
    "Lag-Llama": {1: 0.296, 5: 0.466, 22: 0.532},
    "TimesFM-2.5": {1: 0.217, 5: 0.365, 22: 0.724},
    "Toto": {1: 0.229, 5: 0.339, 22: 0.614},
    "Sundial": {1: 0.198, 5: 0.329, 22: 0.652},
    "TTM": {1: 0.194, 5: 0.294, 22: 0.521},
}

#: **Table 6** -- equal-weighted QLIKE loss ratios versus Log-HAR.
#: This is the paper's preferred view: pooled losses are dominated by a handful
#: of high-volatility assets, so the per-asset equal-weighted ratio is the
#: defensible comparison.
BRINI_RATIO_EQUAL_WEIGHTED: dict[str, dict[int, float]] = {
    "Log-HAR": {1: 1.000, 5: 1.000, 22: 1.000},
    "HAR": {1: 0.998, 5: 1.009, 22: 1.070},
    "HAR-J": {1: 1.157, 5: 1.185, 22: 1.099},
    "HAR-RS": {1: 2.312, 5: 1.774, 22: 1.372},
    "HARQ": {1: 5.132, 5: 3.518, 22: 1.329},
    "ARFIMA": {1: 1.012, 5: 1.049, 22: 1.094},
    "ARMA": {1: 1.000, 5: 1.034, 22: 1.112},
    "MEM": {1: 1.014, 5: 1.030, 22: 1.107},
    "Chronos-Bolt-S": {1: 1.110, 5: 1.188, 22: 1.302},
    "Chronos-Bolt-B": {1: 1.121, 5: 1.181, 22: 1.303},
    "Moirai-2.0-S": {1: 1.038, 5: 1.109, 22: 1.259},
    "Moirai-MoE-S": {1: 1.090, 5: 1.102, 22: 1.140},
    "Lag-Llama": {1: 1.532, 5: 1.582, 22: 1.023},
    "TimesFM-2.5": {1: 1.086, 5: 1.201, 22: 1.331},
    "Toto": {1: 1.214, 5: 1.273, 22: 1.157},
    "Sundial": {1: 0.998, 5: 1.084, 22: 1.182},
    "TTM": {1: 0.982, 5: 0.986, 22: 0.987},
}

#: **Table 8** -- percentage of assets whose Model Confidence Set includes the
#: model.  The paper runs the MCS per asset and reports inclusion rates, rather
#: than one MCS over a pooled loss series.
BRINI_MCS_INCLUSION: dict[str, dict[int, float]] = {
    "Log-HAR": {1: 86.0, 5: 90.0, 22: 90.0},
    "HAR": {1: 86.0, 5: 66.0, 22: 54.0},
    "ARFIMA": {1: 78.0, 5: 56.0, 22: 52.0},
    "ARMA": {1: 90.0, 5: 54.0, 22: 38.0},
    "MEM": {1: 76.0, 5: 58.0, 22: 46.0},
    "Chronos-Bolt-S": {1: 0.0, 5: 0.0, 22: 14.0},
    "Moirai-2.0-S": {1: 30.0, 5: 10.0, 22: 16.0},
    "Lag-Llama": {1: 0.0, 5: 2.0, 22: 84.0},
    "Sundial": {1: 92.0, 5: 26.0, 22: 26.0},
    "TTM": {1: 98.0, 5: 96.0, 22: 94.0},
}

#: **Table 9** -- Mincer-Zarnowitz coefficients averaged over the 50 assets.
#: Only the two models the paper tabulates are recorded here.
BRINI_MINCER_ZARNOWITZ: dict[str, dict[int, dict[str, float]]] = {
    "Log-HAR": {
        1: {"alpha": -0.0004, "beta": 1.043, "r2": 0.519},
        5: {"alpha": 0.0008, "beta": 0.940, "r2": 0.285},
        22: {"alpha": 0.0061, "beta": 0.532, "r2": 0.060},
    },
    "TTM": {
        1: {"alpha": -0.0004, "beta": 1.010, "r2": 0.521},
        5: {"alpha": 0.0003, "beta": 0.969, "r2": 0.295},
        22: {"alpha": 0.0046, "beta": 0.670, "r2": 0.078},
    },
}


def brini_protocol(min_train_size: int = BRINI_CONTEXT_LENGTH, refit_frequency: int = 1):
    """Return the :class:`~voleval.backtest.BacktestConfig` matching the paper.

    The paper's protocol, point by point:

    * **Rolling** origin with a fixed 1,000-day estimation window -- not
      expanding.  The window slides forward one day at a time.
    * **Daily re-estimation** of every econometric model (``refit_frequency=1``).
    * Horizons ``h = 1, 5, 22``.
    * Target: the **point-in-time** value at ``t+h``, ``sigma_{t+h} =
      sqrt(RV_{t+h})`` -- explicitly *not* an average over ``t+1..t+h``, because
      overlapping targets induce serial correlation in the thing being
      predicted.  Our harness already defines the target this way, so no change
      is needed.

    Args:
        min_train_size: Estimation window length; the paper uses 1,000.
        refit_frequency: Re-estimation cadence.  The paper refits daily; raising
            this is the only practical lever if a full 40-asset run with ARMA
            and ARFIMA is too slow, and any such deviation must be reported.

    Returns:
        A ``BacktestConfig`` configured to the paper's protocol.
    """
    from .backtest import BacktestConfig

    return BacktestConfig(
        horizons=BRINI_HORIZONS,
        window="rolling",
        min_train_size=min_train_size,
        rolling_window_size=min_train_size,
        refit_frequency=refit_frequency,
    )


def reference_table(which: str = "ratio") -> pd.DataFrame:
    """Return one of the paper's published tables as a tidy frame.

    Args:
        which: ``"qlike"`` (Table 4), ``"ratio"`` (Table 6), ``"mcs"``
            (Table 8) or ``"mz"`` (Table 9).

    Returns:
        Long frame with ``brini_model``, ``horizon`` and the value column(s).

    Raises:
        ValueError: If ``which`` is not a known table.
    """
    sources = {
        "qlike": (BRINI_QLIKE_EQUITIES, "brini_qlike"),
        "ratio": (BRINI_RATIO_EQUAL_WEIGHTED, "brini_ratio"),
        "mcs": (BRINI_MCS_INCLUSION, "brini_mcs_pct"),
    }
    if which in sources:
        table, column = sources[which]
        rows = [
            {"brini_model": model, "horizon": horizon, column: value}
            for model, per_horizon in table.items()
            for horizon, value in per_horizon.items()
        ]
        return pd.DataFrame(rows)

    if which == "mz":
        rows = [
            {
                "brini_model": model,
                "horizon": horizon,
                "brini_alpha": stats["alpha"],
                "brini_beta": stats["beta"],
                "brini_r2": stats["r2"],
            }
            for model, per_horizon in BRINI_MINCER_ZARNOWITZ.items()
            for horizon, stats in per_horizon.items()
        ]
        return pd.DataFrame(rows)

    raise ValueError(f"unknown reference table {which!r}; expected qlike, ratio, mcs or mz")


def compare_with_brini(
    equal_weighted: pd.DataFrame,
    pooled: pd.DataFrame | None = None,
    *,
    mz: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the side-by-side table of our numbers against the paper's.

    Joins on the mapped model name and horizon, so any model we ran that the
    paper did not report (the naive floors) comes through with blank reference
    columns rather than being dropped.

    Args:
        equal_weighted: Our ``summarize(...)["equal_weighted"]`` frame.
        pooled: Our ``summarize(...)["pooled"]`` frame, for the QLIKE-level
            comparison against Table 4.
        mz: Our :func:`voleval.metrics.mincer_zarnowitz` frame, for Table 9.

    Returns:
        One row per (model, horizon) with our value, Brini's value, the
        difference and the relative difference for each comparable quantity.
    """
    ours = equal_weighted[["model", "horizon", "qlike", "qlike_ratio"]].copy()
    ours = ours.rename(columns={"qlike": "our_qlike_ew", "qlike_ratio": "our_ratio"})
    ours["brini_model"] = ours["model"].map(MODEL_NAME_MAP)

    merged = ours.merge(reference_table("ratio"), on=["brini_model", "horizon"], how="left")

    if pooled is not None:
        pooled_ours = pooled[["model", "horizon", "qlike"]].rename(
            columns={"qlike": "our_qlike_pooled"}
        )
        merged = merged.merge(pooled_ours, on=["model", "horizon"], how="left")
        merged = merged.merge(reference_table("qlike"), on=["brini_model", "horizon"], how="left")

    if mz is not None:
        mz_ours = mz[["model", "horizon", "alpha", "beta", "r2"]].rename(
            columns={"alpha": "our_alpha", "beta": "our_beta", "r2": "our_r2"}
        )
        merged = merged.merge(mz_ours, on=["model", "horizon"], how="left")
        merged = merged.merge(reference_table("mz"), on=["brini_model", "horizon"], how="left")

    # The ratio is the headline comparison: it is scale-free, so it survives any
    # difference in units, annualisation or preprocessing between the two runs.
    merged["ratio_diff"] = merged["our_ratio"] - merged["brini_ratio"]
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["ratio_pct_diff"] = 100.0 * merged["ratio_diff"] / merged["brini_ratio"]
        if "brini_qlike" in merged.columns:
            merged["qlike_pct_diff"] = (
                100.0 * (merged["our_qlike_pooled"] - merged["brini_qlike"]) / merged["brini_qlike"]
            )

    return merged.sort_values(["horizon", "our_ratio"]).reset_index(drop=True)


def format_comparison(comparison: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Slice :func:`compare_with_brini` to one horizon and tidy it for printing."""
    columns = [
        c for c in (
            "model", "brini_model",
            "our_qlike_pooled", "brini_qlike", "qlike_pct_diff",
            "our_ratio", "brini_ratio", "ratio_diff",
        )
        if c in comparison.columns
    ]
    sliced = comparison[comparison["horizon"] == horizon][columns].copy()
    return sliced.reset_index(drop=True)
