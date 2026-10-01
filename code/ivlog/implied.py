"""Implied volatility from one option-chain snapshot.  Pure maths, no network.

The headline number is a **30-day constant-maturity at-the-money implied
volatility** per ticker, built the way the VIX white paper builds its 30-day
horizon, simplified to the at-the-money point:

1. **Forward from put-call parity.**  For each expiry, take the strike where
   the call and put mid prices are closest and set
   ``F = K* + e^{rT} (C - P)``.  This folds dividends and borrow costs into
   the forward without needing a dividend forecast.
2. **Out-of-the-money options only.**  Calls with ``K >= F`` and puts with
   ``K < F``.  Listed US equity options are American; out-of-the-money options
   carry almost no early-exercise premium, so Black-76 on the forward is a
   good approximation for them and a poor one for deep in-the-money options.
3. **Implied volatility from the bid/ask mid** by inverting Black-76.  Yahoo's
   own ``impliedVolatility`` column is kept alongside for comparison; it is
   computed from the last trade, which can be hours stale.
4. **Smile interpolation** in log-moneyness ``k = ln(K / F)``: the ATM value
   is the smile at ``k = 0``; the skew points are ``K/F = 0.9`` and ``1.1``.
   No extrapolation beyond the quoted strikes.
5. **Constant maturity.**  Interpolate *total variance* ``w = sigma^2 T``
   linearly in ``T`` between the two expiries bracketing 30 calendar days,
   then ``sigma_30 = sqrt(w_30 / T_30)``.  Expiries under
   ``MIN_EXPIRY_DAYS`` are skipped: their quotes are dominated by noise and
   event risk, as in the VIX methodology.

Thirty calendar days is about 21 trading days, so ``iv30`` lines up with the
project's h = 22 forecast horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "YEAR_DAYS",
    "TARGET_DAYS",
    "MIN_EXPIRY_DAYS",
    "SKEW_MONEYNESS",
    "black76_price",
    "implied_vol",
    "implied_forward",
    "expiry_smile",
    "interp_smile",
    "constant_maturity",
    "ExpiryPoint",
    "TickerSummary",
    "summarize_chain",
]

YEAR_DAYS = 365.0
TARGET_DAYS = 30
MIN_EXPIRY_DAYS = 7
SKEW_MONEYNESS = (0.9, 1.1)
SIGMA_BOUNDS = (1e-4, 5.0)


# --------------------------------------------------------------------------- #
# Black-76 and its inverse
# --------------------------------------------------------------------------- #
def black76_price(F: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-76 price of a European option on a forward ``F``, discounted at ``r``."""
    df = math.exp(-r * T)
    if sigma <= 0 or T <= 0:
        return df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    vol = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    if is_call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price: float, F: float, K: float, T: float, r: float, is_call: bool) -> float:
    """Invert Black-76 for sigma.  ``NaN`` when the price violates no-arbitrage bounds."""
    if not (np.isfinite(price) and np.isfinite(F) and price > 0 and T > 0 and F > 0 and K > 0):
        return float("nan")
    df = math.exp(-r * T)
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    upper = df * F if is_call else df * K
    if price <= intrinsic + 1e-12 or price >= upper:
        return float("nan")
    lo, hi = SIGMA_BOUNDS

    def gap(s: float) -> float:
        return black76_price(F, K, T, r, s, is_call) - price

    try:
        if gap(lo) > 0 or gap(hi) < 0:
            return float("nan")
        return float(brentq(gap, lo, hi, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


# --------------------------------------------------------------------------- #
# One expiry
# --------------------------------------------------------------------------- #
def _with_mid(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    bid = pd.to_numeric(out["bid"], errors="coerce")
    ask = pd.to_numeric(out["ask"], errors="coerce")
    valid = (bid > 0) & (ask >= bid)
    out["mid"] = np.where(valid, (bid + ask) / 2.0, np.nan)
    return out


def implied_forward(calls: pd.DataFrame, puts: pd.DataFrame, T: float, r: float) -> tuple[float, float]:
    """Forward price from put-call parity at the strike where |C - P| is smallest.

    Returns ``(forward, strike_used)``, or ``(nan, nan)`` if no strike has both
    a valid call and put mid.
    """
    c = _with_mid(calls)[["strike", "mid"]].dropna()
    p = _with_mid(puts)[["strike", "mid"]].dropna()
    both = c.merge(p, on="strike", suffixes=("_c", "_p"))
    if both.empty:
        return float("nan"), float("nan")
    diff = both["mid_c"] - both["mid_p"]
    i = int(np.argmin(np.abs(diff.to_numpy())))
    k_star = float(both["strike"].iloc[i])
    return k_star + math.exp(r * T) * float(diff.iloc[i]), k_star


def expiry_smile(
    calls: pd.DataFrame, puts: pd.DataFrame, spot: float, T: float, r: float
) -> tuple[pd.DataFrame, float]:
    """Implied-vol smile for one expiry from out-of-the-money mids.

    ``calls`` and ``puts`` need ``strike``, ``bid``, ``ask`` and optionally
    ``impliedVolatility`` (Yahoo's value).  Returns the smile, sorted by
    log-moneyness, with columns ``strike, k, type, mid, iv_mid, iv_yahoo``,
    and the forward used.  Falls back to ``spot * e^{rT}`` for the forward
    when parity cannot be applied.
    """
    F, _ = implied_forward(calls, puts, T, r)
    if not np.isfinite(F) or F <= 0:
        F = spot * math.exp(r * T)
    rows = []
    for frame, is_call in ((calls, True), (puts, False)):
        f = _with_mid(frame)
        side = f[f["strike"] >= F] if is_call else f[f["strike"] < F]
        for _, row in side.iterrows():
            K = float(row["strike"])
            iv = implied_vol(float(row["mid"]), F, K, T, r, is_call) if np.isfinite(row["mid"]) else float("nan")
            yv = row.get("impliedVolatility", np.nan)
            rows.append(
                {
                    "strike": K,
                    "k": math.log(K / F),
                    "type": "call" if is_call else "put",
                    "mid": row["mid"],
                    "iv_mid": iv,
                    "iv_yahoo": float(yv) if yv is not None and np.isfinite(yv) and yv > 0 else np.nan,
                }
            )
    smile = pd.DataFrame(rows, columns=["strike", "k", "type", "mid", "iv_mid", "iv_yahoo"])
    return smile.sort_values("k", ignore_index=True), F


def interp_smile(smile: pd.DataFrame, k0: float, column: str = "iv_mid") -> float:
    """Linear interpolation of ``column`` at log-moneyness ``k0``.

    Needs a quoted point on each side of ``k0`` (or exactly at it): no
    extrapolation past the outermost usable strike.
    """
    pts = smile[["k", column]].dropna()
    if pts.empty:
        return float("nan")
    k = pts["k"].to_numpy()
    v = pts[column].to_numpy()
    if k0 < k.min() or k0 > k.max():
        return float("nan")
    return float(np.interp(k0, k, v))


# --------------------------------------------------------------------------- #
# Term structure -> 30 days
# --------------------------------------------------------------------------- #
def constant_maturity(
    points: list[tuple[float, float]], target_days: float = TARGET_DAYS, min_days: float = MIN_EXPIRY_DAYS
) -> tuple[float, float, float, str]:
    """Interpolate total variance to ``target_days``.

    ``points`` are ``(T_years, sigma)`` pairs, one per expiry.  Returns
    ``(sigma_target, near_days, far_days, flag)`` where ``flag`` is ``"ok"``
    when two expiries bracket the target, ``"near_only"`` / ``"far_only"``
    when the nearest usable expiry is used flat, or ``"none"``.
    """
    t_target = target_days / YEAR_DAYS
    usable = sorted(
        (T, s) for T, s in points if np.isfinite(s) and s > 0 and T * YEAR_DAYS >= min_days
    )
    if not usable:
        return float("nan"), float("nan"), float("nan"), "none"
    near = [p for p in usable if p[0] <= t_target]
    far = [p for p in usable if p[0] > t_target]
    if near and far:
        (t1, s1), (t2, s2) = near[-1], far[0]
        w1, w2 = s1 * s1 * t1, s2 * s2 * t2
        w = w1 + (w2 - w1) * (t_target - t1) / (t2 - t1)
        sigma = math.sqrt(w / t_target) if w > 0 else float("nan")
        return sigma, t1 * YEAR_DAYS, t2 * YEAR_DAYS, "ok"
    if far:
        t2, s2 = far[0]
        return s2, float("nan"), t2 * YEAR_DAYS, "far_only"
    t1, s1 = near[-1]
    return s1, t1 * YEAR_DAYS, float("nan"), "near_only"


# --------------------------------------------------------------------------- #
# A whole ticker
# --------------------------------------------------------------------------- #
@dataclass
class ExpiryPoint:
    expiry: pd.Timestamp
    T: float
    forward: float
    atm_mid: float
    atm_yahoo: float
    m90: float
    m110: float
    n_quotes: int


@dataclass
class TickerSummary:
    row: dict
    expiries: list[ExpiryPoint] = field(default_factory=list)
    chain: pd.DataFrame | None = None


def _expiry_close_utc(expiry: pd.Timestamp) -> pd.Timestamp:
    """16:00 New York time on the expiry date, in UTC."""
    local = pd.Timestamp(expiry.date()) + pd.Timedelta(hours=16)
    return local.tz_localize("America/New_York").tz_convert("UTC")


def summarize_chain(
    ticker: str, chain: pd.DataFrame, spot: float, r: float, asof: pd.Timestamp
) -> TickerSummary:
    """Turn one ticker's raw chain into a summary row plus a tidy chain table.

    ``chain`` has one row per contract with columns ``expiry`` (date),
    ``type`` (``"call"``/``"put"``), ``strike``, ``bid``, ``ask`` and
    optionally ``lastPrice``, ``impliedVolatility``, ``volume``,
    ``openInterest``.  ``asof`` is the snapshot time (tz-aware).
    """
    asof = pd.Timestamp(asof)
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    points: list[ExpiryPoint] = []
    tidy_parts = []
    for expiry, grp in chain.groupby("expiry"):
        expiry = pd.Timestamp(expiry)
        T = (_expiry_close_utc(expiry) - asof).total_seconds() / (YEAR_DAYS * 86400)
        if T <= 0:
            continue
        calls = grp[grp["type"] == "call"]
        puts = grp[grp["type"] == "put"]
        smile, F = expiry_smile(calls, puts, spot, T, r)
        points.append(
            ExpiryPoint(
                expiry=expiry,
                T=T,
                forward=F,
                atm_mid=interp_smile(smile, 0.0, "iv_mid"),
                atm_yahoo=interp_smile(smile, 0.0, "iv_yahoo"),
                m90=interp_smile(smile, math.log(SKEW_MONEYNESS[0]), "iv_mid"),
                m110=interp_smile(smile, math.log(SKEW_MONEYNESS[1]), "iv_mid"),
                n_quotes=int(smile["iv_mid"].notna().sum()),
            )
        )
        part = grp.copy()
        part = part.merge(smile[["strike", "type", "iv_mid"]], on=["strike", "type"], how="left")
        part["forward"] = F
        part["T_years"] = T
        tidy_parts.append(part)

    iv30, near_d, far_d, flag = constant_maturity([(p.T, p.atm_mid) for p in points])
    iv30_y, _, _, _ = constant_maturity([(p.T, p.atm_yahoo) for p in points])
    m90, _, _, _ = constant_maturity([(p.T, p.m90) for p in points])
    m110, _, _, _ = constant_maturity([(p.T, p.m110) for p in points])
    row = {
        "ticker": ticker,
        "spot": spot,
        "rate": r,
        "iv30_atm": iv30,
        "iv30_atm_yahoo": iv30_y,
        "iv30_m90": m90,
        "iv30_m110": m110,
        "skew_90_110": m90 - m110 if np.isfinite(m90) and np.isfinite(m110) else float("nan"),
        "near_days": near_d,
        "far_days": far_d,
        "n_expiries": len(points),
        "flag": flag,
    }
    chain_out = pd.concat(tidy_parts, ignore_index=True) if tidy_parts else None
    return TickerSummary(row=row, expiries=points, chain=chain_out)
