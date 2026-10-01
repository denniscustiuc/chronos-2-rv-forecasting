"""Offline tests for the implied-volatility maths in ``ivlog.implied``.

Synthetic option chains are priced with Black-76 from a known volatility
surface, so every test checks that the pipeline recovers a number we chose.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ivlog.implied import (
    YEAR_DAYS,
    black76_price,
    constant_maturity,
    expiry_smile,
    implied_forward,
    implied_vol,
    interp_smile,
    summarize_chain,
)

ASOF = pd.Timestamp("2026-10-01 19:07", tz="UTC")  # 15:07 New York
SPOT = 200.0
RATE = 0.04


def _expiry_after(days: int) -> pd.Timestamp:
    return pd.Timestamp((ASOF + pd.Timedelta(days=days)).date())


def _years_to(expiry: pd.Timestamp) -> float:
    close = (pd.Timestamp(expiry.date()) + pd.Timedelta(hours=16)).tz_localize("America/New_York")
    return (close.tz_convert("UTC") - ASOF).total_seconds() / (YEAR_DAYS * 86400)


def synthetic_chain(
    expiry_days=(3, 10, 24, 38, 66),
    vol=lambda k, T: 0.25,
    div_yield=0.0,
    half_spread=0.0,
    lo=0.7,
    hi=1.3,
    step=2.5,
) -> pd.DataFrame:
    """Price calls and puts on a strike grid from a chosen surface ``vol(k, T)``."""
    rows = []
    strikes = np.arange(round(SPOT * lo / step) * step, SPOT * hi + 1e-9, step)
    for d in expiry_days:
        expiry = _expiry_after(d)
        T = _years_to(expiry)
        F = SPOT * math.exp((RATE - div_yield) * T)
        for K in strikes:
            s = vol(math.log(K / F), T)
            for is_call in (True, False):
                p = black76_price(F, K, T, RATE, s, is_call)
                rows.append(
                    {
                        "expiry": expiry,
                        "type": "call" if is_call else "put",
                        "strike": float(K),
                        "bid": max(p - half_spread, 0.0),
                        "ask": p + half_spread,
                        "lastPrice": p,
                        "impliedVolatility": s,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Black-76 and its inverse
# --------------------------------------------------------------------------- #
def test_black76_satisfies_put_call_parity():
    F, K, T, r, s = 205.0, 190.0, 0.2, 0.04, 0.3
    c = black76_price(F, K, T, r, s, True)
    p = black76_price(F, K, T, r, s, False)
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), abs=1e-10)


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("K", [150.0, 200.0, 260.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trips(sigma, K, is_call):
    F, T, r = 200.0, 0.1, 0.04
    price = black76_price(F, K, T, r, sigma, is_call)
    intrinsic = math.exp(-r * T) * max((F - K) if is_call else (K - F), 0.0)
    if price - intrinsic < 1e-6:  # no time value left: the price carries no volatility information
        return
    assert implied_vol(price, F, K, T, r, is_call) == pytest.approx(sigma, rel=1e-6)


def test_implied_vol_rejects_prices_outside_no_arbitrage_bounds():
    F, K, T, r = 200.0, 180.0, 0.1, 0.04
    intrinsic = math.exp(-r * T) * (F - K)
    assert math.isnan(implied_vol(intrinsic * 0.99, F, K, T, r, True))
    assert math.isnan(implied_vol(F * 1.01, F, K, T, r, True))
    assert math.isnan(implied_vol(0.0, F, K, T, r, True))


# --------------------------------------------------------------------------- #
# Forward and smile
# --------------------------------------------------------------------------- #
def test_forward_from_parity_absorbs_dividends():
    chain = synthetic_chain(expiry_days=(38,), div_yield=0.03)
    expiry = chain["expiry"].iloc[0]
    T = _years_to(expiry)
    calls, puts = chain[chain["type"] == "call"], chain[chain["type"] == "put"]
    F, _ = implied_forward(calls, puts, T, RATE)
    assert F == pytest.approx(SPOT * math.exp((RATE - 0.03) * T), rel=1e-9)


def test_smile_uses_only_out_of_the_money_options():
    chain = synthetic_chain(expiry_days=(38,))
    T = _years_to(chain["expiry"].iloc[0])
    smile, F = expiry_smile(chain[chain["type"] == "call"], chain[chain["type"] == "put"], SPOT, T, RATE)
    assert (smile.loc[smile["type"] == "call", "strike"] >= F).all()
    assert (smile.loc[smile["type"] == "put", "strike"] < F).all()


def test_interp_smile_refuses_to_extrapolate():
    smile = pd.DataFrame({"k": [-0.1, 0.0, 0.1], "iv_mid": [0.3, 0.25, 0.22]})
    assert interp_smile(smile, 0.05) == pytest.approx(0.235)
    assert math.isnan(interp_smile(smile, 0.2))
    assert math.isnan(interp_smile(smile, -0.2))


# --------------------------------------------------------------------------- #
# Constant maturity
# --------------------------------------------------------------------------- #
def test_constant_maturity_interpolates_total_variance():
    t1, t2 = 24 / YEAR_DAYS, 38 / YEAR_DAYS
    s1, s2 = 0.20, 0.30
    sigma, near, far, flag = constant_maturity([(t1, s1), (t2, s2)])
    t = 30 / YEAR_DAYS
    w = s1**2 * t1 + (s2**2 * t2 - s1**2 * t1) * (t - t1) / (t2 - t1)
    assert flag == "ok"
    assert (near, far) == (pytest.approx(24), pytest.approx(38))
    assert sigma == pytest.approx(math.sqrt(w / t))


def test_constant_maturity_flags_missing_brackets():
    assert constant_maturity([(45 / YEAR_DAYS, 0.3)])[3] == "far_only"
    assert constant_maturity([(20 / YEAR_DAYS, 0.3)])[3] == "near_only"
    assert constant_maturity([])[3] == "none"


def test_constant_maturity_skips_expiries_under_a_week():
    sigma, near, _, _ = constant_maturity([(3 / YEAR_DAYS, 2.0), (24 / YEAR_DAYS, 0.2), (38 / YEAR_DAYS, 0.2)])
    assert near == pytest.approx(24)
    assert sigma == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Whole ticker
# --------------------------------------------------------------------------- #
def test_flat_surface_is_recovered():
    summary = summarize_chain("TEST", synthetic_chain(half_spread=0.02), SPOT, RATE, ASOF)
    assert summary.row["flag"] == "ok"
    assert summary.row["iv30_atm"] == pytest.approx(0.25, abs=1e-6)
    assert summary.row["iv30_atm_yahoo"] == pytest.approx(0.25, abs=1e-9)


def test_term_structure_and_skew_are_recovered():
    # Vol linear in log-moneyness (so strike interpolation is exact) and rising with maturity.
    def vol(k, T):
        return 0.20 + 0.5 * T - 0.3 * k

    chain = synthetic_chain(vol=vol, div_yield=0.01)
    s = summarize_chain("TEST", chain, SPOT, RATE, ASOF)
    near = _years_to(_expiry_after(24))
    far = _years_to(_expiry_after(38))
    expected = constant_maturity([(near, vol(0, near)), (far, vol(0, far))])[0]
    assert s.row["iv30_atm"] == pytest.approx(expected, abs=1e-6)
    # Skew between K/F = 0.9 and 1.1 is -0.3 * (ln 0.9 - ln 1.1) at every maturity.
    assert s.row["skew_90_110"] == pytest.approx(0.3 * math.log(1.1 / 0.9), abs=2e-3)


def test_short_dated_noise_does_not_leak_into_iv30():
    def vol(k, T):
        return 1.5 if T * YEAR_DAYS < 7 else 0.25

    s = summarize_chain("TEST", synthetic_chain(vol=vol), SPOT, RATE, ASOF)
    assert s.row["iv30_atm"] == pytest.approx(0.25, abs=1e-6)


def test_tidy_chain_keeps_every_contract():
    chain = synthetic_chain()
    s = summarize_chain("TEST", chain, SPOT, RATE, ASOF)
    assert len(s.chain) == len(chain)
    assert {"forward", "T_years", "iv_mid"} <= set(s.chain.columns)


def test_default_tickers_match_the_brini_equities():
    from ivlog.tickers import DEFAULT_TICKERS
    from voleval.brini import BRINI_EQUITIES

    assert set(BRINI_EQUITIES) <= set(DEFAULT_TICKERS)

