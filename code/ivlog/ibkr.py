"""Daily implied-volatility history from Interactive Brokers.

IBKR's historical-data service can return one bar per day of an underlying's
**implied volatility** (``whatToShow="OPTION_IMPLIED_VOLATILITY"``) and its
**historical (realized) volatility** (``"HISTORICAL_VOLATILITY"``) for stocks,
ETFs and indices.  IBKR describes the implied volatility as "the at-market
volatility estimated for a maturity thirty calendar days forward of the
current trading day ... based on option prices from two consecutive expiration
months", i.e. a 30-day at-the-money IV, the same quantity ``ivlog.implied``
computes from a chain.  Thirty calendar days is about 21 trading days, so it
lines up with the project's h = 22 forecast horizon.

The API only talks to a running, logged-in TWS or IB Gateway on the same
machine, so everything here runs on a desktop, not in the cloud.  ``ib_async``
is imported lazily so the pure helpers stay testable without it.

How far back the history goes is not documented; ``earliest_dates`` asks IBKR
(``reqHeadTimeStamp``) per ticker.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

IV = "OPTION_IMPLIED_VOLATILITY"
HV = "HISTORICAL_VOLATILITY"
DATA_FILE = "ibkr_iv_daily.csv"
COLUMNS = ["date", "ticker", "iv", "iv_open", "iv_high", "iv_low", "hv"]
# Ports TWS / IB Gateway listen on by default.
PORTS = {"tws": 7496, "tws-paper": 7497, "gateway": 4001, "gateway-paper": 4002}


# --------------------------------------------------------------------------- #
# Pure helpers (tested offline)
# --------------------------------------------------------------------------- #
def year_chunks(start: date, end: date, years: int = 2) -> list[tuple[datetime, str]]:
    """Split ``[start, end]`` into requests of at most ``years`` years.

    Returns ``(end_datetime_utc, duration_string)`` pairs, newest first, in the
    form ``reqHistoricalData`` takes.  Consecutive chunks overlap by about a
    week so month-end and leap-day arithmetic can never leave a gap; the
    caller drops the duplicate days and anything before ``start``.
    """
    if end < start:
        return []
    out = []
    cursor = end
    while cursor >= start:
        out.append((datetime(cursor.year, cursor.month, cursor.day, 23, 59, 59, tzinfo=timezone.utc), f"{years} Y"))
        reach = date(cursor.year - years, cursor.month, min(cursor.day, 28))
        cursor = reach + timedelta(days=7)
    return out


def bars_to_frame(bars: Iterable, ticker: str, prefix: str) -> pd.DataFrame:
    """Turn IBKR ``BarData`` objects into ``date, ticker, <prefix>_open/high/low/close``."""
    rows = [
        {
            "date": pd.Timestamp(b.date).strftime("%Y-%m-%d"),
            "ticker": ticker,
            f"{prefix}_open": float(b.open),
            f"{prefix}_high": float(b.high),
            f"{prefix}_low": float(b.low),
            f"{prefix}_close": float(b.close),
        }
        for b in bars
    ]
    cols = ["date", "ticker", f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"]
    return pd.DataFrame(rows, columns=cols).drop_duplicates(["date", "ticker"], keep="last")


def combine(iv: pd.DataFrame, hv: pd.DataFrame) -> pd.DataFrame:
    """Join IV and HV bars into the stored schema (``iv`` is the daily close)."""
    out = iv.merge(hv[["date", "ticker", "hv_close"]], on=["date", "ticker"], how="outer")
    out = out.rename(columns={"iv_close": "iv", "hv_close": "hv"})
    for c in COLUMNS:
        if c not in out:
            out[c] = float("nan")
    return out[COLUMNS].sort_values(["ticker", "date"], ignore_index=True)


def upsert_csv(path: Path, new: pd.DataFrame, keys: tuple[str, ...] = ("date", "ticker")) -> int:
    """Merge ``new`` into ``path``; rows with the same keys are replaced.  Returns the row count."""
    keys = list(keys)
    if path.exists():
        old = pd.read_csv(path, dtype={"date": str, "ticker": str})
        old = old.merge(new[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(keys[::-1], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6g")
    return len(new)


# --------------------------------------------------------------------------- #
# Talking to TWS / IB Gateway
# --------------------------------------------------------------------------- #
@dataclass
class Connection:
    host: str = "127.0.0.1"
    port: int = PORTS["tws"]
    client_id: int = 17
    timeout: float = 60.0


class IBKRClient:
    """Thin wrapper over ``ib_async.IB`` that logs IBKR's error messages per ticker."""

    def __init__(self, conn: Connection):
        from ib_async import IB  # noqa: PLC0415 - optional dependency

        self.ib = IB()
        self.conn = conn
        self.errors: list[str] = []
        self.ib.errorEvent += self._on_error

    def _on_error(self, req_id, code, message, contract=None, *_):
        if code in (2104, 2106, 2107, 2108, 2158, 2174):  # "data farm connection is OK" and similar notices
            return
        sym = getattr(contract, "symbol", "")
        self.errors.append(f"{sym} [{code}] {message}")
        log.warning("IBKR %s [%s] %s", sym or "-", code, message)

    def __enter__(self):
        c = self.conn
        self.ib.connect(c.host, c.port, clientId=c.client_id, timeout=c.timeout, readonly=True)
        return self

    def __exit__(self, *exc):
        self.ib.disconnect()

    def stock(self, ticker: str):
        from ib_async import Index, Stock  # noqa: PLC0415

        contract = Index(ticker.lstrip("^"), "CBOE", "USD") if ticker.startswith("^") else Stock(ticker, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"IBKR does not recognise {ticker}")
        return qualified[0]

    def earliest(self, contract, what: str = IV) -> date | None:
        ts = self.ib.reqHeadTimeStamp(contract, whatToShow=what, useRTH=True, formatDate=1)
        if not ts:
            return None
        return pd.Timestamp(ts).date()

    def daily_bars(self, contract, what: str, end: datetime | str, duration: str) -> list:
        return self.ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow=what,
            useRTH=True,
            formatDate=1,
            timeout=self.conn.timeout,
        )


def fetch_range(client: IBKRClient, ticker: str, start: date, end: date, chunk_years: int = 2, pause: float = 1.0) -> pd.DataFrame:
    """Daily IV and HV for ``ticker`` between ``start`` and ``end`` (inclusive)."""
    contract = client.stock(ticker)
    parts = {IV: [], HV: []}
    for what in (IV, HV):
        for end_dt, duration in year_chunks(start, end, chunk_years):
            bars = client.daily_bars(contract, what, end_dt, duration)
            time.sleep(pause)  # stay well inside IBKR's pacing limits
            if not bars:
                break  # walking back in time: no bars means history starts later (or IBKR refused)
            parts[what].append(bars_to_frame(bars, ticker, "iv" if what == IV else "hv"))
    empty = {p: bars_to_frame([], ticker, p) for p in ("iv", "hv")}
    iv = pd.concat(parts[IV] or [empty["iv"]], ignore_index=True).drop_duplicates(["date", "ticker"])
    hv = pd.concat(parts[HV] or [empty["hv"]], ignore_index=True).drop_duplicates(["date", "ticker"])
    out = combine(iv, hv)
    lo, hi = start.isoformat(), end.isoformat()
    return out[(out["date"] >= lo) & (out["date"] <= hi)].reset_index(drop=True)
