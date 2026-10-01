"""Build the implied-volatility data bank from Interactive Brokers.

Needs TWS or IB Gateway running and logged in on this computer, with the API
turned on (TWS: Edit > Global Configuration > API > Settings > "Enable
ActiveX and Socket Clients").  Then, from the repo root:

    python code/iv_ibkr.py head                       # how far back IBKR's IV history goes, per ticker
    python code/iv_ibkr.py backfill --start 2015-01-01  # once: download the history
    python code/iv_ibkr.py update                     # whenever you need recent days: refresh the last 60

There is no scheduled job: IBKR keeps the history, so the data bank is
downloaded once, committed to the repo as a fixed copy, and topped up on
demand.  Everything lands in ``data/iv/ibkr_iv_daily.csv``: one row per
(date, ticker) with IBKR's 30-day implied volatility (``iv`` is the daily
close, plus ``iv_open/high/low``), IBKR's historical volatility (``hv``) and
the date the row was downloaded (``retrieved``).  Re-running replaces
overlapping rows, so ``update`` fills any gap since the last download.

Default port is 7496 (TWS, live login).  Use ``--port 7497`` for TWS paper,
4001 for IB Gateway, 4002 for IB Gateway paper.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ivlog.ibkr import DATA_FILE, IV, Connection, IBKRClient, fetch_range, upsert_csv
from ivlog.tickers import DEFAULT_TICKERS

log = logging.getLogger("iv_ibkr")


def cmd_head(client: IBKRClient, args) -> int:
    rows = []
    for t in args.tickers:
        try:
            first = client.earliest(client.stock(t), IV)
        except Exception as exc:  # noqa: BLE001
            log.error("%-6s failed: %s", t, exc)
            first = None
        rows.append({"ticker": t, "earliest_iv_date": first})
        log.info("%-6s %s", t, first or "no IV history")
    out = Path(args.out) / "ibkr_iv_earliest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("saved %s", out)
    return 0


def _download(client: IBKRClient, args, start: date, end: date) -> int:
    path = Path(args.out) / DATA_FILE
    failed = []
    for t in args.tickers:
        try:
            frame = fetch_range(client, t, start, end, chunk_years=args.chunk_years, pause=args.pause)
            frame["retrieved"] = date.today().isoformat()
            if frame.empty:
                failed.append(t)
                log.error("%-6s no data returned (see IBKR messages above)", t)
                continue
            total = upsert_csv(path, frame)  # saved per ticker, so an interrupted backfill keeps its progress
            last = frame.dropna(subset=["iv"]).tail(1)
            log.info(
                "%-6s %5d days %s to %s, latest iv %s  (file now %d rows)",
                t, len(frame), frame["date"].iloc[0], frame["date"].iloc[-1],
                f"{last['iv'].iloc[0]:.4f}" if len(last) else "nan", total,
            )
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the rest
            failed.append(t)
            log.error("%-6s failed: %s", t, exc)
    log.info("%d/%d tickers saved; failed: %s", len(args.tickers) - len(failed), len(args.tickers), ", ".join(failed) or "none")
    return 1 if len(failed) > len(args.tickers) / 2 else 0


def cmd_backfill(client: IBKRClient, args) -> int:
    return _download(client, args, date.fromisoformat(args.start), date.today())


def cmd_update(client: IBKRClient, args) -> int:
    return _download(client, args, date.today() - timedelta(days=args.days), date.today())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Implied-volatility data bank from Interactive Brokers.")
    p.add_argument("command", choices=["head", "backfill", "update"])
    p.add_argument("--out", default="data/iv", help="output folder (default: data/iv)")
    p.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    p.add_argument("--start", default="2015-01-01", help="backfill start date (default: 2015-01-01, VOLARE's start)")
    p.add_argument("--days", type=int, default=60, help="update: calendar days to refresh (default: 60)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7496, help="7496 TWS, 7497 TWS paper, 4001 Gateway, 4002 Gateway paper")
    p.add_argument("--client-id", type=int, default=17)
    p.add_argument("--chunk-years", type=int, default=2, help="years of daily bars per request")
    p.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    conn = Connection(host=args.host, port=args.port, client_id=args.client_id)
    try:
        with IBKRClient(conn) as client:
            return {"head": cmd_head, "backfill": cmd_backfill, "update": cmd_update}[args.command](client, args)
    except (ConnectionRefusedError, TimeoutError, OSError):
        log.error(
            "could not connect to %s:%d. Is TWS or IB Gateway running and logged in, with the API "
            "enabled on that port?", args.host, args.port,
        )
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
