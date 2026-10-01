"""Offline tests for the IBKR download helpers (no TWS, no ib_async needed)."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from ivlog.ibkr import COLUMNS, bars_to_frame, combine, upsert_csv, year_chunks


def _covered(chunks) -> set[date]:
    """Days a list of (end, "N Y") requests would cover."""
    days = set()
    for end_dt, duration in chunks:
        years = int(duration.split()[0])
        end = end_dt.date()
        # Conservative: assume IBKR returns only days strictly after (end - N years).
        start = (pd.Timestamp(end) - pd.DateOffset(years=years) + pd.Timedelta(days=1)).date()
        d = start
        while d <= end:
            days.add(d)
            d += timedelta(days=1)
    return days


@pytest.mark.parametrize("start, end", [("2015-01-01", "2026-10-01"), ("2024-02-29", "2026-03-31"), ("2026-09-01", "2026-10-01")])
@pytest.mark.parametrize("years", [1, 2, 3])
def test_year_chunks_leave_no_gaps(start, end, years):
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    chunks = year_chunks(s, e, years)
    covered = _covered(chunks)
    d = s
    while d <= e:
        assert d in covered, d
        d += timedelta(days=1)
    ends = [c[0] for c in chunks]
    assert ends == sorted(ends, reverse=True)


def test_year_chunks_empty_when_range_is_backwards():
    assert year_chunks(date(2026, 1, 2), date(2026, 1, 1)) == []


def _bar(day, value):
    return SimpleNamespace(date=day, open=value, high=value + 0.01, low=value - 0.01, close=value)


def test_bars_to_frame_and_combine_build_the_stored_schema():
    iv = bars_to_frame([_bar(date(2026, 9, 29), 0.25), _bar(date(2026, 9, 30), 0.27)], "AAPL", "iv")
    hv = bars_to_frame([_bar(date(2026, 9, 30), 0.22)], "AAPL", "hv")
    out = combine(iv, hv)
    assert list(out.columns) == COLUMNS
    assert out["date"].tolist() == ["2026-09-29", "2026-09-30"]
    assert out["iv"].tolist() == [0.25, 0.27]
    assert pd.isna(out.loc[0, "hv"]) and out.loc[1, "hv"] == 0.22


def test_bars_to_frame_handles_no_bars():
    assert bars_to_frame([], "AAPL", "iv").empty


def test_upsert_replaces_overlapping_days(tmp_path):
    path = tmp_path / "iv.csv"
    first = pd.DataFrame({"date": ["2026-09-29", "2026-09-30"], "ticker": ["AAPL", "AAPL"], "iv": [0.25, 0.27]})
    upsert_csv(path, first)
    second = pd.DataFrame({"date": ["2026-09-30", "2026-10-01"], "ticker": ["AAPL", "AAPL"], "iv": [0.28, 0.30]})
    assert upsert_csv(path, second) == 3
    saved = pd.read_csv(path, dtype={"date": str})
    assert saved["iv"].tolist() == [0.25, 0.28, 0.30]
