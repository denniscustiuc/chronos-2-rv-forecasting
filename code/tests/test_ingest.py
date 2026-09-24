"""Cleaning and index-normalisation tests (no network access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volpipe.ingest import DataQualityReport, clean_ohlcv, normalize_index


def frame_from(rows: dict, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def test_clean_keeps_good_rows(small_ohlc):
    report = DataQualityReport(ticker="TEST")
    cleaned = clean_ohlcv(small_ohlc, report)
    assert len(cleaned) == len(small_ohlc)
    assert report.total_dropped == 0
    assert report.rows_kept == report.rows_downloaded == len(small_ohlc)


def test_drops_missing_nonpositive_and_inconsistent_rows():
    frame = frame_from(
        {
            "open": [100.0, np.nan, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, -1.0, 99.0],   # row 4: high < open
            "low": [99.0, 99.0, 99.0, 99.0, 98.0],
            "close": [100.5, 100.5, 0.0, 100.5, 100.5],  # row 2: zero close
            "volume": [1e6] * 5,
        },
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"],
    )
    report = DataQualityReport(ticker="TEST")
    cleaned = clean_ohlcv(frame, report)

    assert len(cleaned) == 1
    assert cleaned.index[0] == pd.Timestamp("2020-01-02")
    assert report.dropped["missing_ohlc"] == 1
    assert report.dropped["nonpositive_price"] == 2  # zero close and negative high
    assert report.dropped["ohlc_inconsistent"] == 1
    assert report.total_dropped == 4
    assert len(report.dropped_dates) == 4


def test_duplicate_dates_keep_last():
    frame = frame_from(
        {
            "open": [100.0, 200.0],
            "high": [101.0, 201.0],
            "low": [99.0, 199.0],
            "close": [100.5, 200.5],
            "volume": [1e6, 2e6],
        },
        ["2020-01-02", "2020-01-02"],
    )
    report = DataQualityReport(ticker="TEST")
    cleaned = clean_ohlcv(frame, report)
    assert len(cleaned) == 1
    assert float(cleaned["close"].iloc[0]) == 200.5
    assert report.dropped["duplicate_date"] == 1


def test_zero_volume_becomes_nan_without_dropping_the_row():
    frame = frame_from(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [0.0]},
        ["2020-01-02"],
    )
    report = DataQualityReport(ticker="TEST")
    cleaned = clean_ohlcv(frame, report)
    assert len(cleaned) == 1
    assert np.isnan(cleaned["volume"].iloc[0])
    assert report.total_dropped == 0
    assert any("volume" in w for w in report.warnings)


def test_empty_frame_is_reported_not_raised():
    report = DataQualityReport(ticker="TEST")
    cleaned = clean_ohlcv(pd.DataFrame(), report)
    assert cleaned.empty
    assert report.warnings


def test_normalize_index_strips_timezone_and_time():
    tz_aware = pd.DatetimeIndex(["2024-01-02 16:00:00-05:00", "2024-01-03 16:00:00-05:00"])
    out = normalize_index(tz_aware)
    assert out.tz is None
    assert (out == pd.DatetimeIndex(["2024-01-02", "2024-01-03"])).all()


def test_report_to_dict_round_trips():
    import json

    report = DataQualityReport(ticker="TEST")
    report.record_drop("missing_ohlc", pd.DatetimeIndex(["2020-01-02"]))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["rows_dropped"] == 1
    assert payload["dropped_by_reason"] == {"missing_ohlc": 1}
