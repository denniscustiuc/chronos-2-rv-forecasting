"""PipelineConfig validation tests."""

from __future__ import annotations

import pytest

from volpipe.config import COVARIATE_TAGS, PipelineConfig


def make(**overrides) -> PipelineConfig:
    kwargs = {"tickers": ("AAPL",), "start": "2020-01-01", "end": "2021-01-01"}
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def test_defaults():
    config = make()
    assert config.rv_window == 21
    assert config.annualization_factor == 252
    assert "close_to_close" in config.estimators
    assert config.covariate_tags()["earnings_flag"] == "known_future"
    assert config.covariate_tags()["vix"] == "past_only"


def test_every_known_covariate_is_tagged():
    assert set(COVARIATE_TAGS) >= {"vix", "volume", "earnings_flag", "sector_etf_vol", "risk_free_rate"}
    assert set(COVARIATE_TAGS.values()) <= {"past_only", "known_future"}
    # earnings_flag must be the only known-future covariate.
    known_future = [k for k, v in COVARIATE_TAGS.items() if v == "known_future"]
    assert known_future == ["earnings_flag"]


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"tickers": ()}, "must not be empty"),
        ({"rv_window": 1}, "rv_window"),
        ({"annualization_factor": 0}, "annualization_factor"),
        ({"earnings_window_days": -1}, "earnings_window_days"),
        ({"start": "2022-01-01", "end": "2021-01-01"}, "must be before"),
        ({"estimators": ("nope",)}, "unknown estimator"),
        ({"covariates": ("nope",)}, "unknown covariate"),
        ({"sector_etf_estimator": "intraday_rv"}, "cannot be 'intraday_rv'"),
    ],
)
def test_invalid_configs_raise(overrides, message):
    with pytest.raises(ValueError, match=message):
        make(**overrides)


def test_sector_etf_lookup():
    config = make(sector_etf_map={"AAPL": "XLK"}, default_sector_etf="SPY")
    assert config.sector_etf_for("AAPL") == "XLK"
    assert config.sector_etf_for("aapl") == "XLK"
    assert config.sector_etf_for("UNKNOWN") == "SPY"


def test_to_dict_is_json_serialisable():
    import json

    payload = json.dumps(make().to_dict())
    assert "rv_window" in payload
