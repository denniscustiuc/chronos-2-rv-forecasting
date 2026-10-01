"""Tickers the daily IV log collects.

The 40 US equities from Brini (2026), copied from ``voleval.brini`` so this
package imports without the modelling stack (a test checks the two lists
match), plus SPY as a market-wide reference.
"""

BRINI_EQUITY_TICKERS: tuple[str, ...] = (
    "AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO",
    "CVX", "DIS", "GE", "GOOGL", "GS", "HD", "HON", "IBM", "JNJ", "JPM",
    "KO", "MCD", "META", "MMM", "MRK", "MSFT", "NFLX", "NKE", "NVDA", "ORCL",
    "PG", "PM", "SHW", "TRV", "TSLA", "UNH", "V", "VZ", "WMT", "XOM",
)

DEFAULT_TICKERS: tuple[str, ...] = BRINI_EQUITY_TICKERS + ("SPY",)
