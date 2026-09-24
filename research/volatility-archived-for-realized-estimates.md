
# **VOLARE: The Open Dataset We Should Build On**

Paper: https://arxiv.org/html/2602.19732v1

**One-minute briefing** · Cipollini et al., *VOLatility Archive for Realized Estimates* · arXiv:2602.19732 Read the paper → · Platform → volare.unime.it

> **Why this matters to us:** this is the free, research-grade data source that fixes our biggest data problem — we were going to compute realized volatility from noisy daily yfinance prices, and VOLARE hands us proper 5-minute realized volatility already cleaned and computed. It's also the *same dataset the Brini benchmark paper used*, so building on it makes our results directly comparable to published work.
> 

## **What it is**

An open-access research infrastructure that turns ultra-high-frequency tick data into standardized realized volatility measures. It exists because the old standard (Oxford-Man Realized Library) was discontinued in 2022 with nothing to replace it. Free to download or explore interactively.

## **Why it fits our project**

- **Covers our exact tickers:** 40 big US stocks incl. **AAPL & AMZN** (109 in the extended set), 5 FX pairs, 5 futures — spanning **2015 to 30 Jan 2026** (essentially current).
- **Gives us the target *and* covariates, pre-computed:** realized variance (1-min & 5-min), bipower variation, positive/negative semivariances, realized quarticity, median/min RV, realized kernel, Parkinson & Garman-Klass ranges, OHLC — plus realized **covariance** matrices for our multivariate Chronos-2 mode.
- **Runs our baselines for us:** the platform estimates HAR, HAR-Q, MEM, and AMEM models and reports **MSE and QLIKE** — the exact evaluation metrics we're adopting. Great for sanity-checking our own HAR code.
- **Same data as Brini (2026):** using it makes our Chronos-2 numbers directly comparable to the published benchmark.

## **Useful methodology to cite/copy**

- 5-minute sampling = the standard noise/accuracy compromise (justifies our RV window).
- Cleaning: Brownlees–Gallo (2006) outlier filter; previous-tick sampling to a regular grid.
- QLIKE & MSE loss defined formally (Patton 2011).
- Baseline prediction intervals via empirical residual quantiles — a clean way to give our classical baselines uncertainty bands comparable to Chronos-2's quantiles.

## **What it does NOT solve (still on us)**

- **No implied volatility.** Our IV / mispricing angle still needs a separate live options source (yfinance chains). The historical-IV gap is unchanged.
- **Ends 30 Jan 2026, monthly updates.** Fine for the historical backtest core; our **live advisor tool still needs a live feed**. (This reinforces our two-tool split: VOLARE for the rigorous backtest, live sources for the demo.)
- **Two data quirks:** stock prices are *unadjusted* (no split/dividend adjustment), and Kibot **volume undercounts by ~20%** (odd-lot exclusion) — avoid or flag VOLARE volume if we use it as a covariate.

## **Action items for our project**

- **Use VOLARE as the primary data source** for the realized-vol forecasting core; pull our ticker set from it (→ comparable to Brini).
- **Keep yfinance** only for the live options/IV layer.
- **Add MEM/AMEM as a second classical baseline** alongside Log-HAR — the paper shows MEM fits individual-stock volatility better than HAR, and both are free in the platform.
- **Cite** Cipollini et al. (2026) for the data and the cleaning methodology.