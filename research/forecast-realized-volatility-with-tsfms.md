
# **Foundation Models vs Classical Models for Volatility Forecasting**

Paper: https://arxiv.org/html/2607.05291v1


**One-minute briefing** · Brini, *Forecasting Realized Volatility with Time Series Foundation Models* · arXiv:2607.05291 (Duke, July 2026) Read the paper →

> **Why this matters to us:** it's basically our project, done rigorously and published. That's *good* — it hands us the benchmark, the evaluation methods, and a clear gap to fill. It does not kill our project, but it does change a few of our choices (see Action Items).
> 

## **What the paper does**

The first systematic head-to-head of **9 zero-shot foundation models** vs **8 classical econometric models**, forecasting **realized volatility** (not price) across **50 assets** — 40 US stocks (incl. AAPL & AMZN), 5 FX pairs, 5 futures — at **1-, 5-, and 22-day** horizons. "Zero-shot" = models used off-the-shelf with no fine-tuning, same as our plan. Uses the VOLARE dataset (high-quality 5-minute realized volatility, 2015–Jan 2026).

## **Main findings**

- **Foundation models do NOT clearly win.** Only one — TTM, the *smallest* model tested (<1M params) — beats the classical benchmark at every horizon, and only by ~1.3–1.8%. The other 8 don't beat it on average.
- **Classical HAR models stay competitive** across the board.
- **Which model you pick matters more than foundation-vs-classical** — their single most durable conclusion.
- **Much of the apparent "win" is just calibration** (forecasts sitting at the right level), not genuinely better prediction. A real information gain only appears at the **monthly (22-day) horizon**.

## **Evaluation methods (worth copying)**

- **QLIKE** loss — the standard volatility loss — not just MAE/RMSE.
- Report **per-asset equal-weighted** results, not only pooled averages: pooled numbers get hijacked by a few outlier stocks.
- **Diebold–Mariano** tests + **Model Confidence Set** for statistical significance across many models.
- Robustness: **pre/post-COVID** split and context-window sensitivity checks.

## **The gap WE can close**

1. **They never test Chronos-2** — they use the older Chronos-Bolt. Chronos-2 adds **covariate + multivariate** support, which is *exactly* our univariate / covariate / multivariate ablation. This unanswered question is our contribution.
2. **They ignore implied volatility and mispricing entirely.** Our IV-comparison and options-application layer is fully ours.
3. **They only score point forecasts.** Our probabilistic **calibration** angle is distinct.

## **Action items for our project**

- **Swap our headline baseline from GARCH → Log-HAR.** The field standard for realized vol is the HAR family; this paper deliberately excludes GARCH (different information set). HAR is a simple OLS regression on past daily/weekly/monthly volatility — easy to build. Keep GARCH only as a secondary comparison.
- **Adopt QLIKE + DM test + Model Confidence Set** so our results are credible to an examiner who knows this literature.
- **Consider using VOLARE directly** (same tickers, proper 5-min RV) instead of our noisier daily yfinance estimate — or explicitly flag the measurement difference as a limitation.

## **Our one-line positioning**

> *"Brini (2026) shows first-generation univariate foundation models mostly fail to beat classical HAR models on realized volatility; we test whether Chronos-2's covariate and multivariate features change that, and extend the evaluation to implied-volatility mispricing and probabilistic calibration."*
>