## TO BE EDITED (PLACEHOLDER)

**Simplified (elevator pitch)**

This project uses a model built for forecasting to predict how much a stock or crypto asset's price is likely to swing, rather than trying to guess which direction it'll go. This is useful because predicting the size of price swings is actually more realistic than predicting direction, and it's exactly the kind of information investors and risk managers use to price options and manage risk.

**Project summary**

This project applies Chronos-2, a zero-shot time-series foundation model, to forecast short-term volatility in financial assets such as equities or cryptocurrency, rather than the more commonly attempted task of predicting price direction. The model's forecasts will be benchmarked against standard volatility models (GARCH/EGARCH) using realized volatility as ground truth, evaluated across multiple assets and market regimes (calm vs. volatile periods). The goal is to determine whether a general-purpose forecasting model, with no finance-specific training, can match or outperform the domain-standard statistical approach - and to understand where and why it succeeds or fails. If deemed appropriate, we could fine-tune or even train from scratch specifically on the finance domain to yield higher results.

**Why this project**

- Volatility forecasting has direct real-world value - it underpins options pricing, risk management, and position sizing
- Volatility is more statistically tractable than price direction - under efficient-market assumptions, price movement direction is largely unpredictable, but the magnitude of movement is not
- GARCH-family models have been the industry standard for decades, so testing a modern foundation model against them is a meaningful, well-motivated comparison rather than an arbitrary application of ML to finance
- Requires real evaluation rigor (backtesting, regime analysis) rather than a shallow demo, making it a substantial dissertation topic

