## Ideas

- Univariate (single metric), covariate (single metric + signals), multivariate (multiple correlated metrics) are several areas we can focus on
    - Covariates: test with past covariates and known future covariates
    - Multivariates: find assets that correlate to have a better forecast as opposed to using just one asset
    - For multivariate + covariate, use some form of correlation to avoid wasting time and resources (i.e. find what signal is actually correlated to an asset)
- Make use of quantiles for adjusting risk tolerance
- Fine-tuning or training on this type of data
- 2 parts
    - Historical what-if simulator:
        - Pick a stock, past date, pull historical prices, compute the P&L, then what the model would have done, show comparison between self + model, maybe a small chatbot explaining why it was good or bad (teaches user why a trade is good or bad + shows model capability)
    - Live risk-based advisor:
        - Pick a stock, risk level, amount of capital, run forecast, apply risk-tier rules, show suggested trade

!backtest_vs_live_advisor.png

!trade_recommendation_pipeline.png

Start logging ticker’s current IV and then we will have implied, realized and forecasted to compare with.

S&P500 has IV in the past (what we have above).

Horizon forecast of 30-45 days or longer (shorter is too risky and greedy)

We need to forecast the realised volatility

Volatility clustering

Chronos forecasts will be off IV. Find the difference, and see what happened in the past to determine if there was any value (”when Chronos IV forecast was 10%+ what happened?”)

Feed the vol forecast *into* a pricing model (Black-Scholes or similar) to produce your *own* theoretical option prices, then compare against market prices to find rich/cheap options. This is really a more granular version of #1 — instead of comparing vol-to-vol, you compare price-to-price at the individual-contract level, which also lets you spot mispricings *within* the vol surface (specific strikes/expiries that are off). (**tldr finds the correct options to buy - IF DOING THIS, AVOID 1 FLAT IV ACROSS ALL STRIKES, separate IVs for separate strike prices)**

Use x y z (figure this out) to forecast realised volatility. Compare this to implied to determine if options and overpriced or underpriced. Then use user inputs (stock they want to trade + capital + risk tolerance), pull options chains, determine appropriate trades.

Possible xyz:

- Past realised volatility of a ticker (Ideally computed with a good estimator (Garman-Klass or Rogers-Satchell if you only have daily OHLC data; realized variance from intraday bars if you can get them) rather than noisy close-to-close.)
    - **Close-to-close** — noisiest, uses only closing prices
    - **Parkinson** — uses high-low range, ~5x more efficient
    - **Garman-Klass** — uses open, high, low, close; ~7-8x more efficient
    - **Rogers-Satchell** — like Garman-Klass but handles trending stocks (drift-independent)
    - **Intraday realized variance** — the gold standard, uses many intraday bars (needs high-frequency data)
- Earnings dates, macro calendar, VIX, volume (can be covariates)

Components:

- Extracting/pulling xyz
- Chronos-2 forecasting itself
- User input gathering
- Pulling options
- Component

Starting idea:

- Forecast RV using the possible xyz above, compare it to IV of the time and realised volatility, then do the same but introducing covariates (and find helpful covariates)
    - FV vs RV - > how accurate is Chronos? How good is our model?
    - FV vs IV - > If there is a gap between these, there is a possible trade.
        - FV > IV - > you think it'll move more than the market expects → options look cheap → **buy vol**
        - IV > FV - > you think it'll move less → options look expensive → **sell vol**
    - IV vs RV - > How good is the market at guessing? Usually a little over it. FV needs to be better than IV here to make it worth
- If possible, finding a way to use the forecasts and Black Scholes formula to forecast the options prices. We can then look at the chain and see how off they are and therefor pick the most optimal one (although black scholes is apparently only for European stocks…)