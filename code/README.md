# volpipe — realized-volatility data pipeline (yfinance path)

Produces, per ticker, a clean daily time series of realized-volatility estimates
computed several ways plus aligned covariates, ready to feed a forecasting layer.
**No forecasting lives here** — this is the data foundation only.

The project's primary *historical* source is the VOLARE dataset (pre-computed
5-minute realized measures). This module is the **yfinance path**, which exists for:

1. the live-data layer (VOLARE is not real-time),
2. tickers and dates outside VOLARE's coverage,
3. benchmarking our daily-OHLC estimators against VOLARE's 5-minute `rv5`.

## Install & run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python demo.py --years 5 --csv          # AAPL, AMZN, MSFT
.venv/bin/python -m pytest tests/ -q              # 76 tests, fully offline
```

## Layout

| File | Role |
| --- | --- |
| `volpipe/config.py` | `PipelineConfig` — tickers, dates, window, annualization, estimator/covariate lists, covariate tags |
| `volpipe/ingest.py` | yfinance download, OHLCV cleaning, `DataQualityReport` |
| `volpipe/estimators.py` | The six variance estimators + windowing + annualization |
| `volpipe/covariates.py` | VIX, volume, earnings flag, sector-ETF vol, risk-free rate |
| `volpipe/pipeline.py` | Orchestration, per-ticker frame, multi-ticker panel, JSON manifest |
| `volpipe/validation.py` | Correlation matrix, noise ranking, overlay plot, VOLARE cross-check |
| `demo.py` | End-to-end CLI demo |
| `tests/` | pytest suite (no network access — synthetic series with known volatility) |

## Output schema

One tidy frame per ticker, indexed by trading date (`date`), saved to
`output/<TICKER>.parquet` (+ `.csv` with `--csv`):

| Column | Meaning |
| --- | --- |
| `rv_cc`, `rv_parkinson`, `rv_gk`, `rv_rs`, `rv_yz`, `rv_intraday` | **Annualised realized volatility** over the rolling window ending at `t`: `sqrt(252 * windowed_variance)` |
| `var_cc`, `var_parkinson`, `var_gk`, `var_rs`, `var_yz`, `var_intraday` | **Per-day variance** estimate, daily units, un-annualised |
| `vix` | `^VIX` daily close — `past_only` |
| `volume` | The ticker's own share volume — `past_only` |
| `earnings_flag` | 1 within ±k trading days of a release — **`known_future`** |
| `sector_etf_vol` | (optional) Sector ETF RV via the same estimators — `past_only` |
| `risk_free_rate` | (optional) FRED `DGS3MO` — `past_only` |

Plus `output/panel.parquet`, a combined `(date, ticker)` MultiIndexed panel, and
`output/manifest.json` recording the config, date range, estimator columns,
covariate tags, row counts and every dropped row with its reason.

## Estimators

Let `O, H, L, C` be the day's adjusted prices and `C_prev` the previous close.

| Name | Per-day variance | Notes |
| --- | --- | --- |
| `close_to_close` | `r² `, `r = ln(C/C_prev)` | Windowed value is the **sample variance** of `r` (ddof=1), not the mean of `r²`. Noisiest — one price per day. |
| `parkinson` | `(1/(4 ln2)) · ln(H/L)²` | Range-based, ~5× more efficient than CC. Assumes zero drift; ignores the overnight gap. |
| `garman_klass` | `0.5·ln(H/L)² − (2ln2−1)·ln(C/O)²` | Full OHLC bar. Non-negative by construction. |
| `rogers_satchell` | `ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)` | **Drift-independent** — stays unbiased under a trend. Exactly 0 on a perfectly monotone session. |
| `yang_zhang` | `o² + k·c² + (1−k)·rs` | Most efficient daily estimator. The *windowed* form uses the proper sample variances of the overnight and open-to-close returns; the `var_yz` column is the per-day zero-mean analogue. |
| `intraday_rv` | `Σᵢ rᵢ²` over intraday bars | yfinance serves only ~60 days of 5-minute history, so this is NaN over most of a multi-year sample — **by design**. Degrades to an all-NaN column when unavailable. Historical intraday RV is VOLARE's job. |

Overnight coverage differs by estimator and this shows up in the levels:
`rv_cc` and `rv_yz` include the overnight gap, the range estimators and
`rv_intraday` measure the open-to-close session only. On 5y of AAPL that's
roughly 0.26/0.27 vs 0.22.

## Correctness / no-lookahead discipline

* **Rolling windows.** The value at date `t` uses only the `window`
  observations ending at and including `t` (`min_periods=window`, so partial
  leading windows stay NaN). `tests/test_pipeline.py::test_pipeline_output_has_no_lookahead`
  rebuilds the frame on a truncated history and asserts the values at `t` are bit-identical.
* **Past-only covariates** are forward-filled from the last *observed* value and
  **never back-filled** — a back-fill would pull a future observation onto a date
  on which it was unknown. They are contemporaneous-as-of-close: a forecasting
  layer predicting `t+h` must apply its own lag.
* **`earnings_flag` is the one deliberate exception** and is tagged
  `known_future` in the frame metadata and the manifest. Earnings dates are
  published weeks ahead, so conditioning on the schedule at the forecast horizon
  is legitimate. The anchor is the first trading day *on or after* the release
  date, which handles after-the-close announcements and weekend releases.
* **Annualization** is `variance × 252` and `vol × √252`, configurable via
  `annualization_factor`.
* **Price adjustment.** Everything uses `auto_adjust=True` so O/H/L/C are
  adjusted by the *same* factors — Garman-Klass compares `H/L` against `C/O`
  within one bar, so mixing an unadjusted high with an adjusted close would
  corrupt the estimate outright.

### VOLARE comparability

VOLARE computes its realized measures from **unadjusted** prices; everything
here comes from split/dividend-adjusted OHLC. **Our output will not match
VOLARE exactly** — the series track closely but diverge around splits and
ex-dividend dates. This is expected, not a bug. `rv5` is also a 5-minute
open-to-close measure, so its level should sit near `rv_gk`/`rv_parkinson`
rather than `rv_cc`. Use `validation.compare_with_volare(frame, volare_rv5)`
once the VOLARE loader exists; it returns a clearly-marked stub until then.

## Demo results (5y, run 2026-09-23)

1234 rows per ticker, 2021-10-21 → 2026-09-22. AAPL estimator correlations:

```
              rv_cc  rv_parkinson  rv_gk  rv_rs  rv_yz
rv_cc         1.000         0.912  0.887  0.858  0.884
rv_parkinson  0.912         1.000  0.990  0.971  0.898
rv_gk         0.887         0.990  1.000  0.994  0.910
rv_rs         0.858         0.971  0.994  1.000  0.905
rv_yz         0.884         0.898  0.910  0.905  1.000
```

High correlation throughout, the three range estimators near-identical
(0.97–0.99), and `rv_cc` the noisiest on the noise ranking — exactly what theory
predicts.

## Known edges

* `rv_cc` and `rv_yz` need `window + 1` bars (they consume a *return* series), so
  their first row can be NaN where the range estimators already have a value.
  Warm-up trimming (`dropna_rv`) keeps a row if *any* estimator has a complete
  window — dropping on *all* would empty the frame whenever `intraday_rv` is
  configured.
* Zero/missing-volume bars keep their prices; only the `volume` value is set to
  NaN, so downstream code never reads a 0 as a fact.
* A ticker with no usable history is skipped, listed in the manifest under
  `skipped_tickers`, and does not fail the run.

---

# volmodels + voleval — baseline forecasters and walk-forward evaluation

The second layer: classical baselines behind one shared interface, and a
lookahead-free walk-forward harness that scores them. **No Chronos-2 or Kronos
yet** — but the interfaces are built so they drop into the *same* harness,
metrics and significance tests with zero changes.

```bash
.venv/bin/python demo.py --years 5 --csv     # build the RV series first
.venv/bin/python backtest_demo.py            # then backtest the baselines
```

## Layout

| File | Role |
| --- | --- |
| `volmodels/base.py` | `Forecaster` ABC, `Forecast` object, empirical-residual quantiles |
| `volmodels/naive.py` | `RandomWalk`, `HistoricalMean`, `EWMA` — the floors |
| `volmodels/har.py` | `HAR` (levels) and `LogHAR` (**the benchmark**) |
| `volmodels/mem.py` | `MEM` (multiplicative error model); `AMEM` is an explicit stub |
| `voleval/backtest.py` | `BacktestConfig`, `walk_forward`, `run_backtest` |
| `voleval/metrics.py` | QLIKE (primary), MSE/MAE/RMSE, calibration, aggregation |
| `voleval/significance.py` | Diebold–Mariano, Model Confidence Set |
| `backtest_demo.py` | End-to-end CLI demo |

## The shared interface

```python
class Forecaster(ABC):
    def fit(self, history: pd.Series) -> None: ...     # history ends at origin t
    def forecast(self, h: int) -> Forecast: ...        # point + optional quantiles
    def update(self, history: pd.Series) -> None: ...  # refresh data, reuse params
```

`Forecast` carries `point`, `quantiles`, `model`, `origin` and `horizon`.
Baselines fill `quantiles` from the empirical distribution of their own
in-sample residuals (the VOLARE paper's approach) on the grid
`[0.05, 0.1, 0.2 … 0.9, 0.95]` — **chosen to match Chronos-2's default output**,
so its native quantiles land in the same calibration table with no regridding.

`update()` is what makes `refit_frequency` both fast and honest: parameters may
be stale by design, but the data a forecast conditions on is always current as
of the origin.

**To add Chronos-2:** subclass `Forecaster`, append it to a factory list, run
the same `run_backtest`. Nothing else moves.

## Lookahead discipline — two rules, not one

**Model inputs never see the future.** At origin `t` the slice passed to
`fit`/`update` ends at `t` inclusive. Two independent tests enforce it:

- `test_fit_never_sees_the_future` — a spy model records every date the harness
  shows it, under every window/refit protocol, and asserts none post-dates `t`.
- `test_poisoning_the_future_cannot_change_past_forecasts` — corrupts the tail of
  the series and asserts earlier forecasts stay bit-identical. This catches leaks
  the spy cannot, because it tests *consequences* rather than inputs.

**The target legitimately does.** The target for origin `t`, horizon `h` is the
series value at `t+h`. That is future data by definition — it is what makes the
exercise a forecast. The lookahead rule constrains inputs, not targets.

## Metrics

**QLIKE is primary**: `(r/f) − ln(r/f) − 1` on *variances*. Realized volatility is
an estimate, not an observable, and Patton (2011) shows only a narrow class of
losses ranks models consistently against a noisy proxy — QLIKE and MSE are the
two common members. QLIKE is scale-free and punishes under-prediction harder than
over-prediction, matching how a vol forecast is actually used.

Aggregated **both** ways per horizon — pooled (every observation equal) and
equal-weighted across assets — with `*_ratio` columns versus Log-HAR, where
**< 1 beats the benchmark**.

## Demo results (AAPL/AMZN/MSFT, `rv_gk`, 2021-10 → 2026-09, ~980 origins each)

Pooled QLIKE, ratio vs Log-HAR:

| model | h=1 | h=5 | **h=21** |
| --- | --- | --- | --- |
| `log_har` | **1.000** | **1.000** | 1.000 |
| `har` | 1.008 | 1.028 | 1.036 |
| `random_walk` | 1.091 | 1.069 | **0.998** |
| `ewma_0.94` | 19.7 | 3.19 | 1.112 |
| `historical_mean` | 78.6 | 8.57 | 1.441 |

**The headline finding is at h=21.** Log-HAR wins convincingly at h=1 and h=5 —
it is the *sole* survivor of the Model Confidence Set at both. At h=21 it ties the
random walk (DM p = 0.98), and the MCS survivor set widens to
`{random_walk, log_har, ewma_0.94}`.

That is not a defect in the baselines; it is the overlapping-window artefact
showing up exactly where predicted. On a 21-day rolling RV series, `RV_{t+1}`
shares 20 of its 21 days with `RV_t`, so short horizons reward persistence for
reasons unrelated to skill. **h ≥ the RV window is the only honest comparison,
and there the classical benchmark has no edge over a martingale.** The
forecast-vs-realized plot shows why: Log-HAR's 21-day-ahead forecast tracks the
realized series with roughly a one-month lag, most visibly around the April 2025
spike.

Two further reads:

- `ewma_0.94`'s catastrophic h=1 ratio (19.7) is a target-shape artefact, not a
  bug: applying a slow EWMA to an already 21-day-averaged series double-smooths
  it and leaves it badly lagged.
- **Calibration is mediocre and skewed.** Nominal 5%/50%/95% quantiles realise at
  roughly 13%/62%/96% for Log-HAR at h=21 — the intervals sit too high and too
  wide. Empirical-residual bands assume the residual distribution is stable
  across regimes, which volatility violates. This is the concrete bar Chronos-2's
  native quantiles have to beat.

## Significance

- **Diebold–Mariano** on QLIKE differentials, with a HAC variance truncated at
  `h−1` lags (overlapping multi-step errors are MA(h−1)) and the
  Harvey–Leybourne–Newbold small-sample correction against `t(n−1)`.
- **Model Confidence Set** (Hansen, Lunde & Nason 2011), range-statistic form
  with a circular block bootstrap. Meaningful now, essential once several
  models are registered — it answers "which models can we not rule out as best?"
  without the false-positive inflation of many pairwise tests.

⚠️ **Read the pooled tests with caution.** Both assume the loss differentials are
one well-behaved series. Pooling three mega-cap tech tickers violates that —
their log-RV correlates 0.71–0.85, so the effective sample is far smaller than
the ~2900 rows suggest. `per_ticker_dm` gives the more defensible claim.

## Known edges

- `HAR` in levels can predict negative volatility in calm regimes; those are
  floored at `1e-8` and counted in `Forecaster.n_floored` so it surfaces as a
  finding rather than a silent repair. On the AAPL/AMZN/MSFT sample it never
  triggered (0 floored across 14,760 forecasts) — the guard is there for
  lower-volatility assets, and the levels-vs-logs case rests on the QLIKE
  numbers, not on this.
- Log-HAR's point forecast carries the Jensen bias correction
  (`exp(mu + sigma²/2)`); its **quantiles deliberately do not**, since `exp` is
  monotone and the alpha-quantile maps exactly. Expect `q0.5 < point` — correct
  for a right-skewed predictive law, not a bug.
- `AMEM` raises `NotImplementedError`: the leverage term needs signed returns,
  which the RV-series-only interface deliberately does not carry (that is what
  keeps the harness data-source-agnostic). Widening the interface is a design
  decision to make deliberately, not to smuggle in.
- A model that fails at some origin records `NaN` and the run continues; failures
  are counted in `BacktestResult.diagnostics`.

---

# VOLARE integration + Brini (2026) replication

The rigorous historical core. This layer swaps the data source from our own
yfinance daily-OHLC estimators to **VOLARE**'s true 5-minute realized measures,
and aligns the evaluation protocol with **Brini (2026)** so our numbers — and
Chronos-2's later — are directly comparable to a published benchmark.

**Reference:** Alessio Brini, *Forecasting Realized Volatility with Time Series
Foundation Models: A Comparison with Econometric Benchmarks*,
[arXiv:2607.05291](https://arxiv.org/abs/2607.05291) (6 July 2026). 50 assets
from VOLARE; nine zero-shot foundation models against eight econometric
specifications. Headline result: **only Tiny Time Mixers beats Log-HAR at every
horizon, and only by ~1–2%**. No code or data repository is linked from the
arXiv page, so `voleval/brini.py` encodes the paper's published tables as
reference constants and compares our re-run against them directly.

## ⚠️ You need to download VOLARE data

Nothing under `~` currently holds a VOLARE download, so the replication has
**not been run on real data**. Everything below is built and tested end-to-end
against a synthetic VOLARE-shaped tree; swapping in the real files is a
one-command change:

```bash
# 1. download bulk data from https://volare.unime.it into e.g. ~/volare
# 2. ingest it into our schema (one-off)
python brini_replication.py --volare-root ~/volare --ingest-only
# 3. run the replication
python brini_replication.py --input-dir output/volare
```

## `volpipe/volare.py` — the loader

Auto-detects four on-disk layouts, so whatever shape the bulk download arrives
in should just work:

```
<root>/<SYMBOL>/YYYY_MM_DD.parquet            # VOLARE native, per-day files
<root>/<asset_class>/<SYMBOL>/YYYY_MM_DD.parquet
<root>/<SYMBOL>.parquet | <SYMBOL>.csv        # one file per asset
<root>/<any>.parquet with a symbol column     # one combined file
```

VOLARE's columns are mapped onto **exactly the schema `volpipe` emits**, so the
harness cannot tell the two sources apart:

| VOLARE | ours | meaning |
| --- | --- | --- |
| `rv5` | `var_rv5` + `rv_rv5` | 5-min realized variance → our primary target `sqrt(rv5)` |
| `bv5` | `var_bv`, `jump` | bipower variation; `jump = max(rv5 − bv, 0)` for HAR-J |
| `rq5` | `var_rq` | realized quarticity, for HARQ |
| `rsp5` / `rsn5` | `var_rsp` / `var_rsn` | realized semivariances, for HAR-RS |
| `medrv5`, `minrv5`, `rk`, `rr5`, `rv1`, `rv5_ss` | `var_*` / `rv_*` | carried through |
| `open/high/low/close/volume/trades` | same | carried through |

Column matching goes through an alias table (`MEASURE_ALIASES`) rather than
hard-coded names, since the exact headers in a bulk download could not be
verified without the data. If a column is missing, the loader says which
measure it could not supply and which models that disables.

**VOLARE uses unadjusted prices**; our yfinance estimators use adjusted OHLC.
The series will not match exactly — that divergence is expected and recorded in
the ingest manifest.

## Protocol alignment — matching Brini exactly

| | Brini (2026) | ours |
| --- | --- | --- |
| Horizons | h = 1, 5, **22** | same (`BRINI_HORIZONS`) |
| Window | **rolling**, 1,000 days | same (`brini_protocol()`) |
| Re-estimation | daily (`refit_frequency=1`) | same by default |
| Target | point-in-time `σ_{t+h} = √RV_{t+h}` | same — our harness already defined it this way |
| Loss | QLIKE on variances | same |
| Aggregation | pooled **and** per-asset equal-weighted | same |
| Equities | 40 US large caps, 2015-01-02 → 2026-01-30 | `BRINI_EQUITIES` (all 40) |
| FX / futures | 5 pairs + 5 contracts | `BRINI_FX`, `BRINI_FUTURES` |

**A note that resolves last session's finding.** Brini forecasts the
*point-in-time* value at `t+h`, explicitly because "overlaps induce serial
correlation in the target". That is the same artefact we independently hit on
the 21-day rolling RV series — so the paper's design already avoids it, and our
earlier concern is not a criticism of this benchmark but a confirmation of its
target choice.

## Two tracks, deliberately separate

`brini_replication.py --track {brini,aggregate}`

- **`brini`** — point-in-time `σ_{t+h}`. The only track comparable to the paper.
- **`aggregate`** — `sqrt(mean(RV_{t+1..t+h}))`, the volatility of the *whole*
  horizon. This is the economically relevant quantity for an h-day risk or
  option horizon and is our own preferred target, but its targets overlap across
  origins, so its numbers are **never** merged into the comparison table. The
  runner refuses to print the Brini comparison on this track.

## Models — all eight of Brini's econometric benchmarks

| Brini | ours | needs exog |
| --- | --- | --- |
| HAR | `har` | — |
| Log-HAR | `log_har` (benchmark) | — |
| HAR-J | `har_j` — adds `jump = max(RV − BV, 0)` | ✅ |
| HAR-RS | `har_rs` — splits daily term into `RS+`/`RS−` | ✅ |
| HARQ | `harq` — daily coefficient scaled by `√RQ` | ✅ |
| ARFIMA | `arfima` — GPH-estimated `d`, fractional differencing | — |
| ARMA | `arma` | — |
| MEM | `mem` | — |

**None of Brini's econometric benchmarks is omitted.** The nine foundation
models are out of scope for this step by instruction.

The HAR-family variants estimate in **variance** space (their extra regressors
are variances; mixing them with a volatility target would be dimensionally
incoherent) and return volatility, which QLIKE squares straight back.

### Interface change: the exogenous channel

`Forecaster.fit(history, exog=None)` and `update(history, exog=None)`. This
resolves the design decision flagged last session (the one that blocked AMEM):
models can now consume companion measures. The channel is **subject to the same
no-lookahead rule** — the harness reindexes `exog` onto the training slice, and
`Forecaster._store_history` independently raises if exog reaches past the
origin. Two tests cover it, including a tail-poison check on the VOLARE path.

This is also the channel Chronos-2's covariate support will use.

## Evaluation additions

- **Mincer–Zarnowitz** (`mincer_zarnowitz`) — regress realized on forecast;
  `α=0, β=1` means efficient. This separates *information* (higher R²) from mere
  *calibration* (β nearer 1), which is precisely the split Brini uses to argue
  the foundation models' edge is largely the latter. Reported per-asset then
  averaged, matching the paper's Table 9.
- **Per-asset MCS inclusion rates** (`mcs_inclusion_rates`) — Brini's Table 8
  protocol: run the MCS per asset, report the % of assets where each model
  survives. Avoids the cross-sectional-independence assumption a pooled MCS
  makes over correlated equities.
- **Pre/post-COVID subsamples** (`subsample_summaries`, boundary 2020-03-01).

## The success criterion

Reproducing a paper's *absolute* numbers from its description alone is hard;
preprocessing, window edges and library differences all move the third decimal.
**The defensible claim is a faithful protocol replication whose relative results
match** — loss ratios against Log-HAR, the ordering of the benchmarks, and MCS
membership. `comparison_vs_brini.csv` reports our value, the paper's value and
the gap for every (model, horizon) cell.

QLIKE helps here: it depends only on the realized/forecast **ratio**, so it is
invariant to annualisation and any common rescaling. Our QLIKE levels are
therefore comparable to Brini's regardless of unit conventions. The
Mincer–Zarnowitz **intercept** is not scale-free, which is why the VOLARE loader
defaults to `annualization_factor=1` (daily units, as the paper reports).

Targets to hit on equities (paper's Table 4 / Table 6):

| | h=1 | h=5 | h=22 |
| --- | --- | --- | --- |
| Log-HAR QLIKE | 0.198 | 0.301 | 0.538 |
| HAR ratio vs Log-HAR | 0.998 | 1.009 | 1.070 |
| ARMA ratio | 1.000 | 1.034 | 1.112 |
| MEM ratio | 1.014 | 1.030 | 1.107 |
| HARQ ratio (the fragile one) | 5.132 | 3.518 | 1.329 |
| *TTM (best foundation model)* | *0.982* | *0.986* | *0.987* |

## Runtime

ARMA and ARFIMA run an iterative MLE at every refit (~190 ms each). Brini's
daily re-estimation over 40 assets × ~1,764 origins is therefore an overnight
job, not an interactive one. `--refit k` is the lever; any value other than 1 is
recorded in the manifest under `deviations_from_paper`, because it *is* a
deviation.

## Dry-run status (no real VOLARE data yet)

The whole path — ingest → 8 models → metrics → DM/MCS → MZ → subsamples →
comparison table — has been exercised end-to-end against a **synthetic**
VOLARE-shaped tree (6 assets × 2,200 days, log-HAR data-generating process with
consistent jump/semivariance/quarticity columns). The machinery runs; the
numbers from it are meaningless and are not recorded as results.

What the synthetic run **cannot** tell us, and what to watch when real data
lands:

- **Absolute QLIKE levels.** On synthetic data ours come out ~60–70% below
  Brini's, purely because the simulated noise level differs. Only real `rv5`
  can test the level.
- **HARQ and HAR-RS fragility.** Brini reports these two performing badly
  (ratios of 5.13 and 2.31 at h=1). The dry run reproduced the *mechanism*: at
  1,200 origins HARQ drove a single forecast through zero, which under QLIKE
  produced a mean loss of 1.0e10 — one observation in 7,200 swamping the metric
  (excluding it, HARQ sat at 0.095, ratio ~1.07). The fix was the standard
  **insanity filter** (see below), not a clip. Whether real VOLARE quarticity
  makes HARQ merely *bad* (Brini's 5.13) rather than *degenerate* is still open.
- **ARFIMA's memory parameter.** GPH is verified correct (it recovers a known
  `d` to ±0.05 — see `test_gph_recovers_a_known_memory_parameter`), but on the
  synthetic series it pins at the stationarity bound 0.499 on every window,
  because that DGP is near-integrated. Real log-RV usually sits around d ≈ 0.4;
  if it pins at 0.499 there too, the ARFIMA results should be read with care.

### The insanity filter

The levels-space HAR family (`har`, `har_j`, `har_rs`, `harq`) carries the
Bollerslev–Patton–Quaedvlieg (2016) **insanity filter**: a point forecast that
is negative, non-finite, or above 10× the in-sample maximum is replaced by the
**in-sample unconditional mean**, and the event is counted in
`Forecaster.n_insane` and the harness diagnostics.

This is not cosmetic. QLIKE diverges as the forecast approaches zero, so
flooring an impossible forecast at `1e-8` — which is what the code did before —
lets a single observation dominate the mean loss by ten orders of magnitude.
Without the filter our HARQ numbers would not be comparable to any published
HARQ result. Log-HAR does not carry it: a log-space model cannot produce a
negative forecast in the first place.

Everything above is checked by 224 passing tests, including the lookahead
guards re-run on the VOLARE path with the exogenous channel attached, and a
regression test that reproduces the 1.0e10 failure in one assertion.
