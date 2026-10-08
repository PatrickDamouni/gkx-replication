# Decision log

One entry per decision: what I chose, what I rejected, why. Newest at the bottom.
This file is the interview script.

---

## D001 — Project choice: replicate GKX (2020) instead of a novel model
- **Chose:** Replicate + extend Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine Learning."
- **Rejected:** Building an original trading model; benchmarking TS foundation models; LLMs on filings.
- **Why:** A replication has a ground truth to check against, so mistakes are detectable.
  Not being a PhD student, the strongest signal I can send is evaluation rigor, and a
  replication puts that front and center.

## D002 — Data: CRSP via MSU WRDS + Open Source Asset Pricing signals
- **Chose:** CRSP monthly returns (WRDS, MSU access) merged with OSAP firm-level characteristics.
- **Rejected:** yfinance (survivorship bias — delisted firms drop out); OSAP portfolio-level
  returns only (can't do firm-level return prediction, which is the paper's core task).
- **Why:** Firm-level, survivorship-free returns are required to replicate the paper honestly.

## D003 — Extension: post-publication holdout (2017 → latest)
- **Chose:** Seal everything after the paper's sample and test it once, at the end.
- **Rejected:** Picking an extension after seeing replication results.
- **Why:** Choosing the extension before seeing data avoids a forking-paths problem, and
  "does it survive after publication" is a question a quant researcher actually cares about.

## D004 — Pre-registration locked before data access
- **Chose:** Lock `PREREGISTRATION.md` and tag the commit `prereg-v1` before any WRDS data is pulled.
- **Rejected:** Refining criteria after a first look at the data.
- **Why:** The tag gives a verifiable timestamp that the question, success criteria,
  abandon conditions and splits were fixed ex ante. Any change after this point is a
  logged deviation, not a silent edit.

## D005 — Prereg clarification: R²_oos is pooled over all stock-months (2026-10-07, before any data)
- **Chose:** Headline R²_oos = 1 − Σᵢₜ(r − r̂)² / Σᵢₜ r², pooled over every stock-month in the
  test window, as in GKX. Per-month R² reported only as a diagnostic.
- **Rejected:** Average of monthly cross-sectional R² (the prereg's "monthly R²" could be read this way).
- **Why:** "Monthly" in the prereg referred to return frequency; pooling matches the paper, so the
  replication comparison is like-for-like. Clarified before any model or data was run.

## D006 — Prereg clarification: seed estimand (2026-10-07, before any data)
- **Chose:** Report both: (a) the seed-ensemble (average predictions across ≥5 seeds) as the
  headline, matching GKX; (b) mean ± spread of single-seed results as a stability check.
  The best single seed is never reported.
- **Rejected:** Single-seed mean ± spread only (understates NNs vs. the paper and could flip the
  pre-committed model ranking); ensemble only (hides seed instability).
- **Why:** Keeps the replication faithful while still surfacing variance.

## D007 — Macro-characteristic interactions kept, computed in batches (2026-10-07, before any data)
- **Chose:** Keep GKX's characteristic × macro-predictor interactions, materialized per batch
  inside model fitting instead of stored as a full design matrix.
- **Rejected:** Dropping interactions (weaker replication); a full in-memory matrix
  (~1,800 features × ~4M rows ≈ 29 GB float32 > 26 GB RAM).
- **Why:** Faithful to the paper within this machine's limits. `evaluate` therefore takes
  characteristics and macro predictors separately.

## D008 — Phase 1a test design revised after a local council review
- **Chose:** Seams = load_panel (holdout sealed at load, split at ingest, holdout hash recorded) ·
  align/rank_normalize under a future-invariance property test · walk_forward Folds with a
  snapshot · metrics on per-month summaries (pooled R², DM on monthly mean loss differential via
  statsmodels HAC, shared month-block bootstrap, decile L/S with formation-month ME) · evaluate
  with zero/oracle/cheat models + a power analysis · one slow-marked scale test.
- **Rejected:** Seal inside walk_forward only (a notebook could read 2017+ directly); a planted
  leaky feature caught by an R² threshold (circular, misses realistic ~0.1% leaks); "trees beat
  OLS" in Phase 1a (tests models that don't exist, flaky at realistic SNR); a "noise ⇒ R² ≈ 0"
  assertion (wrong: fitted models score < 0 under the non-demeaned metric).
- **Why:** Tests should sit where leakage actually happens. Record:
  `.claude/council-cache/local-council-1791409997.md` (local, all-Claude reviewers — weak
  corroboration, used for angle coverage).

## D009 — OSAP release 202510 ingested and sealed (2026-10-07)
- **Did:** Downloaded OSAP firm characteristics (release 202510, 209 signals, 5,416,424 rows),
  converted to Parquet, split at ingest, then deleted the full zip and full Parquet so no
  unsealed copy remains on disk.
  - source zip sha256 `aae101c1fd4749c5a68d83734b6ce5e90a0f2f18b456f498e6f3b803081f6ee3`
  - pre-2017 file: 4,482,749 rows, 1925-12 → 2016-12, 31,770 permnos
  - **holdout file sha256 `8dd2dd29b6d54856f8e9612c8dbd9ad3f2015ca86f0060f8d825374e10e286cd`** —
    not opened until Phase 5; this hash proves it is unchanged when it is.
- **Timing convention:** an OSAP signal dated t uses information through end of t and predicts
  the return in t+1. Verified from openassetpricing 0.0.2's own code (STreversal at t = −ret in t),
  pinned in `tests/test_align.py`.
- **Rejected:** Keeping the zip "just in case" (an unsealed copy defeats the seal; it is
  re-downloadable and verifiable by the hash above).

## D010 — Significance tests: Clark–West for nested pairs, paired ΔR² bootstrap for the ranking (2026-10-08, before any returns data)
- **Chose:** Clark–West (2007) for nested comparisons (any model vs. zero forecast; OLS-3 vs.
  OLS-all); Diebold–Mariano for non-nested pairs; tests run on per-month *sums* so they weight
  months the way pooled R² does. The pre-committed ranking (NN/trees > ENet > OLS-all) is tested
  with a paired month-block bootstrap CI on ΔR². HAC lag and block length to be fixed in the
  seam-4 commit, before data.
- **Rejected:** DM everywhere (undersized under the null for nested models).
- **Why:** The prereg's success criterion is a ranking plus "beats baseline"; these are the tests
  that answer those questions correctly. Source: local council 2.

## D011 — Linear baselines reported with both squared and Huber loss (2026-10-08)
- **Chose:** OLS-3, OLS-all and ENet each run with squared loss and with Huber loss (GKX's "+H").
- **Rejected:** Squared loss only (inflates the ML-vs-linear gap relative to GKX); Huber only.
- **Why:** Matches the paper and shows how much the loss choice matters. Both variants count as
  trials for the deflated Sharpe ratio.

## D012 — Tree models get raw characteristics + macro predictors, not explicit interactions (2026-10-08)
- **Chose:** RF/GBRT see characteristics and macro predictors as separate inputs.
- **Rejected:** Materialized char × macro interactions (~1,950 cols × ~4M rows ≈ 32 GB float32;
  sklearn histogram GBM casts to float64); row subsampling.
- **Why:** Trees learn interactions through splits; this is a stated deviation from GKX's inputs.

## D013 — Feature set mapped to GKX's ~94 characteristics (2026-10-08)
- **Chose:** Map OSAP signals to GKX's characteristic list; document each match and each
  characteristic without an OSAP counterpart in `docs/feature_map.md`.
- **Rejected:** All 209 OSAP signals (less like-for-like, ~2× memory). Possible later extension.
- **Why:** Fidelity to the paper and half the width.

## D014 — Panel keyed on return month; models never receive the label (2026-10-08)
- **Bug found by council 2 (reproduced by 4 of 5 members):** `Fold` windows are GKX return months,
  but the aligned panel was keyed on signal month, so the Dec-2016 signal needed a sealed
  Jan-2017 return and would silently drop; and a model reading its own row's label passed the
  future-invariance check.
- **Fix:** `align` keys rows on the return month (`yyyymm`), keeps `signal_month` as metadata, and
  drops rows whose return month is in the holdout. `evaluate` (seam 5) passes models only an
  explicit feature list, so the label is structurally unreachable.

## D015 — Inference settings fixed before data (2026-10-08)
- **HAC lag:** Newey–West rule floor(4·(T/100)^(2/9)); T = 360 test months → **5 lags**. Applied to
  every DM / Clark–West test on monthly series.
- **Bootstrap:** circular month-block bootstrap, **block length 12**, **B = 10,000**, seed 20261008.
  Resamples whole months (keeps each cross-section intact); one shared index matrix for every
  model so CIs on ΔR² and Sharpe are paired.
- **Test sides:** DM / Clark–West are one-sided (H1: the larger/candidate model has lower MSE),
  normal p-values. CIs are percentile, 95%.
- **Sharpe:** monthly mean / monthly sd × √12.
- **Why now:** each is a forking path if chosen after seeing results.

## D016 — Evidence for D010: DM vs Clark–West size under a nested null (2026-10-08)
- Simulation (400 reps, 20 firms, big model = small model + fitted noise, true null, nominal 5%):
  - T = 60 months: DM rejects 0.5%, Clark–West 8.5% (mildly oversized at short T).
  - **T = 360 months (our test length): DM rejects 0.0%, Clark–West 5.8%** (MC se 1.1%).
- **Reading:** plain DM would almost never declare a nested model better even when it is; CW is
  correctly sized at our sample length. Reproduce with the loop in `tests/test_metrics.py`.

## D017 — Power analysis result and the ranking test it implies (2026-10-08, before any returns data)
- **Result** (`results/power_analysis.md`, oracle upper bound, T=360, N=6000, factor + fat tails):
  detectability of a model gap depends on how *stable* the incremental signal's payoff is
  (its long-short Sharpe), far more than on the size of ΔR².
  - Clark–West power: 84–92% at signal Sharpe 0.5; 100% at Sharpe ≥ 1 (all ΔR² 0.1–0.4%).
  - Paired ΔR² 95% CI excluding 0: 26–40% at Sharpe 0.5; 74–76% at Sharpe 1; 100% at Sharpe 2.
  - A first version with a constant premium gave 100% everywhere — uninformative, discarded and
    noted in the script docstring.
- **Chose (ex-ante deviation from PREREGISTRATION.md's ranking wording):** the ranking is tested
  with Clark–West (nested pairs) / Diebold–Mariano (non-nested), one-sided 5%. The paired ΔR² CI
  is reported as the effect size, not as the pass/fail criterion. A non-significant gap is
  reported as "not distinguishable", not "replication failed".
- **Rejected:** ΔR² CI as the criterion (high chance of a false "no difference"); requiring both.
- **Caveat:** fitted models carry estimation error, so real power is lower than shown. Rerun this
  script with real CRSP residual volatility once data arrives.

## D018 — Memory budget measured on the real data path (2026-10-08)
- `scripts/measure_memory.py`, fresh process, 94 signal columns (D013 width): load 3.51 GB →
  rank_normalize 5.42 GB peak (46 s, run once) → largest train window (3,342,721 rows) as float32
  adds nothing measurable. Fits comfortably in 26 GB.
- Remaining pressure is the char × macro expansion for linear/NN models: 94 × 9 = 846 columns ×
  3.34M rows ≈ 11 GB float32 if materialized — handled by batching inside those models (D007);
  trees use raw inputs (D012).
- Replaces the planned in-pytest "scale test" (council 2: in-process `ru_maxrss` is a lifetime
  high-water mark and a synthetic refit of nonexistent models measures the wrong thing).

## D019 — CRSP source: legacy monthly file + explicit delisting returns (2026-10-08)
- **Chose:** `crsp.msf` (legacy SIZ, 1925-12 → 2024-12) merged with `crsp.msedelist`, universe from
  `crsp.msenames` (shrcd 10/11, exchcd 1/2/3 as of each month), rf from `ff.factors_monthly`
  (monthly decimal). Pulled with `date < '2017-01-01'` in the SQL, so no holdout return reaches disk.
- **Rejected:** `crsp.msf_v2` (CIZ). Checked on 2,760 pre-2017 delisting months: its `mthret` equals
  the compounded (1+ret)(1+dlret)−1 in 84% of cases, but for performance delistings flagged `M`
  it keeps only the regular return and drops DLRET (e.g. permno 11460, code 560: DLRET −57%,
  CIZ return +5%). Dropping those losses biases small-cap returns upward (Shumway 1997).
- **Delisting rule:** return in the delisting month = (1+ret)(1+dlret)−1; if ret is missing, dlret
  alone. If dlret is missing and the code is performance-related (500, 520–584): impute −30% on
  NYSE/AMEX, −55% on NASDAQ (Shumway 1997; Shumway & Warther 1999). Affects ~1% of performance
  delistings (e.g. code 560: 15 of 1,407 missing).
- **Value weights:** ME = |prc| × shrout / 1000 ($M); `me_lag` = ME in the signal month (t−1),
  matched by month key, NaN across gaps.
- **Cost:** legacy data ends 2024-12, so the Phase 5 holdout is 2017-01 → 2024-12 (8 years).

## D020 — Universe: every security on NYSE/AMEX/NASDAQ, as in GKX; common shares as robustness (2026-10-08)
- **Evidence:** with the exchange filter only, 1957-03..2016-12 contains 29,943 permnos (GKX: "almost
  30,000"); a later paper reusing GKX's data says no further filters were imposed. Common shares
  only (shrcd 10/11) gives ~24,400. Mean securities/month: 5,216 all vs 4,303 common-only. (An
  earlier recollection of "6,200 per month" could not be verified and is not relied on.)
- **Chose:** main run = exchange filter only (exchcd 1/2/3 as of each month); `shrcd` kept on every
  row; a second run restricted to shrcd 10/11 checks that ADRs/REITs/funds don't drive results.
- **Supersedes** the shrcd 10/11 filter in D019.
- **Validation of the CRSP build (common-only version, before this change):** value-weighted
  excess return vs `ff.factors_monthly.mktrf`, 1957-03..2016-12, 718 months: corr 1.0000, mean
  difference −0.002%/month, tracking sd 0.018%/month.

## D021 — Macro predictors: Welch-Goyal updated file, sealed at ingest (2026-10-08)
- **Source:** Amit Goyal's "Updated Welch-Goyal data (through 2025)" (sites.google.com/view/agoyal145),
  Monthly sheet, 1871-01 → 2025-12. Source xlsx sha256
  `8d210b779442a5a9206702060aac67c1de4283186c0f167ebdb4488d149be872`. Goyal notes that from 2022 lty
  comes from FRED and ltr/corpr from Bloomberg indices — affects only the sealed holdout years.
- **Definitions (GKX §2, Welch & Goyal 2008):** dp = log D12 − log Index; ep = log E12 − log Index;
  bm; ntis; tbl; tms = lty − tbl; dfy = BAA − AAA; svar.
- **Timing:** value at month t joins on the signal month t (known by end of t).
- **Sealed:** split at ingest; **holdout sha256 `4d2640cc569645deaa94cab27bd51ac32e3469eb72371660a3fd07110360448e`**
  in read-only `data/sealed/`; xlsx and full Parquet deleted. Pre-2017 file: 718 signal months
  1957-02 → 2016-11 with no missing values.
