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

## D022 — Characteristics: GKX's own datashare primary; OSAP as an independent-rebuild check (2026-10-08)
- **Revises D013.** Dacheng Xiu (GKX co-author) publishes the paper's 94-characteristic dataset
  (`datashare.zip`, dachxiu.chicagobooth.edu, research use only, last modified 2026-02-26).
- **Chose:** replication runs on datashare characteristics (exact GKX inputs), so any gap to the
  paper isolates method differences from data-build differences. The same models are rerun on
  OSAP signals mapped to GKX's list as a robustness check on an independent build; OSAP also
  supplies characteristics for any holdout years datashare does not cover.
- **Rejected:** OSAP-only (confounds data and method differences); datashare-only (no
  independent-build check, possible holdout coverage gap).
- Datashare is streamed to Parquet without unzipping and split at ingest like every other source.

## D023 — datashare timing: DATE is the return month; stored as signal month; verified on CRSP (2026-10-08)
- **Source:** `datashare.zip` sha256 `5de63312a9b4d8e340c3b69c2d3f2682cd834705aff695922227e6f93ce4bbf3`
  (readme "This Version: April 2019"); 4,117,300 rows, 94 characteristics + sic2.
  **Holdout sha256 `9a87fbf7153c2a4d6d61ac93196df24f59a6f4d590566d2e780a0965e5a1881b`**, read-only in
  `data/sealed/`; zip deleted.
- **Convention:** the readme says DATE=19570329 → response RET at 195703, i.e. rows are keyed on the
  return month with characteristics pre-lagged (monthly ≥1, quarterly ≥4, annual ≥6 months). We
  store `yyyymm` = DATE month − 1 (signal month) so datashare and OSAP share one convention and both
  go through `align` (test: `tests/test_datashare.py`).
- **Empirical check (pre-2017 only):** mom1m at signal month s vs CRSP ret at s: corr 0.936, equal
  within 1e-4 in 98.0% of 3,734,245 rows (residual consistent with our delisting adjustment); vs CRSP
  ret at s+1: corr −0.009 → no lookahead in the conversion or the source. mvel1(s) / CRSP ME(s) =
  1.0000 (median). 29,946 permnos ≈ the all-securities universe (D020: 29,943).

## D024 — Characteristic mapping rule: concept gates, measurement validates (2026-10-08, before any return-based fit)
- **Supersedes** the argmax-|ρ| draft (commit 2eb612a), which local council 3 showed produces
  false matches: tail-only binary signals (MomRev, AccrualsBM overlap 7–9% of rows → inflated ρ),
  a Predictor-only search space, and a ~4-month quarterly timing offset.
- **Priority per GKX characteristic:**
  1. Exact rebuild from CRSP (mvel1, mom1m, mom6m, mom12m, mom36m, chmom), accepted only if it
     agrees with datashare at ρ ≥ 0.99 and ≥ 95% equal values.
  2. Else an OSAP Predictor whose *documented definition* (OSAP `Detailed Definition` vs GHZ 2017
     appendix) measures the same quantity; differences allowed only in horizon, frequency,
     filters or industry adjustment. Written concept note required.
  3. Else missing (imputed like any missing characteristic). OSAP Placebos are not published
     (checked: openassetpricing 0.0.2 and openassetpricing.com/data offer only the 209
     Predictors), so Placebo-only characteristics are missing.
- **Validation of an accepted concept match:** timing shift fixed by frequency from datashare's
  readme, not by search (annual 0; quarterly +4 months); mean monthly Spearman ρ ≥ 0.80 =
  validated, 0.60–0.80 = labelled proxy, < 0.60 = missing; OSAP coverage ≥ 70% of datashare's
  non-missing rows; ≥ 5 distinct values; each OSAP column used at most once; sign taken from
  definitions, never from measured ρ.
- **Representation:** a committed, reviewed `config/feature_map.csv` enforced by tests; the
  measurement script only writes evidence (`results/feature_map_measured.csv`).
- **Comparison:** the OSAP run uses the accepted subset S, and datashare is rerun on the same S,
  so "data build" is not confounded with "fewer features".
- **Disclosure:** the concept review happened after seeing the draft correlations. No return
  data has been used with the map; the rule is mechanical and fixed here.

## D025 — Holdout: two arms, sources never mixed; and a stated deviation from the prereg (2026-10-08)
- **Arm A (primary):** trained and tested on datashare characteristics, holdout = 2017-01 →
  datashare's last month (≈2021 by row count; confirmed only at unseal).
- **Arm B:** trained and tested on the OSAP-mapped subset S (D024), holdout 2017-01 → 2024-12.
- On overlapping holdout years both arms are reported; the gap measures the data-source effect.
  A model is never trained on one source and tested on the other.
- **Deviation:** PREREGISTRATION.md's question 1 says "independently built data"; D022 made the
  authors' own data primary, so the independent build (Arm B / OSAP) becomes the robustness
  check. Made before any return-based results; disclosed here and to be stated in the write-up.

## D026 — Characteristic map frozen (tag `map-v1`), before any return-based fit (2026-10-08)
- **Result** (`results/feature_map_measured.md`): of 94 GKX characteristics, **35 validated** (6 exact
  CRSP rebuilds + 29 OSAP), **7 labelled proxies** (beta, betasq, acc, chatoia, tb, roeq, baspread),
  **52 missing** (27 with no published counterpart — mostly OSAP Placebos; 25 failed validation).
  Arm B (D025) therefore uses the 42 validated + proxy characteristics, and Arm A is rerun on the
  same 42 for the like-for-like comparison.
- **Applied as written, including against intuition:** ep (ρ 0.90, coverage 67%), operprof (ρ 0.96,
  45%), orgcap, tang, ps, sin, rd fail D024's 70% coverage bar because of OSAP's own filters
  (e.g. excluding negative earnings, manufacturing-only); they are missing, not re-thresholded.
  ear fails at the fixed +4 quarterly shift (council 3 saw a peak at +5; shift is not tuned per
  signal).
- **Post-hoc, disclosed:** `agr` measured ρ = −0.998 with OSAP AssetGrowth. Distribution check
  (not ρ): datashare agr median −0.092 with 80% negative; OSAP asset growth median +0.088 → the
  datashare stores agr negated (GHZ code orientation). Declared transform `negate`; status becomes
  validated (ρ 0.998). This is the only change made after seeing the measured table.
- **Interpretation of D024's distinct-values rule:** it applies when the GKX characteristic is
  non-binary (its purpose was to exclude tail-only binary OSAP signals); binary GKX characteristics
  (convind, divi, divo, rd, sin) are validated by ρ (phi) and coverage.
- Every mapped characteristic is rank-normalized within the universe each month (build_panel), so
  orientation affects interpretation, not model fit.

## D027 — The 1987–2016 test window is touched once, for all models together (2026-10-08)
- **Chose (council 4: Devil's Advocate + Performance):** every model — baselines included — is
  developed and debugged on a rehearsal walk-forward whose test years are 1975–1986
  (`walk_forward(first_test_year=1975, last_test_year=1986)`). When harness, numerics, grids,
  seeds and every model family's code are frozen (git tag), all models are run on 1987–2016 in a
  single pass. `evaluate()` refuses any fold with test months ≥ 1987-01 unless explicitly unlocked
  from a clean git tree.
- **Rejected (council 4: Simplicity):** baselines on the test window as soon as bugs are fixed —
  earlier feedback, but test-window numbers could shape the panel and later model choices.
- **Cost:** no real test-window numbers until Phase 3 is complete.

## D028 — What counts as a trial for the deflated Sharpe ratio (2026-10-08)
- **Counts:** every configuration × seed of the candidate models in Arm A (94 characteristics).
- **Reported but not counted:** zero forecast (a column of zeros, not a model run), OLS baselines,
  and robustness reruns (Arm A-42, Arm B, common-shares only). The ledger records a
  `counts_as_trial` flag per run.

## D029 — Model scope fixed before any test-window look (2026-10-08)
- **In:** linear family (OLS, OLS-3, ENet, PCR, PLS, GLM) each with squared and Huber loss where
  GKX use Huber; gradient-boosted trees and random forest via LightGBM; NN1–NN3 with 5 seeds each.
- **Out:** NN4, NN5 (GKX find no gain beyond NN3); exact-split sklearn RF (compute).
- **Why:** council 4 Performance estimated the full set at ~150–300 laptop hours; this scope is
  ~40–80 h. The linear family uses exact Kronecker sufficient statistics
  (X'X = Σ_t (x_t x_t') ⊗ (C_t'C_t)), so it never materializes 920 columns.
- Grids, early-stopping rules and library versions are fixed in the Phase 3 plan before the test look.

## D030 — Reference values and procedure facts from the paper, fixed before any model run (2026-10-08)
- **Source:** Gu, Kelly & Xiu, NBER Working Paper 25398 (Dec 2018, revised Sep 2019), read from the
  PDF text (not memory). Table 1 (monthly R²_oos, %) is committed verbatim as
  `paper/gkx_table1.csv` and tagged `reference-v1` before any model is fit to returns.
- **Facts used:**
  - Table 1's OLS, OLS-3, ENet, GLM and GBRT columns are Huber versions ("+H"). Our squared-loss
    OLS/OLS-3 (D011) are reported without a paper comparison.
  - Huber ξ: Table A.5, "ξ = 99.9% quantile" for OLS-3+H, ENet+H, GLM+H, GBRT+H.
  - Splits: train 1957–1974 expanding by one year per refit; validation 12 years rolling; test
    1987–2016; refit annually; no cross-validation (matches seam 3).
  - Top/bottom 1,000: by market equity, each month.
  - Characteristics ranked to [−1, 1] each month; missing → cross-sectional median (footnote 29;
    matches build_panel).
- **Interpretations where the paper is silent (fixed now, not tuned):**
  - ξ basis: the 99.9th percentile of |residual| from the squared-loss fit on the same training
    window (OLS-all ξ likewise, since Table 1 reports OLS+H).
  - Models without tuning parameters (OLS, OLS-3) are fit on the training window only; the paper
    describes no merge of validation into training.
  - Top/bottom 1,000 ranked on ME at the signal month (`me_lag`).
  - Macro predictors are not transformed in the paper. For the linear family we center/scale them
    with training-window constants before forming x ⊗ c: with the constant included, the span of
    (1, x)⊗c is unchanged, so OLS and Huber predictions are mathematically identical and only the
    numerics improve (council 4: cond(X'X) 5.3e10 raw → 1.1e6 scaled). For penalized models and NNs
    scaling changes the fitted model; that choice is made in the Phase 3 plan.

## D031 — Linear solves use the minimum-norm solution (found in the 1975–1986 rehearsal) (2026-10-08)
- **Finding (rehearsal, D027 — not the test window):** OLS-all scored R²_oos −28,579% over
  1975–1986. Diagnosis on fold 1975: training is 1957-03 → 1962-12 (70 months); X'X has rank
  534/907 (cond 5.2e51) because many GKX characteristics do not exist yet and are median-filled to
  0; in-sample R² is a sane 3.0%; 1975 macro values sit 3–10 training SDs outside the training
  range (m_dfy 10.0σ, m_bm 9.3σ). Cholesky and lstsq gave −71,086% and −57,445% on that fold:
  the explosion is genuine extrapolation, made arbitrary by the rank deficiency.
- **Chose:** solve the normal equations by eigendecomposition and keep eigenvalues above
  1e-12 × the largest (minimum-norm / pseudo-inverse solution). Unidentified directions get
  exactly zero weight (tests: `test_unidentified_directions_get_zero_weight_minimum_norm`,
  `test_minimum_norm_solution_matches_pinv_on_the_materialized_design`). `rank_` is recorded.
- **Rejected:** Cholesky (arbitrary coefficients on singular designs); ridge (changes the model —
  GKX's OLS is unpenalized); trimming characteristics by availability (changes GKX's feature set).
- **Expectation stated before the test look:** rehearsal training windows (6–17 years) are far
  shorter than the test-window ones (≥30 years), so rehearsal OLS-all is not comparable to GKX's
  −3.46%; a large negative OLS-all R² remains expected for genuine reasons (GKX §3.2).

## D032 — Huber IRLS: fixed identified subspace, achievable tolerance, outlier-only updates (2026-10-09)
- **Finding (rehearsal):** IRLS hit its 500-iteration cap on most folds, so the first OLS+H rehearsal
  number (−27,038%, 40 min) was not a converged estimate. Two causes: (1) D031's minimum-norm solve
  re-decided the rank every iteration, so the solution could flip; (2) the 1e-12 relative tolerance
  sat below float64 noise — on fold 1986 the Huber objective was identical to 15 digits from
  ~iteration 10 while coefficients kept moving ~1e-13.
- **Chose:** identify the subspace once from the unweighted X'X (with all weights > 0, X'WX has the
  same null space — exact, not an approximation); stop at 1e-9 relative coefficient change; update
  X'WX as X'X − Σ_downweighted (1−w) z z' (≈0.1% of rows). Rehearsal folds now converge in 5–9
  iterations, ≤ 1.2 s per fold. Tests: convergence on a rank-deficient design, first-order optimality
  (‖Z'ψ‖ ≤ 1e-8 scale), fast path identical to full rebuild.

## D033 — Rehearsal diagnosis of OLS-all: the explosion is macro-interaction extrapolation (2026-10-09)
- Rehearsal 1975–1986 (never the test window): OLS with x ⊗ c + dummies R² −27,153%; the same OLS on
  characteristics + industry dummies only R² −1.58% (most years +0.5% to +1.7%; 1975, trained on 6
  years, −17%). Fast and slow Huber agree to 8.9e-9 relative.
- **Reading:** the characteristic pipeline is sound; the explosion is unpenalized macro interactions
  extrapolating (D031: 1975 macro 3–10 training SDs out of range). GKX's OLS-all is specified exactly
  this way and is reported as failing (−3.46%).
- **Decision:** no change to OLS-all's specification to make it look better (no winsorizing or
  clipping macro). The test-window OLS-all number will be reported as is, with this diagnosis.

## D034 — Deflated-Sharpe trial count revised: N = model variants whose test Sharpe is reported (2026-10-09, before any test look)
- **Revises D028** (council 5: Simplicity, Devil's Advocate). Grid points are selected on the
  validation window and seeds are averaged into the ensemble (D006) — neither is a selection on
  test results. N = the number of Arm A candidate model variants whose test Sharpe is reported and
  compared (≈11). Every grid point and seed is still logged (val scores), and the ledger keeps the
  `counts_as_trial` flag per run.

## D035 — GLM+H kept: quadratic splines on the 94 characteristics, no macro interactions (2026-10-09)
- **Reading fixed before any test look:** GKX eq. 13 prints a non-truncated basis (spans only
  {1, z, z²}); we use the evident intent — order-2 splines with 3 knots, truncated (c − k)₊² — on
  each of the 94 rank-normalized characteristics (knots at fixed rank quantiles −0.5, 0, 0.5), group
  lasso per characteristic, Huber loss, no macro interactions. Matches GKX Fig. 3's "# of Char."
  axis and the model's stated role (nonlinearity without interactions). Stated as an interpretation.
- **Rejected:** dropping GLM (council 5 Simplicity); splines on all 920 z columns.

## D036 — Pre-registered diagnostics (not trials), fixed while blind to 1987–2016 (2026-10-09)
1. Training-mean benchmark: forecast = the training window's mean excess return (council 5: it
   alone scores +0.470% on the rehearsal window, ~86% of OLS-3's +0.546%).
2. Within-month demeaned R²: R² after subtracting each month's mean prediction (pure ranking skill).
3. Secondary R² excluding return months 1987-11 and 2008-11 (macro inputs far outside training
   ranges after the Oct-1987 and Oct-2008 crashes).
Reported for every model next to the primary pooled R²; none changes the success criteria.

## D037 — Procedure rules for Phase 3, fixed before the freeze (council 5, paper text where cited)
- **No refit after selection:** estimates come from the training window alone; validation only
  selects (paper lines 376–377).
- **Selection objective:** validation MSE (equivalently R²) for every model, including +H ones.
- **Scaling:** macro predictors standardized with training-window constants for every
  scale-dependent model (OLS/Huber invariant; ENet, PCR, PLS, GLM, NN not). No raw-macro run. No
  clipping or winsorizing of any input (as D033).
- **PCR/PLS** on demeaned Z (paper footnote 12), K ∈ {0, 1, …, 100}; all K from one decomposition.
- **ENet+H** only (Table 1 reports the +H version; D011's squared ENet dropped); λ log grid of 10 on
  (1e-4, 1e-1), ρ = 0.5, loss mean-scaled by 1/n as in GKX eq. 4. **NN λ₁:** 3-point log grid on
  (1e-5, 1e-3) — the same reading rule as ENet.
- **Huber ξ for +H models:** one ξ per fold = 99.9th percentile of |residual| of OLS-all (squared
  loss) on that fold's training window, shared by OLS+H, ENet+H, GLM+H, GBRT+H and logged.
- **Trees (LightGBM):** RF and GBRT run once (no seed ensemble); `num_leaves = 2**max_depth`;
  features per split as `feature_fraction_bynode = k / n_inputs`; RF subsamples without
  replacement (LightGBM has no bootstrap — stated deviation); bins from training rows only.
- **NN seeds:** 5 (D029 stands; Performance's 10-seed option declined).
- **Intercept:** every linear model gets an unpenalized intercept (fix for council 5's finding that
  rows without sic2 — 17.8% in rehearsal — were predicted at level 0).

## D038 — Model libraries pinned (2026-10-09)
- lightgbm 4.7.0 (needs Homebrew `libomp` on macOS — installed), torch 2.14.1 (CPU, 8 threads by
  default). Exact versions are in `uv.lock`. Thread counts and determinism flags are fixed in the
  Phase 3 freeze, not now.

## D039 — ENet+H solver: FISTA with KKT stopping; IRLS outer tolerance above the inner noise floor (2026-10-09)
- Verified against scikit-learn ElasticNet (squared loss; GKX eq. 8 ↔ sklearn alpha = λ/2,
  l1_ratio = 1 − ρ) and by the KKT conditions on the materialized design (Huber).
- **Finding (rehearsal fold 1986):** squared loss converged at every λ (KKT ≤ 1e-10, ≤ 1,450
  iterations), but Huber reported non-convergence: the IRLS outer loop's relative change plateaued
  at 1e-10–3e-10 — the noise floor set by the inner solver's 1e-10 KKT tolerance — so a 1e-10 outer
  tolerance could not be met (same mechanism as D032).
- **Chose:** outer tolerance 1e-8 (inner KKT stays 1e-10). Rehearsal folds 1975/1980/1986 converge
  in ≤ 4 outer iterations, 1–3 s per fold. Each fold's `converged` flag is logged in the ledger.

## D040 — Raw macro predictors for PCR, PLS, ENet+H and the NNs (reverses D037's scaling rule) (2026-10-09, before any test look)
- **Finding (council 6 Devil's Advocate; reproduced through the real harness, ledger runs
  `*-rawmacro-rehearsal` at a clean commit):** with macro standardized (D037), PCR/PLS/ENet+H had zero
  or negative cross-sectional ranking skill on the 1975–1986 rehearsal (within-month demeaned R²:
  PCR −0.172%, PLS 0.000%, ENet+H −0.000%; PLS chose K=0 in all 12 folds). With raw macro, as GKX
  eq. 21 writes x_t: pooled R² PCR +0.760% (K 9–67), PLS +0.728% (K 2–9), ENet+H +0.718%; demeaned
  R² +0.334%, +0.298%, +0.292%. Chosen K ranges match GKX Fig. 3's axes. Mechanism: centering turns
  c·m into c·(m − μ), putting the unstable interaction directions (D033) on equal footing with the
  characteristics, which the penalty and the leading components then cannot tell apart.
- **Chose:** raw macro for every scale-dependent model (PCR, PLS, ENet+H, NN1–NN3). OLS and OLS+H keep
  training-window standardization: their predictions are mathematically identical either way (D030;
  test `test_scale_macro_switch_and_ols_invariance`) and it improves conditioning.
- **Disclosure:** this reverses a pre-registered rule after seeing rehearsal results — a researcher
  degree of freedom. Grounds: the paper's text (eq. 21) states raw x_t and never standardizes it;
  the 1987–2016 test window is untouched; both variants stay in the ledger and the write-up reports
  the change. The standardized variants are not run on the test window.

## D041 — Trees: two pre-registered arms for the market level (2026-10-10, before any test look)
- **Finding (council 7; verified by the main session on rehearsal folds 1986 and 1980):** with the raw
  target, validation MSE is dominated by the monthly market level, which characteristics cannot
  predict; rank-normalized ties and median-filled not-yet-available characteristics carry era
  stamps that trees use to fit month means, so selection collapses GBRT to 1 tree (fold 1986: test
  demeaned R² −0.001%, rank IC −0.026; fold 1980: −0.018%, −0.001) or keeps anti-ranking trees.
  Trained on the within-month demeaned target: 978 / 662 trees, test demeaned R² +0.397% / +0.211%,
  rank IC +0.094 / +0.030. Linear models are immune (a linear function of ranked characteristics has
  ~zero cross-sectional mean each month).
- **Arm 1 — faithful (GKX):** raw target, selection on raw validation MSE, plus a constant fallback:
  if no configuration beats the training-mean forecast on validation, the model predicts the
  training mean (recorded as `chosen = "constant"`).
- **Arm 2 — disclosed deviation:** train and select on the within-month demeaned target (training
  and validation each demeaned by their own months); prediction = model output + the training-window
  mean of the raw target. Huber ξ for this arm = 99.9th pct of |OLS-all residual| fit on the demeaned
  training target.
- Both arms are reported (and both count under D034 as reported variants). This is the second rule
  set after rehearsal evidence (after D040); grounds: the mechanism is measured, the test window is
  untouched, and the faithful arm is kept.

## D042 — Tree inputs: rank-normalized characteristics + sic2, no standalone macro (revises D012)
- D012's raw macro inputs let trees learn month levels directly (rehearsal mean forecast 2.07% vs a
  1.29% realized mean) and are unfaithful: GKX's z has no standalone macro feature.
- **Chose:** 94 rank-normalized characteristics + sic2 (categorical; inert in rehearsal), 95 inputs.
  Stated as a "characteristics-only trees" deviation. D012's rationale is also corrected: at GBRT's
  depth 1–2 a stump cannot express a characteristic × macro interaction.
- **Rejected for the primary run:** full z (847–920 columns): ≈16–29 h and a memory rewrite (council
  7 Performance), and expected to worsen level-fitting. Optional disclosed robustness run after the
  freeze only.

## D043 — NN1–NN3 follow the same two-arm rule as the trees, fixed before any NN exists
- Arm 1 raw target (with the same constant fallback), Arm 2 within-month demeaned target with the
  training mean added back. Inputs: GKX z with raw macro (D040). Decided now so it is not another
  post-rehearsal change.

## D044 — Provenance, crash rule, thread pins, reproduction claim (council 8, 2026-10-10, before any test look)
- **Panel:** `data/processed/panel_datashare.parquet` sha256
  `dfffda9aada2ddddbd9a74ecf1acdabaa91ea74f40f38f039a83b796ec1d56fc`; `scripts/run_test.sh` refuses
  to start if it differs. Environment at the freeze: macOS 26.6.2, Apple M4 Pro (12 cores, 24 GB),
  Python 3.12.13, lightgbm 4.7.0, torch 2.14.1, numpy 2.5.3, pandas 2.2.3;
  `uv.lock` sha256 `9678171616b6a5f61c0be90b9f43e0f53ad5e48bb3750c1750c2b1c675f6a2e3`;
  runs use `uv run --locked`.
- **Threads:** LightGBM `num_threads=8` (trees.COMMON); torch `set_num_threads(4)` (nn.NUM_THREADS,
  council 6); `OMP_NUM_THREADS=8`, `VECLIB_MAXIMUM_THREADS=8` exported by `run_test.sh`. Corrects
  D038, which said torch uses 8 threads.
- **Crash rule:** re-run the same launcher command at the same commit. `evaluate()` keys every run on
  (name, commit, features, seeds, folds, target): a finished run whose prediction file still matches
  its logged sha256 is **reused** (ledger event `reused`, no refit, no new look); an unfinished run
  **resumes** from its hashed per-fold checkpoints under the same run_id. A re-run at the same commit
  is therefore not a new look. Any code change after a crash is a new commit, logged here as a
  deviation, and every earlier ledger row is kept.
- **Reproduction claim:** bit-identical predictions on this machine with this lock file and thread
  environment; on other hardware or BLAS, equal within floating-point tolerance only.
- **Disclosure:** the `reference-v1` tag was meant to hold GKX Table 1 but held only `.gitkeep`:
  `.gitignore`'s `*.csv` silently excluded `paper/gkx_table1.csv`, and I reported it as committed
  when it was not. The values (verbatim from GKX 2020 Table 1) were committed at `d322049`, file
  sha256 `de01425559ba62ed8a30a477ae4c0c399adf683a9124e63488bedf4656139b42`, before any test look.

## D045 — Verdict rule: headline model fixed, one declared confirmatory family with Holm (council 8, user-approved 2026-10-10, before any test look)
> **Amended by D047 and D048** (verdict labels; Holm became descriptive). The current rule is D048.
The prereg says "Replicates" = the paper's ranking (NN/trees > penalized linear > OLS-all) and "the
best model's" R²_oos > 0 with a 95% CI excluding zero, but never fixed which model is "best" or how
several tests combine. Fixed now, and `scripts/final_report.py` prints the verdict mechanically:
- **Headline model = NN3** (raw target, faithful arm), GKX's headline. Fixed ex ante; it was the
  *worst* raw NN in rehearsal (R² 0.481% vs NN2 0.620%), so the choice is not a rehearsal pick.
- **Primary criterion:** NN3 pooled R²_oos > 0 and its 95% month-block bootstrap CI (D015) excludes
  zero (`paired_r2_ci(zero, NN3)`). If it fails, the verdict is **does not replicate** whatever
  the other tests say (prereg "abandon / reframe" clause).
- **Confirmatory family** (raw arm only; one-sided; Holm step-down at family-wise α = 0.05):
  1. NN3 vs the training-mean forecast (DM) — the prereg's "beats baseline"; the baseline is D036's
     training-window mean, because rehearsal showed level alone earns R² ≈ 0.47%.
  2–5. NN3 vs each of PCR, PLS, ENet+H, GLM+H (DM) — "NN > penalized linear" against all four
     linear models, not just the softest one.
  6. GBRT+H vs ENet+H (DM) — "trees > penalized linear".
  7. ENet+H vs OLS-3+H (Clark–West, nested) — "penalized linear > OLS". Replaces ENet+H vs OLS+H,
     which cannot lose (OLS+H is −27,000% in rehearsal, D033); that pair is still reported,
     descriptively.
- **Verdict:** **replicates** = primary passes and Holm rejects all 7; **partially replicates** =
  primary passes, some rejected (the report lists which); **does not replicate** = primary fails.
- **Secondary (never change the verdict):** the `-dm` arms (NN3-dm, GBRT+H-dm vs ENet+H) use a
  demeaned-error DM (each month's mean target and mean forecast removed), so they test ranking
  only; all ΔR² CIs; demeaned R²; size subsamples; Sharpe and DSR; per-seed spread (D006b).
- **What was visible when this was chosen:** the full 1975–1986 rehearsal table
  (`results/final_report_rehearsal.md`, earlier pair set): NN3 vs ENet+H DM t −1.06, GBRT+H vs ENet+H
  t −2.29, NN3-dm vs ENet+H p 0.0095, GBRT+H-dm vs ENet+H p 0.055; raw RF −0.29%, GBRT+H +0.14%. On
  those numbers this rule would say "partially replicates" at best for the raw arm, so it was not
  tuned to pass. The rehearsal years are a development set (they are also fold 1987's validation
  window); rehearsal behaviour may not carry over to 18–47-year training windows.
- **DSR caveat:** N = 14 reported variants; it ignores variants abandoned in rehearsal and treats
  correlated variants as independent. Both are stated next to the number.

## D046 — Run schedule: linear → trees → two concurrent NN processes (council 8 Performance, user-approved)
> **Amended by D047:** NN arms run serially by default; concurrency is opt-in (`NN_CONCURRENT=1`).
- Linear (~45 min) then trees (~6–7 h), each alone. Then the NN raw arm (NN1, NN2, NN3) and the -dm
  arm (NN1-dm, NN2-dm, NN3-dm) as two concurrent processes, 4 torch threads each (8 of 12 cores).
  Never three processes; never trees alongside NNs (LightGBM's 8 threads would oversubscribe).
- Thread counts per process are unchanged, so predictions are identical to a serial run (checked
  against the serial rehearsal predictions before the freeze).
- Condition: the machine has 24 GB. If the frozen-runner rehearsal measures combined peak RSS of the
  two NN processes above 20 GB, the NN arms run one after the other instead (logged here).
- Run under `caffeinate -dimsu` in tmux, on AC power with the lid open.

## D047 — Verdict rule amended; fail-closed report; serial NNs (council 9, user-approved 2026-10-10, before any test look)
Council 9 found the D045 code and text disagreed at the edge that matters, and that a NaN counted
as a pass. Decided by the user before any test-window number exists:
- **"Replicates" = intersection-union test:** the primary passes and *every* one of the 7
  confirmatory one-sided p-values is ≤ 0.05. That is the valid 5% test of "every leg holds"; Holm
  on all 7 was stricter than the claim needs (seven p = 0.01 would fail it). Holm at 5% is still
  computed and reported to say which individual legs hold.
- **"Partially replicates"** needs the primary *and* test 1 (NN3 vs the training-mean forecast)
  rejected under Holm: the ML model must beat a constant. Primary passes without test 1 → **"does
  not replicate (level only)"** (rehearsal: the training mean alone earns R² ≈ 0.47%). Replaces
  D045's "some rejected", which the code had implemented as "any number including zero".
- **Fail closed:** a NaN p-value never rejects (Holm treats it as 1.0, intersection-union fails),
  and a NaN primary CI bound fails the primary.
- **Prereg amendment, stated plainly:** test 7 (ENet+H vs OLS-3+H, Clark–West) amends
  PREREGISTRATION.md's "penalized linear > OLS-all" and D010's pairing; OLS-all (OLS+H) cannot lose
  (D033), so the replacement is stricter. ENet+H vs OLS+H is still reported as secondary.
- **Freeze check in the report:** test mode refuses unless every run comes from the `freeze-v1`
  commit, is clean and on the test window, and `src/`, `scripts/`, `pyproject.toml`, `uv.lock`
  are unchanged since; `provenance_errors()` is unit-tested.
- **D046 amended — NNs run serially by default** (`NN_CONCURRENT=1` opts in). The rehearsal RSS
  (1975–1986 panel, ~⅓ of the rows) cannot test the 20 GB condition; council 8's tiled measurement
  (9–13 GB per process) already implies 18–26 GB for two on a 24 GB machine with swap in use.
  Total run ≈ 30 h. Thread counts are unchanged, so predictions do not depend on the schedule.
- **Bit-identity pass rule** (written before the tree/NN comparison was read): for each of the 18
  models, every `pred*` column of the 01f0936 frozen-runner rehearsal run equals, with
  `np.array_equal`, the most recent earlier finished rehearsal run of the same name. Any mismatch
  blocks the tag until it is explained in a D-entry.
- **What the rehearsal covers:** the `01f0936 → freeze-v1` diff touches only the report side
  (`src/gkx/report.py`, `scripts/final_report.py`), the launcher (`scripts/run_test.sh`: serial NNs,
  30 GB disk gate, refusals logged), `scripts/check_bit_identity.py`, `.gitignore`, tests and docs. `report.py` and `final_report.py` are not imported by `run_test.py`, so the dry run still
  validates the frozen prediction path; `final_report.py --rehearsal` is re-run at the tag.
- **D044 addendum — an analysis bug found after the look:** fix it in a new commit logged as a
  deviation, re-run `final_report.py --deviation D0xx` (the flag names the decision; the changed files are
  printed in the report header), and publish both outputs side by side; predictions are never
  re-fit.
- **During the run:** no Claude session or council writes inside the repo (a new file dirties the
  tree and the next spec refuses); `.claude/settings.local.json` is git-ignored; progress is read
  only as counts of `fold_done` rows, never the rows themselves (they carry test-year validation
  scores).

## D048 — "Partially replicates" uses test 1 at raw α, so the verdict is monotone (council 10, user-approved 2026-10-10, before any test look)
- Council 10 found D047's labels non-monotone: "replicates" used each raw p ≤ 0.05, but
  "partially" needed test 1 rejected under Holm across all 7. Seven p = 0.01 gave "replicates"
  while printing "test 1 not rejected"; raising only test 7 to 0.06 dropped the verdict two levels
  to "does not replicate (level only)".
- **Rule (final):** primary fails → "does not replicate". Otherwise every confirmatory p ≤ 0.05 →
  "replicates"; else test 1 (NN3 vs the training-mean forecast) p ≤ 0.05 → "partially replicates";
  else "does not replicate (level only)". Both labels are intersection-union tests of a
  conjunction (valid at 5%), "replicates" implies "partially", and a worse p-value can never
  improve the label (`test_verdict_is_monotone_in_each_p_value`). NaN still fails closed.
- Holm at 5% is still computed and reported, as a description of which individual legs survive a
  family-wise correction; it no longer decides any label.
- Prediction path untouched (report code only); `.claude/council-cache/` is now git-ignored so a
  council record can never dirty the tree during the run (records are added with `git add -f`).

## D049 — Phase 4 step 1: zero-refit diagnostics, listed before any is computed (council 11, 2026-10-11, after the single look)
Everything here is **exploratory**: the 1987–2016 window was seen at `cb59143` (`result-v1`). The
list is fixed now so the diagnostics do not grow with what they show. Inputs: the 18 frozen test
prediction files (hash-checked against the ledger), the panel and `crsp_returns_pre2017.parquet`;
nothing after 201612 is loaded and `data/sealed/` is never opened. Code: `scripts/phase4_diagnostics.py`
→ `results/exploratory/phase4_diagnostics.md`, every output headed by a forced EXPLORATORY banner.
1. **Seed fragility** (NN1–NN3, raw and -dm): R² and test 1 (DM vs train_mean) for all 31 seed
   subsets; summary by subset size; every single-seed and leave-one-seed-out value for NN3.
2. **Level vs ranking:** SSE split into a between-month part (Σ n_t (r̄_t − p̄_t)²) and a
   within-month part, each as % of Σr², per model; NN3's gain over train_mean split the same way.
   Pooled Mincer–Zarnowitz slope (target on forecast) per model.
3. **Where the R² lives:** R² for train_mean, ENet+H, GBRT+H, RF, NN3, NN3-dm within NYSE-breakpoint
   size quintiles (`me_lag`, `exchcd == 1`), common shares only (`shrcd` 10/11), and excluding each
   permno's last panel month (proxy for the delisting month, D019). Price ≥ $5 needs a new WRDS
   pull (no price in the build) — deferred, stated.
4. **Influential months:** NN3 vs train_mean per-year ΔR²; test 1 recomputed after dropping the 1, 3
   and 5 months contributing most to NN3's advantage.
5. **GBRT+H:** its constant-fallback folds; GBRT+H vs ENet+H on the remaining years.
6. **OLS+H:** share of its squared error from the worst 1/5/10 months and worst 0.01% of rows.
7. **DM weighting sensitivity:** the 7 confirmatory tests with each month's loss differential
   *averaged* over stocks (equal-weighted months) instead of summed (D010), and a note on D008's
   wording ("monthly mean") vs the code (monthly sums).
No result here changes the D048 verdict; none is a new confirmatory test.
- **Addendum (council 12, before any further analysis):** the code moved to
  `exploratory/phase4_diagnostics.py`. Under `scripts/` it made `final_report.py`'s frozen-path check
  refuse at HEAD. `tests/test_phase4.py` now asserts `src/`, `scripts/`, `pyproject.toml` and
  `uv.lock` are unchanged since `freeze-v1`. Deviations from the list above, logged:
  - item 3 uses each permno's last *CRSP* month, not last panel month (it carries the D019
    delisting return and also mergers), and conditioning on it is look-ahead;
  - NYSE breakpoints use all NYSE panel rows, not common stocks only;
  - item 7 re-weights the 6 DM tests; test 7 is Clark–West and is not re-weighted;
  - item 4 adds the symmetric trim (train_mean's best months) and names the column as post hoc.
  Wording: the within-month SSE part is "cross-sectional fit" (ordering and forecast scale), not
  pure ranking.
- **Correction (council 13):** "seen at `cb59143` (`result-v1`)" above conflates two commits:
  `cb59143` holds the test-run report; tag `result-v1` points at `d2086a7`, which adds the council-11
  review and README update on top of it.

## D050 — Phase 5 pre-registration: the sealed holdout, one confirmatory test (user-approved 2026-10-12; revised after council 14; binding at tag `phase5-prereg`, before the holdout is opened)
Written after the 1987–2016 look (`result-v1`), so it is informed by that window. Fixed and pushed
to https://github.com/PatrickDamouni/gkx-replication and tagged `phase5-prereg` before any 2017+
characteristic, macro value or return is read. (Git dates are author-set; the public push is the
external evidence.)

**1. Deviation from the pre-registration, stated plainly.** PREREGISTRATION.md (prereg-v1) asked
whether the *ranking* (NN/trees > penalized linear > OLS) holds after publication. The ranking did
not hold on 1987–2016, and on ~60 holdout months its tests have little power (NN3-dm vs ENet+H
within-month ≈ 60% at full strength, ≈ 23% at half; NN3 vs ENet+H ≈ 13%). D050 replaces that question
with a narrower one (H1) and reports the ranking tests without letting them decide anything. H1
is not a test of whether machine learning beats linear models.

**2. Scope: Arm A only.** GKX's datashare characteristics (sealed, sha256
`9a87fbf7153c2a4d6d61ac93196df24f59a6f4d590566d2e780a0965e5a1881b`, D023) and Welch–Goyal macro
(sealed, sha256 `4d2640cc569645deaa94cab27bd51ac32e3469eb72371660a3fd07110360448e`). Test months = every return month from 201701 to the last
month with datashare characteristics, inner-joined with CRSP returns: whatever length that is
(estimated 24–84 months), no minimum, a thin or partial final year kept and reported. Arm A can
never be swapped for Arm B. Arm B (OSAP, 2017–2024) is deferred; once Arm A is read, only months
after Arm A's last month could ever support a confirmatory Arm B claim.

**3. Returns.** Pre-2017 rows, including December-2016 market equity for January 2017's `me_lag`,
always come from the frozen `crsp_returns_pre2017.parquet`; pre-2017 CRSP is never re-pulled.
2017+ CRSP is pulled once from legacy `crsp.msf`/`crsp.msedelist` with the D019/D020 build logic,
written straight to `data/sealed/`, its sha256 recorded in a decision entry and pushed before
unseal, and not read until then. Before that pull, 2015–2016 is re-pulled and must equal the
frozen file (checks that the legacy tables are unchanged). If legacy tables no longer serve 2017+,
stop: new decision entry before anything else.

**4. Models: unchanged.** The 18 specifications of `scripts/run_test.py` at `freeze-v1`, NN seeds
0–4, same grids, thread pins and libraries; NNs run one at a time (concurrent NN processes are
refused in Phase 5: ≈ 13–14 GB each at the largest holdout fold on a 24 GB machine).

**5. Folds.** GKX's schedule continued: one refit per test year; training 1957-03 → the year before
validation (expanding); validation the 12 years before the test year (2017: train 1957–2004,
validate 2005–2016). Earlier holdout years therefore enter later folds' validation and model
selection — by design; no test year enters its own fit. The fold list is **2016 → last year**:
2016 is a canary whose predictions must equal the `freeze-v1` 2016 test predictions
(`np.array_equal` on every `pred*` column, all 18 models). It reveals nothing about the holdout
and is the only test of the merged loader (row order drives NN batching and tree binning).

**6. Code: `freeze-v2`, by whitelist.**
- Byte-identical to `freeze-v1`: `src/gkx/{linear,reduction,enet,glm,trees,nn,evaluate,metrics,
  report,preprocess,build,macro,datashare}.py`, `scripts/run_test.py`, `scripts/final_report.py`,
  `pyproject.toml`, `uv.lock`.
- May change only by additive keyword arguments whose defaults keep the seal:
  `src/gkx/{panel,walk_forward,align,crsp}.py`.
- New files: a holdout panel builder, `scripts/run_holdout.py`, a launcher, and
  `scripts/phase5_report.py`, plus tests.
- Every test that exists at `freeze-v1` passes unmodified at `freeze-v2`. New tests on synthetic
  data cover: the pre/holdout join with no gap at 201612→201701 (January 2017's return row comes
  from the pre-2017 datashare file, which is split on signal month); a partial final year; sha
  checks failing closed; refusal of mixed commits or run tags.
- Run names end in `-check2016` (pre-tag check, `counts_as_trial=False`) or `-holdout`; the
  1987–2016 side is always loaded by commit `46c1584`, never by "latest".
- Before tagging `freeze-v2`, at the exact commit to be tagged: the 2016 fold of all 18 models is
  re-run on the frozen panel through the new runner and must match `freeze-v1` bit-identically, and
  `phase5_report.py` run on the 1987–2016 predictions must reproduce H1 t = 4.569 and test 1
  t = 2.213. A mismatch blocks the tag: first restore the environment (macOS/Accelerate drift is the
  likely cause); otherwise a new decision entry before unseal. No tolerance is pre-set.

**7. Panel lock and the run.** The holdout panel is built once; its sha256 is written to
`results/holdout_panel.sha256` before the first fit; it is never rebuilt; its ≤ 201612 rows and its
schema and column order must equal the frozen panel (sha `dfffda9a…`, D044) or the build aborts.
The launcher refuses a different panel, any commit but `freeze-v2`, or concurrent NNs. Crash
(D044): re-run the same command — finished runs reused, unfinished resumed. A code bug found after
unseal: fixed once at `freeze-v2.1`, logged as a deviation, all 18 holdout runs refit (≈ 8 h),
every pre-crash run must reproduce, the canary re-verified, the report run once.

**8. The one confirmatory test (H1).** NN3-dm (5-seed ensemble, demeaned-target arm, D043) beats
the constant (each training window's mean raw return) on the holdout test months:
`gkx.metrics.diebold_mariano(df, base="train_mean", candidate="NN3-dm")` — one-sided, monthly summed
squared-error differentials, HAC with the frozen constant `metrics.HAC_LAGS = 5` (D015's rule would
give 3 at T ≈ 60; 5 is kept to match the frozen code and the power simulation), normal p,
α = 0.05 nominal.
- **Disclosed:** H1 was picked as the top of five candidate tests in
  `results/exploratory/phase5_power.md`, after seeing 1987–2016 (frozen-code t = 4.569 there).
  NN3-dm's forecast is the model output plus the same training-window mean as the baseline, so H1
  is mostly a test of within-month cross-sectional fit. For contrast, ENet+H vs the constant had
  t = 1.63 (p = 0.052) on 1987–2016.
- **Power** (block bootstrap, exact frozen HAC): at 60 months ≈ 68% if the 1987–2016 effect persists
  unchanged (optimistic: winner's curse, top-of-five selection), ≈ 30% at half strength — the
  realistic central case given typical post-publication decay. At 24–36 months ≈ 56–58% / ≈ 29%.
  The test over-rejects in short samples: simulated size ≈ 6% at 60 months, ≈ 8% at 36, ≈ 12% at 24.
  **The most likely outcome is "not detected".**
- **Reading, fixed now** (printed mechanically by `phase5_report.py` from p ≤ 0.05):
  - reject → "On GKX's characteristics, the demeaned-target NN3's within-month edge over a constant
    survived the first post-publication years (Arm A, N months). This is not evidence that machine
    learning beats linear models."
  - not reject → "Not detected (Arm A, N months). With ≈ 30% power at half the 1987–2016 effect,
    this is weak evidence of decay, not evidence of absence."

**9. Reported, deciding nothing.** All 18 models' holdout R² with month-block bootstrap CIs; the
seven D045 tests (with their power stated in advance: test 1 ≈ 23%); NN3-dm vs ENet+H on
within-month errors; ENet+H vs the constant; each beside its 1987–2016 value. No D048 label is
printed for the holdout (declared uninformative in advance), and no decay ratio. With ~5 blocks of
12 months, bootstrap CIs are coarse; the report says so. NN3's raw-target edge already faded after
2002 (exploratory, D049), so a null for the raw arm is the expected result.

**10. Single look.** A "look" is any metric computed on holdout *test-month* predictions. It happens
once: `phase5_report.py` runs after all 18 holdout runs finish from `freeze-v2`. Until then progress
is read only as counts of `fold_done` rows (they carry validation scores on 2017+ months).

## D051 — How freeze-v2 implements D050 (council 15, 2026-10-12, before the CRSP pull and before any holdout read)
Fixed now so nothing is decided after the look. Nothing here changes H1, its reading or the models.
- **Diff from `freeze-v1`:** `align.py` (`through=`) and `walk_forward.py` (`allow_holdout=`) only,
  both with seal-keeping defaults; `gkx/crsp.py` stays byte-identical (the date-bounded queries live
  in `scripts/pull_crsp_holdout.py` with bound parameters); new `gkx/holdout.py` and five scripts.
  `tests/test_freeze_v2.py` enforces this, and that no test file present at `freeze-v1` changed.
- **Gate, in Python and in the launcher:** sealed data and holdout predictions are only touched
  from a clean checkout of exactly `freeze-v2` or `freeze-v2.1` (`holdout.require_freeze_v2`).
- **Test months (D050 §2, made exact):** return months from 201701 to the last month that has
  characteristics, macro values and a non-missing CRSP excess return, contiguous, with no missing
  macro predictor; the builder refuses otherwise. (Macro ends 2025-12, so the macro cap should not
  bind; it is stated because the code applies it.)
- **"≤ 201612 rows equal the frozen panel" (D050 §7):** met by construction: the holdout panel holds
  return months ≥ 201701 only, and `load_merged` reads every earlier row from the sha-checked frozen
  panel through the same call `run_test.py` used. The unseal builder itself was rehearsed on real
  pre-2017 data with 2016 playing the holdout (`build_panel_holdout.py --verify-mirror`): it
  reproduced the frozen panel's 68,730 2016 rows exactly.
- **Panel lock:** after `--build`, the holdout panel and `results/holdout_panel.sha256` are made
  read-only and the panel's sha256 is published as the message of annotated tag `phase5-panel`,
  pushed to origin, before any holdout fit. Every holdout invocation logs a `phase5_inputs` ledger
  row with the panel hash it used; the report refuses unless all of them equal the published one.
- **One launcher at a time** (`results/.phase5.lock`); NN specs run serially.
- **`freeze-v2.1` (D050 §7, made exact):** only for a code bug found after unseal; the refit at
  `v2.1` governs; any model whose `v2` holdout predictions differ from its `v2.1` predictions is
  listed in the report header. No `freeze-v2.2`.
- **The look:** `phase5_report.py --holdout` checks every input first (tag, clean tree, published
  lock, one panel hash, `-check2016` evidence at the same commit, canary fold 2016, no NaN), then
  computes, writes `results/phase5_report.md` once (refuses if it exists) and appends a
  `phase5_look` ledger row. The 1987–2016 comparison values are recomputed from the `46c1584`
  predictions by the same code; the dry run (`results/phase5_report_dryrun.md`) exercises that
  whole path on 1987–2016 and reproduces H1 t = 4.569 and test 1 t = 2.213.
- **Runbook:** (1) `pull_crsp_holdout.py --check`; (2) `--pull`; its sha256 goes into
  `holdout.CRSP_HOLDOUT_SHA` and a decision entry, committed; (3) at that commit: dry run and
  `run_holdout.sh --check2016`, then `phase5_report.py --check2016` must show 18/18; (4) tag and push
  `freeze-v2`; (5) the user's go-ahead; (6) from a detached `git checkout freeze-v2`:
  `build_panel_holdout.py --build`, tag and push `phase5-panel`, `run_holdout.sh --holdout`,
  `phase5_report.py --holdout`; (7) only then commit results (the ledger, logs and report) on
  `master`. Nothing under `results/` is committed or pushed between (5) and (7): the ledger carries
  validation scores on 2017+ months.
