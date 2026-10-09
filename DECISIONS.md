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
