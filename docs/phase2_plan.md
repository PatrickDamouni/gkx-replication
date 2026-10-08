# Phase 2 plan: modeling panel + pre-registered baselines (draft for review)

Status: draft, written before any model has been fit to returns. Decisions referenced are in
DECISIONS.md.

## 1. Modeling panel (Arm A, datashare)
- Characteristics: datashare's 94 (signal-month convention, D023), restricted to the CRSP
  universe at the signal month and rank-normalized to [-1, 1] within that universe each month,
  missing → 0 (= cross-sectional median) — `build_panel` (Phase 1b).
- Target: CRSP excess return at t+1 incl. delisting returns (D019), all NYSE/AMEX/NASDAQ
  securities (D020). Value weights: ME at the signal month.
- Macro: 8 Welch-Goyal predictors at the signal month (D021), not normalized.
- Industry: 74 sic2 dummies from datashare's sic2.
- GKX covariates: z_it = x_t ⊗ c_it with x_t = (1, 8 macro) and c_it = 94 chars → 846 columns,
  plus 74 dummies = 920. Built in batches inside models (D007), never stored whole.
- Sample: return months 1957-03 → 2016-12; walk-forward folds (seam 3).

## 2. Baselines (PREREGISTRATION.md, D011)
1. Zero forecast.
2. OLS-3: size (mvel1), book-to-market (bm), momentum (mom12m), no interactions — squared loss
   and Huber loss.
3. OLS-all: all 920 covariates — squared and Huber.
- Huber threshold ξ: GKX choose it adaptively as a high quantile of residuals; proposal: ξ = 99.9th
  percentile of |residual| from the squared-loss fit on the same training window.
- Fitting: expanding training window per fold, refit annually (30 folds). OLS has no
  hyperparameters; the validation window is not used for these models.
- OLS-all computation: accumulate per-year Gram matrices X'X, X'y (920² floats), cumulative sums
  per fold → closed form; Huber via IRLS over row batches.

## 3. Evaluation (all on 1987–2016, logged to results/trials.jsonl)
- Pooled R²_oos (D005); Clark–West vs zero and OLS-3 vs OLS-all (D010, D017); paired ΔR² CI
  (effect size); decile long-short EW/VW Sharpe with month-block bootstrap CI (D015).
- Reference values to compare against GKX Table 1 (monthly R²_oos, %): to be filled from the paper,
  not from memory.

## 4. Discipline
- Proposal: develop and debug the pipeline only on fold 1's validation window (1975–1986), freeze
  the code (git tag), then run each baseline once on the test window. Every evaluate() call is
  logged to the trial ledger.
- Common-shares-only robustness (D020) after the main baselines.

## Open questions for review
- Is this the right next step, or should something come first (e.g. a sanity reproduction of a
  single GKX number, or the full-panel memory check with 920 covariates)?
- Is "develop on validation only, then one test run" sufficient, and how should it be enforced?
- Huber ξ rule; OLS-3 with or without macro interactions; whether OLS-all needs any guard
  (GKX report it as badly negative — that is expected, not a bug).
- Anything that would make the baseline numbers incomparable with GKX's Table 1.
