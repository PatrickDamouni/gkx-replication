# Pre-registration (LOCKED 2026-10-07 — tag `prereg-v1`)

Locked before any data was loaded. The git tag `prereg-v1` is the timestamped proof.
Any later deviation is logged in `DECISIONS.md` with the reason, so ex-ante vs ex-post
choices stay distinguishable.
## Question

Do the machine-learning gains in Gu, Kelly & Xiu (2020, RFS) — "Empirical Asset
Pricing via Machine Learning" — replicate on independently built data, and do they
**survive after the paper's sample ends** (2017 onward)?

Two parts:
1. **Replication.** Using Open Source Asset Pricing (Chen & Zimmermann) signals +
   CRSP returns, do nonlinear models (trees, neural nets) beat linear models
   out-of-sample over the paper's test window (1987–2016), by a margin larger than noise?
2. **Extension (post-publication holdout).** Over 2017–latest available CRSP month,
   does that ranking hold? Published anomalies are known to decay after publication
   (McLean & Pontiff 2016); this tests whether ML-combined signals decay too.

## Success criteria (fixed now)

- **Primary metric:** monthly out-of-sample R², GKX definition (denominator is the sum
  of squared *raw* excess returns, not demeaned).
- **"Replicates"** = same model *ranking* as the paper (NN/trees > penalized linear >
  OLS-all), and the best model's R²_oos is positive with a 95% CI excluding zero.
  Matching the paper's exact numbers is **not** required — different data build.
  > Operationalized and amended before any test-window look in DECISIONS.md D045 (headline
  > model, confirmatory family), D047 and D048 (final verdict labels). The tag `prereg-v1`
  > keeps the original text.
- **"Beats baseline"** = Diebold-Mariano test vs. the baseline, p < 0.05, on the test window.
- **Economic metric:** decile long-short portfolio, value-weighted and equal-weighted,
  monthly Sharpe with bootstrap CI. Report the deflated Sharpe ratio
  (Bailey & López de Prado 2014) given the number of model configs tried.
- **Seeds:** every stochastic model run with ≥5 seeds; report mean ± spread, never the best seed.

## What would make me abandon / reframe

- If the best model's R²_oos CI includes zero over 1987–2016, the replication failed —
  the write-up becomes a diagnosis of *why* (data build, features, tuning), not an extension.
- If a leakage check fails at any point, all downstream results are discarded and rerun.

## Splits (locked)

| Window | Use |
|---|---|
| 1957-03 → 1974-12 | initial training (expanding, refit annually) |
| 12-year rolling window after training | validation / hyperparameter selection |
| 1987-01 → 2016-12 | test (the paper's window) |
| **2017-01 → latest CRSP month** | **sealed holdout — not loaded into any model or plot until Phase 5** |

## Baselines (built first, in this order)

1. Zero forecast (R²_oos = 0 by definition under GKX's metric).
2. OLS on 3 characteristics: size, book-to-market, momentum.
3. OLS on all characteristics (expected to do badly — that's the point).

## Known data caveats (stated up front)

- Feature set will differ from GKX's 94 characteristics; mapping documented in `DECISIONS.md`.
- Delisting returns handled per standard CRSP practice; choice logged.
- Academic-use-only data (WRDS) — raw data never committed.
