# Local council 4 — Phase 2 plan (panel + baselines) before the test window

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Question: is Phase 2 (docs/phase2_plan.md, commit b49fbde) the best next step, and what must
change before 1987–2016 is touched? Members (`--roles=devil,simplicity,performance`):
Devil's Advocate, Simplicity Champion, Performance Optimizer.

---

## 🗳️ Simplicity Champion
**Position.** Phase 2 is right; cut it to four OLS fits on Arm A from persisted predictions; fix
six cheap things first; no separate sanity-reproduction or memory step.
**Key points.** *Confirmed bugs:* macro `bm`/`ep` collide with characteristics `bm`/`ep` in
build_panel (tests miss it — fixture macro has only dp/tbl); evaluate() drops `me_lag` and the
ledger stores only R² → VW Sharpe needs a re-run. Fix by persisting predictions + me_lag to
results/preds with a hash in the ledger. Zero forecast is a column of zeros, not a trial; its
decile sort is meaningless (ties → permno order). OLS-3 without interactions (prereg). OLS-all:
refit per fold, no cross-fold Gram cache; Huber via X'X minus outlier-row correction; float64.
Guard: evaluate() refuses folds overlapping 1987–2016 unless the git tree is clean; debug on
`walk_forward(first_test_year=1975, last_test_year=1986)` with ledger=None.
**Risks.** Pin before running: OLS on train vs train+val (check GKX); which ledger rows count as
deflated-Sharpe trials (baselines/robustness arms should not). Commit GKX Table 1 reference
values before the run. Keep Phase 2 to four test calls; robustness arms after Phase 3. 6.8% of
rows lack sic2. ξ by rule, not tuning.
**Confidence.** high (bugs reproduced).

## 🗳️ Devil's Advocate
**Position.** Build the panel; do NOT run baselines on 1987–2016 yet. Fix the bugs, run every
baseline on a pseudo-test window (train 1957–74, predict 1975–86), freeze code for *every*
model, then touch the test window once for all models.
**Key points.** (1) bm/ep collision (confirmed). (2) decile_long_short with ties sorts by
permno: a constant forecast gave a +0.76%/month spread on synthetic data → zero forecast and
tied OLS-3 predictions produce spurious Sharpes. (3) Ledger can't support "one test run": row
written only after all folds (crash = unlogged look), ledger=None skips, dirty tree accepted,
predictions not persisted, only R² logged, no 1987–2016 protection. (4) OLS-all numerics: raw
macro makes size×dp correlate −0.999 with size; X'X condition ~1.8e7 for one characteristic's
9 columns — "badly negative OLS-all" could be numerical. Fix float64, solver, no ridge,
intercept. (5) Sequential test-window looks let the test window tune the panel; baselines
confirm known numbers (low value, high contamination). Use pre-1987 checks (e.g. Ken French
decile sorts) instead. If baselines must run first, pre-register a discrepancy protocol.
**Risks.** GKX comparability undecided in three places (OLS train vs train+val; Huber ξ fixed vs
tuned; Table 1 OLS rows may be Huber-only — recalled, unverified). Table 1 also reports top/bottom
1,000 stocks — define now. D017's "rerun power with real residual vol" must stay pre-1987. Macro
predictors leave their training range after 1987. Time one Huber fold first.
**Confidence.** high.

## 🗳️ Performance Optimizer
**Position.** Baselines aren't the bottleneck; the harness is. Lock harness, numerics and the full
compute budget, prove them on a pre-1987 pseudo walk-forward, then run baselines once.
**Key points.** *Bugs:* bm/ep collision; CRSP loads as nullable Float64/Int64 → `to_numpy()`
gives dtype=object; sic2 not in panel (since fixed in 79938bf via `keep`). *Measured (≤1986
return months, 1.25M rows):* build_panel 11 s, 4.3 GB peak; batched 846-column X'X 0.6–1.7 s per
1M rows; Kronecker sufficient statistics X'X = Σ_t (x_t x_t') ⊗ (C_t'C_t), X'y = Σ_t x_t ⊗ C_t'y_t
exact (rel. error 3e-14) in 0.26 s for all rows, weighted pass (Huber IRLS) also 0.26 s — the
whole linear family (OLS, Huber, ENet covariance-mode, PCR, PLS) never materializes 920 columns
and folds become prefix sums. *Numerics:* raw macro → cond(X'X) 5.3e10; standardized macro →
1.1e6; float32 accumulation ~2e-5 relative error → float64 required; standardize macro with
train-only constants (OLS predictions unchanged) or QR/eigh with logged rank tolerance.
*Harness:* evaluate() copies per seed, hides tuning inside fit, doesn't persist predictions,
serial and pandas-bound → month-sorted float32 memmaps + offset index; harness owns grid × seed
loop, logs every config's validation score, persists predictions + me_lag. *Budget:* NN ≈ 10–20
h per architecture (4 configs × 5 seeds × 30 folds); 5 architectures × 3 arms ≈ 150–300 laptop
hours; sklearn exact-split RF is hundreds of hours → histogram boosting / max_samples; torch and
lightgbm not installed. Fix grid, seeds, refit cadence and libraries in DECISIONS before the test touch.
**Risks.** Dev mode unenforceable today (walk_forward defaults to 1987–2016). Pre-1987 rehearsal is
2.7× smaller than the largest fold. Disk 97% full — plan prediction retention. Check GKX for OLS
train vs train+val.
**Confidence.** high on measurements; medium on NN/RF hours.

---

## Synthesis — angles, not consensus

**Shared starting points (common prior — stress-test).** All three independently found the
bm/ep column collision (all reproduced it). All three: persist predictions (+ me_lag) so metrics
never require a re-run; ledger must log before/independently of completion and capture Sharpe;
enforce test-window protection in code (clean tree + explicit unlock); float64 and a fixed solver
for OLS-all; check GKX's OLS training window and ξ rule from the paper, not memory; commit the
Table 1 reference values before any test run. Shared blind spot: all three recalled GKX details
(Huber-only OLS rows, ξ rule, train vs train+val) from memory — the paper must be read.

**Genuine tension.** Simplicity: run the four OLS baselines on the test window as soon as the
bugs are fixed and the guard exists. Devil + Performance: no test-window look until the harness
and every model family's code, grid and compute budget are frozen; then one look for all models.
Trade-off: earlier feedback and a smaller Phase 2 vs. eliminating the forking path where
test-window baseline numbers shape the panel or the later model choices.

**Unique points.** Devil: tie-breaking in decile sorts creates spurious Sharpes (+0.76%/month on a
constant forecast). Performance: Kronecker sufficient statistics make the entire linear family
exact and fast; macro standardization fixes conditioning (5.3e10 → 1.1e6); NN/RF budget ≈ 150–300 h.
Simplicity: the zero forecast is not a trial.

**Not covered by anyone.** Whether the laptop budget makes the full GKX model set feasible at all
before the Summer 2027 application deadline — a scoping decision for the user.
