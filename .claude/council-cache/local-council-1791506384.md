# Local council 5 — Phase 2 rehearsal findings + Phase 3 plan, before the freeze

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Reviewed: D031–D033, src/gkx/linear.py, docs/phase3_plan.md (commit 5b3b612).
Members (`--roles=devil,simplicity,performance`). All checks used data ≤ 1986 only.

---

## 🗳️ Simplicity Champion
**Position.** Freeze the linear family as is; the plan is over-built (harness-owned grid,
configs × seeds trial count, GLM engine) and under-built in one blocker.
**Key points.** *Blocker:* `results/trials.jsonl` is git-tracked → the first unlocked evaluate()
dirties the tree → every later test-window call raises TestWindowLocked; rehearsal never checks
dirtiness so it can't catch this. Fix: exclude results/ from the dirty check + a test. Selection
belongs inside fit() (validation data already passed); PCR/PLS give every K from one
decomposition, ENet one warm-started path, LightGBM native early stopping; log `val_scores_` and
`chosen_`; unselected configs never predict on test. Reuse one z code path (`_design_rows`,
`_normal_equations`); mean-scale A, b by n (GKX loss is 1/NT). LightGBM traps: `num_leaves =
2**max_depth`; features-per-split → `feature_fraction_bynode = k/103`; rf needs bagging; GBRT
deterministic (one run). Answers: (1) one scaling rule everywhere — macro standardized with
training constants, incl. PCR/PLS; no raw "robustness" run (a fork); (2) drop GLM+H, log it; (3)
keep 5 NN seeds; (4) freeze linear; drop squared ENet (Table 1 reports +H only); (5) checklist:
ledger fix → synthetic tests → build cheapest-first → rehearsal per model → time the real 1987
fold's fit → seed-0 tuning if NN/RF > 20 h → pin libs → chunked run_test.py → crash-rerun rule → tag.
**Risks.** D028's configs × seeds count is the biggest framing error: grid points are selected on
validation and seeds averaged — N should be the ~9–10 model specs whose test Sharpe is seen. ξ for
ENet+H / GBRT+H needs a rule (one per fold from OLS-all residuals). Each seed re-runs the grid.
Long run needs fold chunking + caffeinate.
**Confidence.** high on the ledger bug, LightGBM cap, selection-in-fit, dropping GLM.

## 🗳️ Performance Optimizer
**Position.** Compute fits a weekend, but: the disk premise is false (145 MiB free), GLM is cheap
not infeasible, RF/GBRT seeds are waste, and the harness can't survive a 20–40 h pass.
**Key points.** *Disk:* 100% full; `uv add torch` will fail; a disk-full write mid-test leaves an
orphan "started" row; reclaimable: ~/.cache/uv 1.8 GB (user should approve). Test-window preds ≈
70 MB per single-seed model, ~270 MB per NN architecture at 10 seeds, ~1.5 GB per arm; float32
halves it. *GLM:* measured float64 Gram 5,000×3,754 in 0.095 s → full-panel Gram ~70 s; yearly
additive stats; ≤1–2 h for 30 folds. Printed eq. 13 basis (1, z, (z−c)²) is non-truncated → spans
{1, z, z²} → Kronecker-factorizable; truncated (z−c)₊² is 3,700 columns. *NN:* average test fold
trains on ~1.7M rows (fold 1987: 476k); ~0.5–2 min per training; S=5 ≈ 8–12 h wall with 4 workers,
S=10 ≈ 15–25 h; recommends 10 seeds for Arm A only if a full-scale timing confirms ≤ 2 min.
RF/GBRT one seed. Keep early-stopped best weights (no refit). Share fold-level A, b, y'y for
validation MSE of every grid point; build LightGBM Dataset once per fold, bins from training rows.
**Risks.** Same git-tracked ledger blocker; per-(model, fold) checkpoints + resume event; rehearsal
timings don't transfer (415k vs 3.34M rows) — tile ≤1986 rows to 3.3M; pin threads and determinism;
KKT/duality-gap logging for ENet/GLM; LightGBM RF subsamples without replacement; check libomp.
Freeze checklist: ≥ 10 GB free preflight, pins, tiled timing, resume tested, determinism, cheapest-first.
**Confidence.** medium (disk/git/linear algebra measured; LightGBM/torch estimated).

## 🗳️ Devil's Advocate
**Position.** Don't freeze yet: the rehearsal looks cleaner than it is.
**Key points.** *Level:* of OLS-3's +0.546% rehearsal R², ~+0.47 is the monthly mean prediction
(demeaned within month: +0.08%); a training-mean forecast alone scores +0.470%; OLS-nomacro has R²
−1.58% but EW Sharpe 2.08 vs OLS-3 1.02. "R² > 0" is near-automatic for any positive-mean model,
Clark–West vs zero mostly tests the intercept. Pre-register a training-mean benchmark and a
within-month-demeaned R² (not trials). *sic2 bug:* 17.8% of rehearsal rows lack sic2; OLS-all has
no intercept and relies on dummies → those rows get level 0 (mean pred 0.24% vs 0.84%; realized
1.22% vs 1.30%) — every linear model inherits it; tilts the ranking toward nonlinear models. Fix:
missing-sic2 category or unpenalized intercept. *Paper settles:* estimates from training data alone
(no refit); tuning on the validation *objective*; PCR/PLS on demeaned Z (footnote 12); PLS needs
K=0, PCR up to ~100; eq. 13 non-truncated basis; inconsistent λ grid readings. Answers: (1)
standardize macro for every scale-dependent model incl. PCR/PLS; raw macro as reported sensitivity;
no clipping; (2) keep GLM (GKX's "nonlinearity without interactions" control) — pick 920 vs 94
reading ex ante; (3) keep 5 seeds; (4) sic2/intercept fix, level decomposition, rerun D031–D033 from a
clean commit (7/9 ledger runs dirty); zero-coverage chars persist into fold 1987's training; (5)
ranking pairs and headline model named in advance; power analysis rerun with rehearsal residual vol;
synthetic 3.34M timing; dry-run the test script; pins; crash/bug protocol.
**Risks.** D028 miscalibrated (N ≈ number of reported model variants, ~13). Covariate shift: svar in
Oct-1987/2008 an order of magnitude above rehearsal max; dp below rehearsal min — pre-register a
secondary R² excluding the two crash-following months now. D027 leaks: run_baselines prints each R²
and the ledger records r2_oos per model before later models run → write/hash all predictions first,
compute metrics in a separate final step. `_git()` fails open; LEDGER cwd-relative; rehearsal loads
data through 2016 — use end=198612. Rehearsal is for bugs, not settings. LightGBM: no bootstrap,
num_leaves, features-per-split of 920 vs 103, sic2 encoding, GBRT ξ basis; ½ loss scaling.
**Confidence.** medium-high.

---

## Synthesis — angles, not consensus

**Shared starting points (common prior — stress-test).** Two members independently found the
git-tracked ledger blocker. Two (Simplicity, Devil) say D028's configs × seeds over-deflates; the
right N is the number of model variants whose test Sharpe is reported. All three: one macro-scaling
rule (training-window standardization) for every scale-dependent model; RF/GBRT one run;
no refit after selection; LightGBM `num_leaves = 2**depth` and features-per-split as a fraction.

**Unique, measured findings.** Devil: the level decomposition (+0.47 of +0.55) and the missing-sic2
level bug (17.8% of rows predicted near 0). Performance: the disk is full (145 MiB); GLM is
feasible (~1–2 h). Devil: D027 leak via per-model metrics printed during the test run.

**Genuine tensions.**
- GLM: drop (Simplicity) vs keep (Performance: cheap; Devil: it is GKX's evidence for the
  nonlinearity mechanism). Also basis reading (truncated splines vs as printed; 920 vs 94 inputs).
- NN seeds: 5 (Simplicity, Devil) vs 10 for Arm A if timing allows (Performance).
- Raw-macro sensitivity: a fork (Simplicity) vs a reported non-trial sensitivity (Devil).

**Not covered by anyone.** Whether the disk filled because of this session (it did not: the
project is ~4 GB) or something else on the machine; the user must decide what to free.
