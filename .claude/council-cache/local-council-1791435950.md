# Local council 2 — seams 1–3 code + seams 4–6 design

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Members: Devil's Advocate, Simplicity Champion, Maintainability Advocate, Performance
Optimizer, Developer Experience. Three members (Devil, Maintainability, Performance)
hit an API rate limit on the first run and were re-run after the reset; by then seam 3
(walk_forward) had been committed, so they reviewed seams 1–3 while Simplicity and DX
reviewed seams 1–2 plus the seam-3 code that existed when they read the repo.

---

## 🗳️ Developer Experience
**Position.** Seams 1–2 are readable; the main DX risk is `yyyymm` meaning signal month in
some places and return month in others, which breaks two planned pieces.
**Key points.** (1) *Confirmed by running:* a CheatModel that copies its own row's
`ret_excess_next` passes `assert_future_invariant` — the label sits on row t, which is never
scrambled. A survivorship filter also passes (row presence isn't perturbed). Fix: track
information time; add a drop-future-rows mode; prefer structural controls (predict only
receives X). (2) Fold months must be declared label months; otherwise the 2016-12 signal
needs a sealed 2017-01 return and silently drops. (3) Float32: `rank_normalize` upcasts to
float64 and copies the frame; the invariance helper emits a pandas FutureWarning on float32.
(4) Planned `evaluate(model, panel, folds)` contradicts D007 (separate macro) and sklearn
`fit(X, y)` has no slot for the validation window. (5) `seal()` doesn't delete the zip;
rerunning ingest re-creates an unsealed copy; downloaded hash never compared; holdout sits
beside pre-2017 file (directory read bypasses `load_panel`) — move to `data/sealed/`.
**Risks.** `ru_maxrss` units/process-lifetime; scale-test `rank_normalize`, not a
nonexistent model; pooled R² (N_t-weighted) vs DM (equal-weighted) can disagree — pre-commit;
FutureLeak errors should name the first differing cell; power analysis as a fixed-seed
result in `results/`, not a flaky assertion.
**Confidence.** high (cheat blind spot, float32 warning, upcast, survivorship all run).

## 🗳️ Simplicity Champion
**Position.** Seams 1–3 right-sized; one real bug; several seam 4–6 pieces duplicate,
can't catch what they target, or test code that doesn't exist.
**Key points.** (1) Key the modeling panel on *return month* inside `align`; add a test that
the max label month across folds is 201612. Store month ranges, not row slices. (2) Make the
cheat impossible rather than detected: `evaluate` builds X from an explicit feature list.
Keep the oracle — it catches row-order/merge bugs. (3) Pooled R² > 0 ⇔ mean_t(d_t) > 0 with
d_t = Σr² − Σ(r−r̂)², so one `hac_mean_test` covers "beats zero" and the R² CI; bootstrap
only for Sharpe. Use per-month *sums* so the test matches pooled R². Pin HAC lag and block
length now. (4) Cut seam 6; fix `rank_normalize` memory now (float32, vectorized).
(5) 209×9 interactions drive the memory complexity — mapping to ~94 GKX characteristics is
more faithful and halves width.
**Risks.** Power analysis as a script + DECISIONS entry. Rank after the CRSP universe
filter. Invariance never adds/removes rows. CRSP pull needs its own seal.
**Confidence.** high.

## 🗳️ Maintainability Advocate
**Position.** Time semantics are undocumented; the cheat-check bug follows; several seam
plans contradict logged decisions.
**Key points.** (1) *Confirmed:* cheat passes invariance; add as negative control; or ensure
labels never reach predict. (2) Fold month unit unspecified; test the 201612 boundary.
(3) Harness, not models, should own fit→select-on-val→predict and seed averaging (D006),
and take macro separately (D007). (4) Start the trial ledger now (~10 lines). (5) Persist
sums (sse, Σr², n_t), not means; pin maxlags, use_t, block length, Sharpe annualisation;
keep a raw `me` column outside features for VW.
**Risks.** Phase numbering conflict (prereg: sealed until Phase 5; README: holdout =
Phase 4). D008 cites a gitignored file. No `unseal()` verifying the hash. Missing tests:
align negative control (t+2), rank ties/one-firm months, osap.py.
**Confidence.** high.

## 🗳️ Devil's Advocate
**Position.** Code is clean, but the core leakage check has a blind spot exactly where
seam 5 relies on it, the fold month axis collides with the seal, and several seam 4–6
choices guard against the wrong failure.
**Key points.** (1) *Confirmed:* cheat passes invariance. Structural fix: evaluate strips
the label; a spy model asserts max month seen by fit ≤ fold.train end. (2) *Confirmed:*
invariance false positive — scrambling an int column upcasts the whole column, so an
identity step fails (warning now, error in pandas 3). (3) Fold month axis (as others).
(4) Zero forecast and OLS-3 are nested → plain DM is undersized; use Clark–West for nested
comparisons; DM on cross-sectional mean weights months equally vs pooled R² by N_t
(6,937–10,073 OSAP firms/month); the prereg criterion is a *ranking* — build a paired ΔR²
bootstrap CI. (5) Ledger from day one; every evaluate() on 1987–2016 is a look at the test
window — consider developing only on fold 1's validation window until frozen.
**Risks.** Trees can't batch ~1,800 interaction features → deviation needed now. Power
analysis optimistic (estimation error, fat tails, model gaps ~0.1–0.2%); pre-register the
response if power < 80%. ZeroModel R²=0 is true by construction. GKX linear baselines use
Huber loss ("+H"). OSAP universe ≈ 7–10k permnos vs GKX ≈ 6.2k — filter then rank. Hash proves
unchanged, not unread; hash depends on pyarrow version; `split_holdout` lacks a row-count
check. Checked: every OSAP signal's publication year ≤ 2016.
**Confidence.** high.

## 🗳️ Performance Optimizer
**Position.** Leakage design is sound; memory/compute is what breaks this.
**Key points.** (1) *Measured on the real file:* `rank_normalize` returns float64 and copies;
20 cols: RSS 1.31→2.48 GB; 60 cols: 3.25→5.87 GB; 0.49 s/col; full width projects to
~13–14 GB peak. Fix: float32, no full copy, run once after universe filter and ≥195703;
save a (yyyymm, permno)-sorted float32 memmap + month-offset index → zero-copy fold slices.
(2) Parquet is permno-sorted; every row group spans all months → no filter pushdown.
(3) D007 conflicts with `fit(X, y)`: ~4.08M × ~1,950 ≈ 32 GB float32; sklearn HGB casts to
float64 (verified in sklearn 1.9.1). Linear: per-year Gram matrices make the λ grid free;
NN: contract interactions in the first layer; trees: raw chars + macro (deviation).
(4) sklearn early stopping uses a random 10% of train, ignoring the chronological val
window. Seed only stochastic models. NN budget ≈ hundreds of CPU-hours. (5) Metrics on
per-month summaries; one shared bootstrap index matrix; keep per-month summaries for every
config (DSR needs them), stock-level predictions only for selected configs.
**Risks.** Seam-6 synthetic test misses the real drivers; macOS compresses/swaps instead of
OOM-killing — assert wall time too. Disk tighter than stated. Cheat bug; fold month bug.
Row identity across models for DM. Universe defined by "has t+1 return" uses future survival.
**Confidence.** high (measured / verified).

---

## Synthesis — angles, not consensus

**Shared starting points (common prior — stress-test, don't treat as corroboration).**
All five independently found (a) the CheatModel can't fail `assert_future_invariant`
(four reproduced it by running code), and (b) Fold windows are ambiguous between signal
and return months, which silently drops Dec-2016 against the seal. Four flagged
`rank_normalize`'s float64 upcast + copy (Performance measured it). Because all are Claude,
the shared blind spot would be GKX conventions recalled from memory (Huber "+H" baselines,
~6.2k firms/month, DM usage) — verify against the paper before encoding.

**Genuine tensions.**
- Simplicity (cut seam 6, keep `evaluate(model, X, y, folds)` small, map to ~94 chars) vs
  Performance/Maintainability (harness owns val selection, seeds, macro; memmap artifact).
  Resolution: interface must carry val + macro (D006/D007 require it), but build it only
  when the first real model needs it; replace seam 6 with a real-path measurement script.
- Simplicity's "one HAC test covers R² CI" vs Devil's "nested models need Clark–West".
  Both are right in different places: Clark–West for nested (vs zero, OLS-3 vs OLS-all),
  paired ΔR² bootstrap for the pre-committed ranking.

**Blind spots.** No member checked whether the OSAP panel already applies any universe
filter, or how delisting returns will enter `ret_excess` before alignment.

**Suggested direction.** Fix the confirmed bugs before seams 4–6: return-month keying,
structural label isolation, dtype-safe invariance with a drop-future-rows mode,
float32 `rank_normalize`, row-count check in `split_holdout`, move holdout to
`data/sealed/`, phase-number consistency, commit council records. Start the trial ledger.
Make power analysis and scale measurement scripts with committed results. Log the
fidelity decisions (nested tests, Huber, tree interactions, feature set) before data.
