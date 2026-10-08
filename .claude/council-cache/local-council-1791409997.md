# Local council — Phase 1a test seams

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Question: review the five proposed Phase 1a seams (walk_forward, r2_oos,
diebold_mariano, long_short, evaluate-on-synthetic-panels).
Members: Devil's Advocate, Simplicity Champion, Maintainability Advocate,
Performance Optimizer, Developer Experience.

---

## 🗳️ Devil's Advocate

**Position.** The seams test arithmetic, the part least likely to be wrong. Timing,
alignment and normalization sink GKX replications and are covered by one planted
canary. Two end-to-end assertions are statistically wrong, and the primary prereg
criterion (95% CI on R²_oos) has no seam.

**Key points.**
- Holdout seal is at the wrong layer: osap.py writes one Parquet through 202510; EDA,
  pooled winsorization/imputation, dropna would touch 2017+ without calling
  walk_forward. Split at ingest into pre2017/holdout files; record holdout sha256.
- Replace the 5(d) canary with a perturb-the-future invariance test (bit-identical
  predictions at t when everything after t is randomized). Planted-leak + high-R²
  threshold is circular; real leaks add ~0.1% R².
- Missing seam: align(signals, returns) — signal at t → excess return at t+1; needs rf.
  Testable on OSAP data before WRDS.
- 5(a) wrong: fitted models on noise score *below* 0 under GKX's metric; assert ≤ 0.
  5(c) flaky at realistic SNR and tests models that don't exist yet — cut.
- Use synthetic panels for a power analysis: with true R² ≈ 0.4%, T≈360, can the
  DM test / R² CI reject? If not, the locked criterion is unreachable.

**Risks & blind spots.** DM must run on the monthly cross-sectional mean loss
differential. R² CI needs a month-block bootstrap; prereg says "monthly R²" but GKX
pools (i,t) — log a deviation. GKX ensemble NN predictions across seeds; prereg's
mean ± spread of single seeds understates NNs. VW weights must use ME at formation.
Synthetic panels hide schema bugs (delistings, permno reuse, NaNs) — add a small
CRSP-shaped fixture. DSR is gameable without an automatic trial ledger.

**Confidence.** high on timing/alignment/seal, DM level, missing R² CI; medium on cutting 5(c).

---

## 🗳️ Simplicity Champion

**Position.** Seams 1–3 are about right; 4 is overloaded; 5 tests models that don't
exist while skipping the riskiest step (join + normalization). Cut 5 to deterministic
baselines plus one invariance test; add a panel-alignment seam.

**Key points.**
- Perturbation-invariance replaces "detect a leaky feature" — exact, no threshold.
- Missing seam: align (t→t+1) and per-month rank_normalize; both pure, testable now.
- Cut 5(c). Use zero forecast (R²=0 exactly), oracle (analytic R²), OLS-3 on planted
  signal. Noise "≈ 0" is wrong; assert ≤ 0 + tol. Use sklearn fit/predict, no custom protocol.
- Split seam 4; one shared month-bootstrap for R² CI and Sharpe CI. Defer DSR to Phase 3.
- DM: test the GKX cross-sectional aggregation; use statsmodels HAC instead of
  hand-rolled Newey-West.

**Risks & blind spots.** Seal belongs at load time; skip the unseal flag until Phase 5
(YAGNI). Check the validation window rolls, not expands. ~8 sharp deterministic tests
read as more rigorous than an evaluate() framework. Don't mock WRDS. Seed aggregation
is a groupby, not a seam.

**Confidence.** high.

---

## 🗳️ Maintainability Advocate

**Position.** Right functions, wrong boundaries. Leakage lives in the feature/label
join and per-month transforms, which have no seam; R² CI and ≥5-seed rule have no code.

**Key points.**
- Add build_panel(chars, returns) + validate_panel (unique (permno, month), explicit
  `ret_excess_next` target, VW weight = ME at formation). Other seams take a Panel.
- One generic future-invariance helper reused for rank-normalize/impute,
  walk_forward and evaluate.
- Signatures: DM takes a month-indexed Series and raises on duplicate months;
  r2_oos documents pooling; docstrings cite the paper section.
- Split seam 4; one block bootstrap shared with R².
- Seam 5: Oracle/Zero/Cheat fake models; move model sanity tests to Phase 3.

**Risks & blind spots.** Seal in the single loader; unsealing leaves a timestamped
trail. Deflated Sharpe needs an append-only trials ledger, which also enforces ≥5 seeds.
The synthetic generator needs its own tests. Split CRSP code into I/O vs. pure
transform and test the transform on a ~10-row fixture with a delisting. osap.py uses
undeclared `requests` and private `ap._get_url`.

**Confidence.** high structural; medium on exact DM form and OSAP lag convention.

---

## 🗳️ Performance Optimizer

**Position.** Missing a seam proving the experiment fits this laptop. Without it, the
first real run OOMs or takes weeks, and the workarounds become unlogged forking paths.

**Key points.**
- Scale/budget test: synthetic float32 panel ~4M×200 (~3.2 GB), one refit at smallest
  and largest window, assert peak RSS < ~12 GB, extrapolate wall time to full grid.
- walk_forward should return integer row ranges on a (yyyymm, permno)-sorted panel →
  zero-copy slices. Seal via pyarrow filter pushdown at read time.
- Per-month rank-normalize is month-local → compute once, cache; column-wise into
  preallocated float32. Anything fit across time must be refit per window on train only.
- Metrics consume per-month summaries (SSE_t, SST_t, n_t, decile returns), so DM and
  bootstraps run on T=360, not 2M rows. Bootstrap resamples months, vectorized, seeded.
- Mark tree tests slow; keep TDD loop < 5s. Trial ledger doubles as resumable
  prediction cache (persist predictions, never models).

**Risks & blind spots.** GKX macro interactions (~1,800 features × 4M rows ≈ 29 GB)
won't fit — decide now: batch-compute inside the model or drop and log a deviation.
Disk: delete the zip after conversion. Seeds vs. configs in the ledger. float32
storage, float64 metrics. Prereg "monthly R²" wording vs. pooled.
(Claim that the download `.part` was stuck at 1.09 GB was wrong — it was progressing.)

**Confidence.** high on arithmetic; fit-time extrapolations are estimates.

---

## 🗳️ Developer Experience

**Position.** Seams roughly right, but contracts (units, levels, error types) are
loose enough that the author couldn't debug their own failures. Fix contracts first.

**Key points.**
- Seal in one load_panel() raising a dedicated SealedHoldoutError with an actionable
  message; tests use pytest.raises(SealedHoldoutError).
- Fix month type (pd.Period('M')) at the boundary; frozen Fold dataclass with readable
  repr; committed text snapshot of the 30-fold schedule; PREREG_SPLITS constant.
- Missing seam: make_panel lag + per-month rank normalization; one
  assert_future_invariant helper.
- Split seam 4; test DSR against Bailey & López de Prado's worked example; VW uses
  ME at t−1; r2_oos pooled vs. monthly must be explicit; DM arg named monthly_loss_diff.
- Seam 5 with scripted models (oracle with analytic R² returned by the generator,
  cheater must raise LookaheadError structurally); evaluate returns an EvalReport.

**Risks & blind spots.** Noise test must be one-sided on mean-zero noise. No seam for
the R² CI or seed rule. Test names should read as claims and cite DECISIONS.md IDs.
(Claim of a "stale" `.part` file was wrong — it was the in-progress download.)

**Confidence.** medium-high.

---

## Synthesis — angles, not consensus

**Shared starting points (a common prior to stress-test, not corroboration).** All five
said: leakage risk lives in the signal→return alignment and per-month normalization,
which had no seam; the holdout seal belongs at load time, not in walk_forward; seam 4
should be split with one shared month-block bootstrap; the R² CI (the primary prereg
criterion) had no seam; DM must run on the monthly cross-sectional mean loss
differential; 5(c) "trees beat OLS" tests Phase 3 models and should go. Because all five
are Claude, they would share the same blind spot — e.g. all leaned on GKX conventions
from memory (DM aggregation, NN seed ensembling, 1/4/6-month lags) that should be
checked against the paper before being encoded in tests.

**Genuine tensions.**
- Simplicity (sklearn fit/predict, no unseal flag, defer ledger) vs. DX/Maintainability
  (SealedHoldoutError, Fold dataclass, EvalReport, ledger now). Given the interview
  goal, cheap readability wins (custom error, Fold repr); framework-y pieces
  (EvalReport, ledger) wait until real trials exist.
- Performance's scale test vs. Simplicity's "≈8 sharp tests". The disk/RAM situation
  makes the scale test worth one slow-marked test.

**Blind spots.** No member covered the stock universe definition (CRSP share codes
10/11, exchanges NYSE/AMEX/NASDAQ) or which risk-free series defines excess returns
beyond naming it. Devil's power-analysis point is unique and important: it checks
whether the locked success criterion is even reachable.

**Prereg clarifications needed (ex ante, logged as deviations):** pooled vs. monthly
R²_oos; seed estimand (ensemble vs. single-seed spread); macro interactions kept or dropped.
