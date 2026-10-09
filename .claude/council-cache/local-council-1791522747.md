# Local council 6 — GLM+H and Phase 3 so far (before trees and NNs)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

Reviewed: src/gkx/glm.py, enet.py, reduction.py, linear.py, evaluate.py; D034–D039; rehearsal
results. Members (`--roles=devil,simplicity,performance`). All checks used data ≤ 1986 only.

---

## 🗳️ Simplicity Champion
**Position.** GLM+H and Phase 3 are correct, faithful and on track — freeze them; don't widen grids
or keep tuning on rehearsal; spend effort on one LightGBM wrapper, one NN class, one runner.
**Key points.** GLM+H checked against eq. 14 and Table A.5 (IRLS weights, λ/L threshold with L valid
for w ≤ 1, KKT for active and zero groups, ξ shared with ENet). λ at the grid edge 1e-4 in 4/12
folds: do not widen (Table A.5 fixes the range); log it. Rehearsal R² no longer separates the
linear models (most fall back to the intercept). Duplication: leave enet/glm's ~60 shared lines;
add one shared ξ helper. Trees: one LGBM class, one Dataset per fold reused across the grid,
`metric="l2"` explicit (else early stopping uses the Huber metric), sic2 categorical. NNs: 6 configs
(2 LR × 3 λ₁), not 4; time one fold, then decide; allowed fallback = select on seed 0, train seeds
1–4 at that config (pre-register).
**Risks.** docs/phase3_plan.md is stale (coordinate descent, K 1–50, 4 NN configs) — cut it to a
checklist. Merge the runner scripts. No crash recovery — one invocation per model. Trees see 103
inputs vs NNs 920 — state it. Thread/determinism flags still unset.
**Confidence.** high.

## 🗳️ Devil's Advocate
**Position.** GLM+H is correct, but D037's macro standardization — not in the paper — drags every
920-feature model to the training-mean level; reverse it before trees/NNs are built.
**Key points.** Scratch check (rehearsal only, no ledger): with raw macro (GKX eq. 21's x_t),
PCR +0.320% → +0.760% with K 9–67 (Fig. 3's 0–100 axis); PLS K=0 everywhere → +0.728% with K 2–9
(Fig. 3's 0–10 axis); ENet+H +0.462% → +0.718%. Mechanism: centering turns c·m into c·(m−μ),
giving the interaction directions (which D033 showed explode) equal footing with the
characteristics; raw c·dp stays close to a rescaled characteristic, acting as a prior. Raw-macro
rehearsal ordering matches GKX Table 1's. GLM: no bugs; objective scaling matches eq. 6; D035's
truncated basis is an interpretation (literal eq. 13 is a quadratic). Grid-edge hits are an era
artifact (80% of 1957–62 characteristic values are imputed 0 vs 34% in 1975–86) — do not widen;
pre-register edge-hit reporting. Pin LightGBM details (½-scaled Huber, sic2 encoding,
determinism, rf bagging, min_data_in_leaf/max_bin defaults as undocumented regularization).
**Risks.** Forking path: defensible only because the paper text supports raw and the test window is
untouched — log both runs. ξ from in-sample OLS-all residuals is biased low and binds on ~0.1% of
rows, so "+H" is nearly cosmetic — say so. Report demeaned R² with every claim. GLM reading gap.
**Confidence.** high on the macro finding.

## 🗳️ Performance Optimizer
**Position.** The 30-fold test run fits in one night (~8–14 h); robustness, not speed, is the risk —
LightGBM and torch deadlock in one process, and evaluate() has no per-fold checkpointing.
**Key points (measured; ≤1986 rows, tiled ×8 to 3.32M):** PCR 1.4 → 5.4 s; ENet+H 3.2 → 17 s; GLM+H
12.7 → 135 s (peak RSS 7.3 GB; 117 s in `_apply` rebuilding the spline design — a factored version
is 0.67 s/call, matches to 2e-14, optional). LightGBM: binning 0.9 s; GBRT+H 18–23 ms/tree; RF 69–79
ms/tree at depth 6; 4 threads only 1.4–1.6× slower than 12 → RF grid ≈ 2–3 h for 30 folds, GBRT ≈
20–30 min. NN: 3.3–3.9 ms/step with a factored first layer (never materialize z: 12 GB); real fold
1986 early-stops at epochs 7–27; 2,700 trainings ≈ 9–10 h serial, ≈ 4–5 h with ~2× concurrency;
stay on CPU (MPS slower, less deterministic).
**Must-haves before unlock:** separate processes for trees and NNs (three libomp copies; any
multi-threaded LightGBM + torch op in one process hangs at 0% CPU — reproduced 4/4 orders);
per-(model, fold) persistence with resume; heartbeat log + watchdog; `caffeinate -i` on AC;
pinned threads/determinism recorded in the ledger; full 12-fold rehearsal of RF/GBRT/NN first.
**Risks.** Tiling doesn't change iteration/epoch counts on real 1957–2016 data. load_panel peaks at
3.0 GB for a 0.59 GB frame — many workers would OOM; dump features once to a float32 .npy and mmap.
Thermals on multi-hour runs.
**Confidence.** medium.

---

## Synthesis — angles, not consensus

**Shared starting points (common prior — stress-test).** All three judge GLM+H's math correct and
advise against widening the λ grid. Two (Simplicity, Devil) say rehearsal can find bugs and
degeneracies but cannot rank models.

**Verified after the council (by the main session, through the real harness, logged, clean
commit):** raw-macro rehearsal PCR +0.760% (K 20,29,20,60,63,9,12,66,64,67,51,57), PLS +0.728%
(K 2–9), ENet+H +0.718% — reproducing Devil's numbers exactly. Within-month demeaned R² (D036):
standardized PCR −0.172%, PLS 0.000%, ENet+H −0.000% vs raw-macro PCR +0.334%, PLS +0.298%,
ENet+H +0.292% (GLM+H, no macro, +0.267%; OLS-3 +0.088%). Standardization removed the models'
cross-sectional ranking skill, not just level.

**Genuine tension.** Reversing a pre-registered rule (D037) on rehearsal evidence is a forking
path (Devil names it) vs keeping a rule the paper never stated that demonstrably degenerates the
models. Mitigations: the paper text (eq. 21) supports raw; the test window is untouched; both
variants are logged.

**Unique.** Performance's LightGBM/torch deadlock (must run in separate processes) and the
checkpointing requirement are freeze blockers no one else raised.
