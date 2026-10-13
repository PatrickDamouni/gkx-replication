# Local council 17 — review of the Phase 5 single look (46e7466, result-v2)

> **Local council** — these perspectives all come from Claude playing different
> roles, not from different AI vendors. Treat agreement as a shared starting
> point to pressure-test, not as independent confirmation.

`--roles=devil,simplicity,maintainability,dx`; read-only.

## 🗳️ Devil's Advocate
Look is clean and per plan (ledger 1121–1274 audited; canary; train_mean folds; DM settings). PLS =
constant (K=0 every fold) → test 3 ≡ test 1. Forbid: "edge disappeared", "NN beats constant",
"marginal" (size 6.1%), holdout-vs-1987 R² levels (constant moved 0.18→0.27; a ΔR² comparison is a
decay ratio in disguise), "penalized linear beats OLS" (both trail the constant), "OLS gap resolved",
counting signs. Say "post-sample". D054: facts, verbatim reading, PLS, p "0" erratum, why 202112,
window spent, no H1 variants. WRITEUP: fix title, §1 "not opened", §7, and the false freeze-v1 diff
claim. Risks: external timestamp decays; 2020 confound temptation; Arm B pooling.
## 🗳️ Simplicity Champion
Correct and simple; stop here. One honest sentence for the project; D054 ≈ 10 lines; README/WRITEUP:
one holdout bullet + short section, fix stale lines, disclose HAC 5 vs 3. Resist HAC-3, equal
weighting, splits, decay ratio, Arm B.
## 🗳️ Maintainability Advocate
Trail verified (append-only ledger; 1 unseal, 8 inputs, 18 runs, 108 folds, 1 look; canary via
fold_sha256 grep). Record untraced D053 steps; record tags via ls-remote; fallback counts for the -dm
NNs (bear on H1); fix the false "git diff freeze-v1 HEAD" claim; manifest extension after the
WRDS-terms note; dry-run file contains the "reject" sentence for 360 months — annotate, don't edit.
## 🗳️ Developer Experience
Docs stale ("has not been opened", Phase 5 "—"). Two-window README headline; holdout item near the
top of WRITEUP; one tests table, no figure; footnote the report's quirks; never regenerate the
report; banned wordings; put advance commitments beside the result.

## Synthesis — angles, not consensus
**Shared:** clean, per plan; record and stop; no H1 variants; no level comparisons; fix stale docs.
**Unique:** Maintainability — fallback counts (verified: NN1-dm 21/25, NN2-dm 11/25, NN3-dm 6/25);
Devil — "post-sample", timestamp decay. **Applied:** D054; WRITEUP §3 (holdout), result item 3,
title, §1, §7–§8, checks; README two-window headline and phase row.
