# Project phases

How to read the history: every design choice is an entry in [`DECISIONS.md`](../DECISIONS.md)
(D001–D055). Tags mark the points that matter — `prereg-v1` (pre-registration), `freeze-v1`
(code for the single 1987–2016 run), `result-v1`, `phase5-prereg` (holdout plan), `freeze-v2`
(holdout code), `phase5-panel` (holdout panel hash), `result-v2` (holdout result). Many commits
reference AI review rounds ("councils"); their records are in `.claude/council-cache/`.

## Status

| Phase | Status |
|---|---|
| 0. Setup + pre-registration | done (`prereg-v1`) |
| 1. Data pipeline + leakage checks | done: CRSP + GKX datashare panel, 2017+ holdout sealed until Phase 5, OSAP map (D001–D026) |
| 2. Baselines | done: OLS-3, OLS-all (+Huber), training-mean benchmark |
| 3. Replication (linear, trees, NNs; walk-forward, multi-seed) | done: single 1987–2016 run from `freeze-v1` (2026-10-11); pre-specified verdict under D048 **partially replicates** — 5-seed NN3 ensemble R² 0.454% [0.165, 0.733] beats a training-mean forecast (p = 0.013, does not survive Holm); GKX's model ranking not reproduced. [`results/final_report.md`](results/final_report.md). Primary data is GKX's own characteristics (D022), not an independent build |
| 4. Ablations + error analysis (pre-2017 only) | done: exploratory zero-refit diagnostics only (D049); write-up |
| 5. Unseal post-publication holdout, final test, write-up | done: single look from `freeze-v2`, 2017-01..2021-12 (60 months); H1 t = 1.503, p = 0.066, not rejected (D050–D054); [`results/phase5_report.md`](results/phase5_report.md), tag `result-v2` |

