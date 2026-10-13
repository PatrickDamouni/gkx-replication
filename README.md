# GKX replication

**Result.** A pre-registered reproduction of Gu, Kelly & Xiu (2020) on their own characteristics,
tested in two windows, each looked at once:
- **1987–2016:** GKX's model ranking did not reproduce; a 5-seed NN3 ensemble beat a constant
  forecast (ΔR² +0.27 pp, p = 0.013, not Holm-significant), with the edge in microcaps and before 2003.
- **2017–2021 (sealed holdout, 60 months):** the one test fixed and published before unsealing
  did not detect an edge (t = 1.50, p = 0.066; power ≈ 30% if the effect had halved).


- Question, success criteria and locked splits: [`PREREGISTRATION.md`](PREREGISTRATION.md)
- Every design choice and why: [`DECISIONS.md`](DECISIONS.md)
- **Both results, written up: [`WRITEUP.md`](WRITEUP.md)**

## Phases

| Phase | Status |
|---|---|
| 0. Setup + pre-registration | done (`prereg-v1`) |
| 1. Data pipeline + leakage checks | done: CRSP + GKX datashare panel, 2017+ holdout sealed until Phase 5, OSAP map (D001–D026) |
| 2. Baselines | done: OLS-3, OLS-all (+Huber), training-mean benchmark |
| 3. Replication (linear, trees, NNs; walk-forward, multi-seed) | done: single 1987–2016 run from `freeze-v1` (2026-10-11); pre-specified verdict under D048 **partially replicates** — 5-seed NN3 ensemble R² 0.454% [0.165, 0.733] beats a training-mean forecast (p = 0.013, does not survive Holm); GKX's model ranking not reproduced. [`results/final_report.md`](results/final_report.md). Primary data is GKX's own characteristics (D022), not an independent build |
| 4. Ablations + error analysis (pre-2017 only) | done: exploratory zero-refit diagnostics only (D049); write-up |
| 5. Unseal post-publication holdout, final test, write-up | done: single look from `freeze-v2`, 2017-01..2021-12 (60 months); H1 t = 1.503, p = 0.066, not rejected (D050–D054); [`results/phase5_report.md`](results/phase5_report.md), tag `result-v2` |

## Running

```bash
uv run pytest -q -W error                       # main suite
GKX_TORCH=1 uv run pytest -q tests/nn           # NN tests (torch and LightGBM never share a process)
nohup bash scripts/run_test.sh --rehearsal >/dev/null 2>&1 &   # dry run on 1975-1986
uv run python scripts/final_report.py --rehearsal
```

The 1987–2016 test window is looked at once, from the `freeze-v1` tag
([`docs/freeze_checklist.md`](docs/freeze_checklist.md)); the verdict is printed mechanically by
`scripts/final_report.py` under D045 as amended by D047–D048.

## Setup

```bash
uv sync
cp .env.example .env   # add your WRDS username
```

Data comes from WRDS (CRSP) under an academic license and is never committed.
