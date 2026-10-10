# GKX replication

Replication and post-publication test of Gu, Kelly & Xiu (2020),
*Empirical Asset Pricing via Machine Learning*, Review of Financial Studies.

- Question, success criteria and locked splits: [`PREREGISTRATION.md`](PREREGISTRATION.md)
- Every design choice and why: [`DECISIONS.md`](DECISIONS.md)

## Phases

| Phase | Status |
|---|---|
| 0. Setup + pre-registration | done (`prereg-v1`) |
| 1. Data pipeline + leakage checks | done: CRSP + GKX datashare panel, sealed 2017+ holdout, OSAP map (D001–D026) |
| 2. Baselines | done: OLS-3, OLS-all (+Huber), training-mean benchmark |
| 3. Replication (linear, trees, NNs; walk-forward, multi-seed) | all 18 variants rehearsed on 1975–1986; frozen run list, verdict rule (D045) and launcher ready; single 1987–2016 run pending the `freeze-v1` tag |
| 4. Ablations + error analysis (pre-2017 only) | — |
| 5. Unseal post-publication holdout, final test, write-up | — |

## Running

```bash
uv run pytest -q -W error                       # main suite
GKX_TORCH=1 uv run pytest -q tests/nn           # NN tests (torch and LightGBM never share a process)
nohup bash scripts/run_test.sh --rehearsal >/dev/null 2>&1 &   # dry run on 1975-1986
uv run python scripts/final_report.py --rehearsal
```

The 1987–2016 test window is looked at once, from the `freeze-v1` tag
([`docs/freeze_checklist.md`](docs/freeze_checklist.md)); the verdict is printed mechanically by
`scripts/final_report.py` under D045.

## Setup

```bash
uv sync
cp .env.example .env   # add your WRDS username
```

Data comes from WRDS (CRSP) under an academic license and is never committed.
