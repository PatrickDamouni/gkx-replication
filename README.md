# GKX replication

Replication and post-publication test of Gu, Kelly & Xiu (2020),
*Empirical Asset Pricing via Machine Learning*, Review of Financial Studies.

- Question, success criteria and locked splits: [`PREREGISTRATION.md`](PREREGISTRATION.md)
- Every design choice and why: [`DECISIONS.md`](DECISIONS.md)

## Phases

| Phase | Status |
|---|---|
| 0. Setup + pre-registration | done (`prereg-v1`) |
| 1. Data pipeline + leakage checks | — |
| 2. Baselines | — |
| 3. Replication (linear, trees, NNs; walk-forward, multi-seed) | — |
| 4. Post-publication holdout + ablations | — |
| 5. Write-up | — |

## Setup

```bash
uv sync
cp .env.example .env   # add your WRDS username
```

Data comes from WRDS (CRSP) under an academic license and is never committed.
