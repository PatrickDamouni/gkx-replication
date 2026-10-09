# Phase 3 plan: the remaining models (draft for review)

Status: draft, written before any test-window look (D027). Model scope is fixed by D029; grids
follow GKX Table A.5 (read from the paper, `paper/gkx_table1.csv` + D030). Everything below is
developed on the 1975–1986 rehearsal schedule; the single 1987–2016 run happens only after the
code, grids and seeds below are frozen and tagged.

## Done in Phase 2 (rehearsal only)
- OLS-3, OLS-3+H: rehearsal R² ≈ +0.55%. OLS, OLS+H: −27,000% — diagnosed as macro-interaction
  extrapolation (D033). Linear solves exact via Kronecker statistics, min-norm on rank-deficient
  designs (D031), Huber IRLS converged in 5–9 iterations (D032).

## Models and grids (GKX Table A.5 unless noted)
| Model | Implementation | Tuning on validation | Notes |
|---|---|---|---|
| PCR | eigendecomposition of X'X from Kronecker stats | K components | GKX give no K range; proposal K ∈ {1, 2, …, 50} |
| PLS | SIMPLS from X'X and X'y (no rows needed) | K | same K grid |
| ENet+H | covariance-mode coordinate descent on X'WX, Huber via IRLS | λ ∈ (1e-4, 1e-1) log grid (10 values), ρ = 0.5 | penalty depends on feature scaling — see open question 1 |
| GLM+H | order-2 spline, 3 knots, group lasso, Huber | λ ∈ (1e-4, 1e-1) | breaks the Kronecker structure (splines of z); see open question 2 |
| RF | LightGBM `rf` mode, 300 trees | depth 1–6, features per split {3, 5, 10, 20, 30, 50} | raw chars + macro + sic2 (D012) |
| GBRT+H | LightGBM, Huber loss with α = ξ (99.9% rule) | depth 1–2, trees ≤ 1000 with validation early stopping, LR {0.01, 0.1} | D012 inputs |
| NN1–NN3 | PyTorch CPU; 32 / 32-16 / 32-16-8 ReLU, batch norm, Adam | LR {0.001, 0.01}, L1 λ ∈ (1e-5, 1e-3) | batch 10,000, ≤100 epochs, patience 5; inputs z (920) built per mini-batch |

## Seeds and ensembles
- GKX ensemble 10 seeds for NNs (Table A.5); D029 fixed 5. Open question 3.
- Deterministic models (PCR, PLS, ENet, GLM) run once; RF/GBRT/NN use the seed list.

## Harness changes needed
- Validation-based selection owned by the harness: each grid point is fit on train, scored on
  validation (R² on the validation window), the best is refit/used for test; every grid point's
  validation score is logged (council 4 Performance).
- Deflated-Sharpe trial count = configs × seeds of candidate models in Arm A (D028).
- Prediction retention: keep test-window predictions for selected configs and all seeds; keep
  only validation scores for unselected grid points (disk ~12–17 GB free).
- Libraries to add and pin: lightgbm, torch (CPU).

## Compute budget (to be measured on one rehearsal fold per model before freezing)
- Linear family: seconds per fold.
- LightGBM RF/GBRT: estimate per fold after one timed rehearsal fold.
- NN1–NN3: council 4 estimate ≈ 1–2 min per training; 4 configs × seeds × 30 folds per architecture.

## Open questions for review
1. Feature scaling for penalized models and NNs: chars are rank-normalized; macro raw vs scaled with
   training-window constants changes ENet/GLM/NN fits (OLS/Huber unaffected, D030). Which is
   defensible as faithful to GKX?
2. GLM+H: splines of all 920 z columns (≈3,700 columns, no Kronecker shortcut) vs drop GLM from scope
   vs a documented approximation.
3. NN seeds: 10 (GKX) vs 5 (D029) given the compute budget.
4. Anything in the Phase 2 rehearsal findings (D031–D033) that should change before freezing the
   linear family.
5. Ordering: what must be measured on rehearsal before the freeze tag, and what is the exact freeze
   checklist?
