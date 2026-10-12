# Phase 5 report — DRY RUN on 1987–2016 (not the holdout)

Code `0541701`; the freeze-v1 test predictions play the holdout, so every 1987–2016 column equals its own value.

## H1 — the one confirmatory test

NN3-dm vs the constant, one-sided Diebold–Mariano: t = 4.569, p = 0.0000 (reject at α = 0.05); 1987–2016: t = 4.569.

**On GKX's characteristics, the demeaned-target NN3's within-month edge over a constant survived the first post-publication years (Arm A, 360 months). This is not evidence that machine learning beats linear models.**

## Reported, deciding nothing

Bootstrap CIs use 12-month blocks over 360 months, so they are coarse. No D048 label and no decay ratio are reported (D050 §9).

### R² by model

| model      |      R2 % | R2 95% CI           |   1987–2016 R2 % |
|:-----------|----------:|:--------------------|-----------------:|
| OLS-3      |    0.2192 | [0.004, 0.428]      |           0.2192 |
| OLS-3+H    |    0.2101 | [0.016, 0.400]      |           0.2101 |
| OLS        | -170.026  | [-463.782, -17.699] |        -170.026  |
| OLS+H      | -169.98   | [-465.870, -16.769] |        -169.98   |
| PCR        |    0.2951 | [0.052, 0.517]      |           0.2951 |
| PLS        |    0.2841 | [0.043, 0.505]      |           0.2841 |
| ENet+H     |    0.3029 | [0.086, 0.512]      |           0.3029 |
| GLM+H      |    0.353  | [0.110, 0.570]      |           0.353  |
| GBRT+H     |    0.089  | [-0.123, 0.312]     |           0.089  |
| GBRT+H-dm  |    0.6721 | [0.413, 0.910]      |           0.6721 |
| RF         |    0.2774 | [0.062, 0.483]      |           0.2774 |
| RF-dm      |    0.7028 | [0.421, 0.958]      |           0.7028 |
| NN1        |    0.5142 | [0.198, 0.827]      |           0.5142 |
| NN1-dm     |    0.5569 | [0.245, 0.844]      |           0.5569 |
| NN2        |    0.4526 | [0.163, 0.743]      |           0.4526 |
| NN2-dm     |    0.6182 | [0.321, 0.874]      |           0.6182 |
| NN3        |    0.4539 | [0.165, 0.733]      |           0.4539 |
| NN3-dm     |    0.6566 | [0.373, 0.908]      |           0.6566 |
| train_mean |    0.1814 | [-0.033, 0.396]     |           0.1814 |

### The seven D045 tests (test 1 had ≈ 23% power at 60 months)

| candidate   | vs         | test   |       t |   p (one-sided) |   ΔR2 % | 95% CI           |   1987–2016 t |   1987–2016 p |
|:------------|:-----------|:-------|--------:|----------------:|--------:|:-----------------|--------------:|--------------:|
| NN3         | train_mean | DM     |  2.213  |          0.0134 |  0.2725 | [0.045, 0.499]   |        2.213  |        0.0134 |
| NN3         | PCR        | DM     |  1.4716 |          0.0706 |  0.1589 | [-0.026, 0.359]  |        1.4716 |        0.0706 |
| NN3         | PLS        | DM     |  1.6633 |          0.0481 |  0.1698 | [-0.010, 0.356]  |        1.6633 |        0.0481 |
| NN3         | ENet+H     | DM     |  1.3194 |          0.0935 |  0.151  | [-0.052, 0.362]  |        1.3194 |        0.0935 |
| NN3         | GLM+H      | DM     |  0.9447 |          0.1724 |  0.1009 | [-0.087, 0.301]  |        0.9447 |        0.1724 |
| GBRT+H      | ENet+H     | DM     | -1.9154 |          0.9723 | -0.214  | [-0.431, -0.004] |       -1.9154 |        0.9723 |
| ENet+H      | OLS-3+H    | CW     |  5.9517 |          0      |  0.0928 | [-0.028, 0.205]  |        5.9517 |        0      |

### H1 and secondary tests

| test                                  |      t |   p (one-sided) |   1987–2016 t |   1987–2016 p |
|:--------------------------------------|-------:|----------------:|--------------:|--------------:|
| H1: NN3-dm vs constant (confirmatory) | 4.5689 |           0     |        4.5689 |         0     |
| NN3-dm vs ENet+H, within-month errors | 4.0366 |           0     |        4.0366 |         0     |
| ENet+H vs constant                    | 1.6254 |           0.052 |        1.6254 |         0.052 |
