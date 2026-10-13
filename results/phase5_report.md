# Phase 5 report — sealed holdout, Arm A (D050)

Holdout predictions from `freeze-v2` (`0541701`); holdout panel `6bdbd52861ef…` (published as tag phase5-panel); test months 201701–202112 (60). Canary fold 2016: 18/18 bit-identical to freeze-v1.

## H1 — the one confirmatory test

NN3-dm vs the constant, one-sided Diebold–Mariano: t = 1.503, p = 0.0664 (not rejected at α = 0.05); 1987–2016: t = 4.569.

**Not detected (Arm A, 60 months). With ≈ 30% power at half the 1987–2016 effect, this is weak evidence of decay, not evidence of absence.**

## Reported, deciding nothing

Bootstrap CIs use 12-month blocks over 60 months, so they are coarse. No D048 label and no decay ratio are reported (D050 §9).

### R² by model

| model      |    R2 % | R2 95% CI       |   1987–2016 R2 % |
|:-----------|--------:|:----------------|-----------------:|
| OLS-3      |  0.2131 | [-0.419, 0.702] |           0.2192 |
| OLS-3+H    |  0.1906 | [-0.341, 0.608] |           0.2101 |
| OLS        | -0.5756 | [-2.869, 0.882] |        -170.026  |
| OLS+H      | -0.4553 | [-2.534, 0.859] |        -169.98   |
| PCR        |  0.2679 | [-0.326, 0.730] |           0.2951 |
| PLS        |  0.2713 | [-0.310, 0.723] |           0.2841 |
| ENet+H     |  0.2534 | [-0.260, 0.647] |           0.3029 |
| GLM+H      |  0.2483 | [-0.269, 0.644] |           0.353  |
| GBRT+H     |  0.25   | [-0.291, 0.616] |           0.089  |
| GBRT+H-dm  |  0.3417 | [-0.224, 0.720] |           0.6721 |
| RF         |  0.2873 | [-0.286, 0.670] |           0.2774 |
| RF-dm      |  0.3926 | [-0.281, 0.850] |           0.7028 |
| NN1        |  0.374  | [-0.474, 1.002] |           0.5142 |
| NN1-dm     |  0.2916 | [-0.346, 0.793] |           0.5569 |
| NN2        |  0.3332 | [-0.426, 0.877] |           0.4526 |
| NN2-dm     |  0.3511 | [-0.313, 0.859] |           0.6182 |
| NN3        |  0.3438 | [-0.439, 0.916] |           0.4539 |
| NN3-dm     |  0.3474 | [-0.307, 0.862] |           0.6566 |
| train_mean |  0.2713 | [-0.310, 0.723] |           0.1814 |

### The seven D045 tests (test 1 had ≈ 23% power at 60 months)

| candidate   | vs         | test   |       t |   p (one-sided) |   ΔR2 % | 95% CI          |   1987–2016 t |   1987–2016 p |
|:------------|:-----------|:-------|--------:|----------------:|--------:|:----------------|--------------:|--------------:|
| NN3         | train_mean | DM     |  0.6039 |          0.273  |  0.0726 | [-0.239, 0.231] |        2.213  |        0.0134 |
| NN3         | PCR        | DM     |  0.6863 |          0.2463 |  0.0759 | [-0.204, 0.221] |        1.4716 |        0.0706 |
| NN3         | PLS        | DM     |  0.6039 |          0.273  |  0.0726 | [-0.239, 0.231] |        1.6633 |        0.0481 |
| NN3         | ENet+H     | DM     |  0.613  |          0.2699 |  0.0904 | [-0.264, 0.294] |        1.3194 |        0.0935 |
| NN3         | GLM+H      | DM     |  0.6481 |          0.2585 |  0.0956 | [-0.237, 0.295] |        0.9447 |        0.1724 |
| GBRT+H      | ENet+H     | DM     | -0.0556 |          0.5222 | -0.0034 | [-0.121, 0.106] |       -1.9154 |        0.9723 |
| ENet+H      | OLS-3+H    | CW     |  4.2765 |          0      |  0.0629 | [0.005, 0.123]  |        5.9517 |        0      |

### H1 and secondary tests

| test                                  |       t |   p (one-sided) |   1987–2016 t |   1987–2016 p |
|:--------------------------------------|--------:|----------------:|--------------:|--------------:|
| H1: NN3-dm vs constant (confirmatory) |  1.5035 |          0.0664 |        4.5689 |         0     |
| NN3-dm vs ENet+H, within-month errors |  1.3024 |          0.0964 |        4.0366 |         0     |
| ENet+H vs constant                    | -0.4678 |          0.68   |        1.6254 |         0.052 |
