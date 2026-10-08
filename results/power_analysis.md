# Power analysis: detecting a model gap (oracle upper bound)

T=360 months, N=6000 stocks/month, 200 reps per cell, bootstrap B=2000, block 12, seed 20261008. Total sd 15%/month, market factor sd 4.5%, Student-t(4) idiosyncratic, base-model R² 0.2%. DGP in the script docstring. Both models know the true average coefficients: real fitted models will have LESS power than shown.

| ΔR² planted   |   signal Sharpe (ann.) | ΔR² realized 5–95%   | Clark–West power   | ΔR² CI excludes 0   |
|:--------------|-----------------------:|:---------------------|:-------------------|:--------------------|
| 0.1%          |                    0.5 | -0.03% to 0.21%      | 84%                | 26%                 |
| 0.1%          |                    1   | 0.03% to 0.16%       | 100%               | 74%                 |
| 0.1%          |                    2   | 0.08% to 0.13%       | 100%               | 100%                |
| 0.2%          |                    0.5 | -0.03% to 0.42%      | 86%                | 28%                 |
| 0.2%          |                    1   | 0.07% to 0.30%       | 100%               | 76%                 |
| 0.2%          |                    2   | 0.14% to 0.26%       | 100%               | 100%                |
| 0.4%          |                    0.5 | -0.04% to 0.93%      | 92%                | 40%                 |
| 0.4%          |                    1   | 0.12% to 0.66%       | 100%               | 76%                 |
| 0.4%          |                    2   | 0.28% to 0.54%       | 100%               | 100%                |
