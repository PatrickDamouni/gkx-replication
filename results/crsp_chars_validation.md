# CRSP-rebuilt characteristics vs GKX datashare (pre-2017)

Rule (D024): Spearman ≥ 0.99 and ≥ 95% equal (rtol/atol 1e-4). Same (permno, signal month).

| characteristic   |   spearman | equal   | coverage of datashare   |      rows | accepted   |
|:-----------------|-----------:|:--------|:------------------------|----------:|:-----------|
| mvel1            |     1      | 99.9%   | 100.0%                  | 3,764,606 | yes        |
| mom1m            |     0.9997 | 98.0%   | 99.9%                   | 3,734,245 | yes        |
| mom6m            |     0.9991 | 97.9%   | 98.6%                   | 3,573,749 | yes        |
| mom12m           |     0.9991 | 97.8%   | 98.0%                   | 3,388,081 | yes        |
| mom36m           |     0.9982 | 96.8%   | 96.8%                   | 2,768,019 | yes        |
| chmom            |     0.9987 | 97.7%   | 98.0%                   | 3,386,500 | yes        |
