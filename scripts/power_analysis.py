"""Power analysis: can the locked criteria detect the model gaps at stake? (council 1, D017)

First attempt used a constant premium; with N=6000 stocks that is detectable at any size
(100% power everywhere) and therefore uninformative. Real anomaly premia vary month to
month, so the binding quantity is how *stable* the incremental signal's payoff is,
summarized by its long-short Sharpe ratio. This version plants that directly.

DGP per return month t, stock i (mean zero):
    r_it = μ1·x1_it + (μ2 + s_t)·x2_it + f_t + e_it
    x1, x2 ~ N(0,1) characteristics
    μ1 sets a base model's R² (0.2%); μ2 sets the INCREMENTAL R² (the gap being tested)
    s_t ~ N(0, sd_s): time-varying premium on x2; sd_s = μ2·√12 / SR (SR = annual Sharpe of
          the x2 payoff)
    f_t: market factor, sd 4.5%/month; e_it: Student-t(4) idiosyncratic; total sd 15%/month
Model A predicts μ1·x1; model B predicts μ1·x1 + μ2·x2 (B nests A).
Both know the true average coefficients, so this is an UPPER BOUND on power: fitted models
add estimation error.

Tests are the ones the study will run: Clark–West B vs. A (D010), HAC 5 lags, one-sided 5%;
paired month-block bootstrap 95% CI on ΔR² = R²(B) − R²(A), success = lower bound > 0.

Usage: uv run python scripts/power_analysis.py   (writes results/power_analysis.md)
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.metrics import BLOCK, hac_mean_test, month_block_indices

T, N, REPS, N_BOOT = 360, 6000, 200, 2000
SIGMA, SIGMA_F, T_DF, BASE_R2 = 0.15, 0.045, 4, 0.002
DELTA_R2 = (0.001, 0.002, 0.004)
SHARPE = (0.5, 1.0, 2.0)
SEED = 20261008


def one_rep(
    delta_r2: float, sr: float, rng: np.random.Generator, boot_idx: np.ndarray
) -> tuple[bool, bool, float]:
    mu1, mu2 = np.sqrt(BASE_R2) * SIGMA, np.sqrt(delta_r2) * SIGMA
    sd_s = mu2 * np.sqrt(12) / sr
    sd_e = np.sqrt(SIGMA**2 - SIGMA_F**2 - mu1**2 - mu2**2 - sd_s**2)
    x1, x2 = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    s = rng.normal(scale=sd_s, size=(T, 1))
    f = rng.normal(scale=SIGMA_F, size=(T, 1))
    e = rng.standard_t(T_DF, size=(T, N)) * sd_e / np.sqrt(T_DF / (T_DF - 2))
    r = mu1 * x1 + (mu2 + s) * x2 + f + e
    pa, pb = mu1 * x1, mu1 * x1 + mu2 * x2
    sst = (r**2).sum(1)
    sse_a, sse_b = ((r - pa) ** 2).sum(1), ((r - pb) ** 2).sum(1)
    cw = sse_a - (sse_b - ((pa - pb) ** 2).sum(1))
    _, pval = hac_mean_test(pd.Series(cw))
    boot = (sse_a[boot_idx].sum(1) - sse_b[boot_idx].sum(1)) / sst[boot_idx].sum(1)
    realized = (sse_a.sum() - sse_b.sum()) / sst.sum()
    return pval < 0.05, np.percentile(boot, 2.5) > 0, realized


def main() -> None:
    rng = np.random.default_rng(SEED)
    boot_idx = month_block_indices(T, N_BOOT, BLOCK, SEED)
    rows = []
    for delta_r2, sr in product(DELTA_R2, SHARPE):
        reps = [one_rep(delta_r2, sr, rng, boot_idx) for _ in range(REPS)]
        cw, ci, real = map(np.array, zip(*reps, strict=True))
        rows.append(
            {
                "ΔR² planted": f"{delta_r2:.1%}",
                "signal Sharpe (ann.)": sr,
                "ΔR² realized 5–95%": f"{np.percentile(real, 5):.2%} to {np.percentile(real, 95):.2%}",
                "Clark–West power": f"{cw.mean():.0%}",
                "ΔR² CI excludes 0": f"{ci.mean():.0%}",
            }
        )
        print(rows[-1], flush=True)

    out = Path("results/power_analysis.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "# Power analysis: detecting a model gap (oracle upper bound)\n\n"
        f"T={T} months, N={N} stocks/month, {REPS} reps per cell, bootstrap B={N_BOOT}, block "
        f"{BLOCK}, seed {SEED}. Total sd {SIGMA:.0%}/month, market factor sd {SIGMA_F:.1%}, "
        f"Student-t({T_DF}) idiosyncratic, base-model R² {BASE_R2:.1%}. DGP in the script "
        "docstring. Both models know the true average coefficients: real fitted models will "
        "have LESS power than shown.\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
