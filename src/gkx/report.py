"""Pre-registered final-analysis helpers (D005, D015, D030, D036), written and tested before the
single test-window look so that every reported number comes from code fixed in advance."""

import numpy as np
import pandas as pd

from gkx.metrics import BLOCK, N_BOOT, SEED, month_block_indices, sharpe

CRASH_MONTHS = (198711, 200811)  # D036: return months after the Oct-1987 and Oct-2008 crashes


def demeaned_r2(df: pd.DataFrame, pred: str) -> float:
    """R² after removing each month's mean target and mean prediction (pure ranking skill, D036)."""
    month = df.groupby("yyyymm")
    t = df["target"] - month["target"].transform("mean")
    p = df[pred] - month[pred].transform("mean")
    return float(1 - ((t - p) ** 2).sum() / (t**2).sum())


def r2_excluding(df: pd.DataFrame, pred: str, months) -> float:
    """Pooled GKX R² over all months except `months` (D036 crash-month secondary)."""
    d = df[~df["yyyymm"].isin(months)]
    return float(1 - ((d["target"] - d[pred]) ** 2).sum() / (d["target"] ** 2).sum())


def size_masks(df: pd.DataFrame, n: int = 1000) -> tuple[pd.Series, pd.Series]:
    """Top / bottom n stocks by formation-month market equity, each month (D030)."""
    rank_desc = df.groupby("yyyymm")["me_lag"].rank(method="first", ascending=False)
    rank_asc = df.groupby("yyyymm")["me_lag"].rank(method="first", ascending=True)
    return rank_desc <= n, rank_asc <= n


def sharpe_ci(
    monthly: pd.Series, n_boot: int = N_BOOT, block: int = BLOCK, seed: int = SEED
) -> tuple[float, float, float]:
    """(annualized Sharpe, 2.5%, 97.5%) with the month-block bootstrap of D015."""
    x = monthly.to_numpy(dtype=float)
    idx = month_block_indices(len(x), n_boot, block, seed)
    boot = x[idx]
    sr = boot.mean(1) / boot.std(1, ddof=1) * np.sqrt(12)
    lo, hi = np.percentile(sr, [2.5, 97.5])
    return sharpe(monthly), float(lo), float(hi)
