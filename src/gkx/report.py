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


def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down: reject the k-th smallest p while p ≤ α / (m − k + 1) (D045)."""
    p = [x if np.isfinite(x) else 1.0 for x in pvalues]  # fail closed: NaN never rejects (D047)
    order = np.argsort(p, kind="stable")
    reject = [False] * len(p)
    for k, i in enumerate(order):
        if not p[i] <= alpha / (len(p) - k):
            break
        reject[i] = True
    return reject


def demeaned_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Copy with each month's mean removed from the target and from `cols` (ranking-only DM, D045)."""
    out = df[["yyyymm", "target", *cols]].copy()
    month = out.groupby("yyyymm")
    for c in ["target", *cols]:
        out[c] = out[c] - month[c].transform("mean")
    return out


def verdict(
    primary_lo: float, pvalues: list[float], first_rejected: bool, alpha: float = 0.05
) -> str:
    """D045 as amended by D047; every NaN fails closed.

    Primary CI must exclude zero. "replicates" = every confirmatory p <= alpha (intersection-union
    test of the conjunction). Otherwise "partially replicates" needs test 1 (headline vs the
    training-mean forecast) rejected under Holm; without it the result is level only.
    """
    if not primary_lo > 0:
        return "does not replicate"
    if all(p <= alpha for p in pvalues):  # NaN <= alpha is False
        return "replicates"
    return "partially replicates" if first_rejected else "does not replicate (level only)"
