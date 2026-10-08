"""Out-of-sample metrics and forecast-comparison tests (D005, D010, D015).

Everything reduces to per-month sums first, so tests run on T ≈ 360 months, not
millions of stock-months, and weight months the way pooled R² does (by their
number of stocks).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HAC_LAGS = 5  # D015: Newey-West rule for T = 360


def r2_oos(df: pd.DataFrame, pred: str) -> float:
    """GKX pooled R²: 1 − Σ(r − r̂)² / Σr² over all stock-months. Not demeaned (D005)."""
    err = df["target"] - df[pred]
    return float(1 - (err**2).sum() / (df["target"] ** 2).sum())


def hac_mean_test(x: pd.Series, lags: int = HAC_LAGS) -> tuple[float, float]:
    """Newey-West t-test of H1: E[x] > 0. Returns (t, one-sided normal p)."""
    fit = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(fit.tvalues[0])
    return t, float(stats.norm.sf(t))


def _monthly_sum(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(df["yyyymm"]).sum()


def diebold_mariano(df: pd.DataFrame, base: str, candidate: str) -> tuple[float, float]:
    """Non-nested comparison. Positive t means `candidate` has lower squared error."""
    d = (df["target"] - df[base]) ** 2 - (df["target"] - df[candidate]) ** 2
    return hac_mean_test(_monthly_sum(df, d))


def clark_west(df: pd.DataFrame, small: str, big: str) -> tuple[float, float]:
    """Nested comparison (Clark & West 2007). Positive t means `big` adds predictive value.

    Adds back (ŝ − b̂)², the MSE penalty the big model pays for estimating
    parameters that are zero under the null, which makes plain DM undersized.
    """
    r, s, b = df["target"], df[small], df[big]
    f = (r - s) ** 2 - ((r - b) ** 2 - (s - b) ** 2)
    return hac_mean_test(_monthly_sum(df, f))


BLOCK = 12  # D015
N_BOOT = 10_000
SEED = 20261008


def month_block_indices(
    n_months: int, n_boot: int = N_BOOT, block: int = BLOCK, seed: int = SEED
) -> np.ndarray:
    """Circular block bootstrap over months: (n_boot, n_months) month positions.

    Shared across models, so every CI built from it is paired (D015).
    """
    rng = np.random.default_rng(seed)
    n_blocks = -(-n_months // block)
    starts = rng.integers(0, n_months, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_months
    return idx.reshape(n_boot, -1)[:, :n_months]


def paired_r2_ci(
    df: pd.DataFrame, a: str, b: str, n_boot: int = N_BOOT, block: int = BLOCK, seed: int = SEED
) -> tuple[float, float, float]:
    """(ΔR², 2.5%, 97.5%) for R²(b) − R²(a), resampling whole months jointly for both models."""
    sse_a = _monthly_sum(df, (df["target"] - df[a]) ** 2).to_numpy()
    sse_b = _monthly_sum(df, (df["target"] - df[b]) ** 2).to_numpy()
    sst = _monthly_sum(df, df["target"] ** 2).to_numpy()
    point = float((sse_a.sum() - sse_b.sum()) / sst.sum())
    idx = month_block_indices(len(sst), n_boot, block, seed)
    boot = (sse_a[idx].sum(1) - sse_b[idx].sum(1)) / sst[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def decile_long_short(df: pd.DataFrame, pred: str) -> pd.DataFrame:
    """Monthly top-minus-bottom decile returns, equal- and value-weighted.

    `me_lag` must be market equity at the signal (formation) month; using the
    return month's ME would leak that month's return into the weights.
    Months with fewer than 10 stocks are skipped.
    """
    df = df[df.groupby("yyyymm")["permno"].transform("count") >= 10]
    if df.empty:
        return pd.DataFrame(columns=["ew", "vw"], dtype=float)
    rank = df.groupby("yyyymm")[pred].rank(method="first")
    n = df.groupby("yyyymm")[pred].transform("count")
    decile = np.ceil(rank * 10 / n).astype(int)
    legs = df.assign(leg=np.select([decile == 10, decile == 1], [1, -1], 0))
    legs = legs[legs["leg"] != 0].assign(w_ret=lambda d: d["target"] * d["me_lag"])
    g = legs.groupby(["yyyymm", "leg"])
    ew = g["target"].mean().unstack()
    vw = (g["w_ret"].sum() / g["me_lag"].sum()).unstack()
    return pd.DataFrame({"ew": ew[1] - ew[-1], "vw": vw[1] - vw[-1]})


def sharpe(monthly: pd.Series) -> float:
    """Annualized: monthly mean / monthly sd × √12 (D015)."""
    return float(monthly.mean() / monthly.std(ddof=1) * np.sqrt(12))
