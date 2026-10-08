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
