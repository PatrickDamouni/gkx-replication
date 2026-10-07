"""GKX (2020) §2 characteristic preprocessing: per-month ranks mapped to [-1, 1]."""

import pandas as pd


def rank_normalize(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rank each column within each month, map to [-1, 1], set missing to 0 (the median).

    Uses only same-month data, so it is computed once for the whole panel and
    never needs refitting per training window.
    """
    out = panel.copy()
    by_month = out.groupby("yyyymm")
    for col in cols:
        rank = by_month[col].rank(method="average")
        n = by_month[col].transform("count")
        scaled = 2 * (rank - 1) / (n - 1) - 1
        out[col] = scaled.where(n > 1, 0.0).fillna(0.0)
    return out
