"""Characteristics rebuilt exactly from CRSP monthly data (D024 rule 1).

Windows are relative to the signal month s and were identified by reproducing GKX's
datashare (scripts/validate_crsp_chars.py -> results/crsp_chars_validation.md). A missing
month inside a window gives NaN, never a shorter window.
"""

import numpy as np
import pandas as pd

# name: (oldest offset, newest offset) of a compounded window, offsets counted back from s
WINDOWS = {"mom1m": (0, 0), "mom6m": (5, 1), "mom12m": (11, 1), "mom36m": (35, 12)}
CHMOM = ((5, 0), (11, 6))  # GHZ quirk: the recent leg includes month s


def _month_index(yyyymm: pd.Series) -> pd.Series:
    return yyyymm // 100 * 12 + yyyymm % 100


def crsp_characteristics(crsp: pd.DataFrame) -> pd.DataFrame:
    """crsp: permno, yyyymm, ret, me ($M). Returns permno, yyyymm (signal month) + characteristics."""
    m = _month_index(crsp["yyyymm"])
    ret = crsp.assign(m=m).pivot(index="m", columns="permno", values="ret")
    ret = ret.reindex(range(int(m.min()), int(m.max()) + 1))  # explicit gaps
    with np.errstate(divide="ignore"):
        logr = np.log1p(ret)  # ret = -1 -> -inf -> compounded -1

    def compound(oldest: int, newest: int) -> pd.DataFrame:
        n = oldest - newest + 1
        return np.expm1(logr.shift(newest).rolling(n, min_periods=n).sum())

    frames = {name: compound(*w) for name, w in WINDOWS.items()}
    frames["chmom"] = compound(*CHMOM[0]) - compound(*CHMOM[1])

    long = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    long.index.names = ["m", "permno"]
    long = long.reset_index()
    out = crsp[["permno", "yyyymm", "me"]].assign(m=m).merge(long, on=["m", "permno"], how="left")
    out["mvel1"] = out["me"] * 1000  # datashare mvel1 is in $ thousands (D023)
    return out[["permno", "yyyymm", "mvel1", "mom1m", "mom6m", "mom12m", "mom36m", "chmom"]]
