"""Pair each signal month t with the excess return in month t+1, keyed on the return month.

This join is the most likely place for lookahead in the whole project, so it
lives in one function with its own tests (D008). Rows are keyed on the return
month because GKX's windows are return months (D014): `yyyymm` is when the
target is realized, `signal_month` is when the features were known.
"""

import pandas as pd

from gkx.panel import HOLDOUT_START


def next_month(yyyymm: pd.Series) -> pd.Series:
    """201512 -> 201601. Integer month arithmetic, no calendar ambiguity."""
    year, month = yyyymm // 100, yyyymm % 100
    return (year + month // 12) * 100 + month % 12 + 1


def align(
    signals: pd.DataFrame, returns: pd.DataFrame, through: int = HOLDOUT_START - 1
) -> pd.DataFrame:
    """Re-key `signals` from signal month t to return month t+1 and attach `target`.

    A missing t+1 return stays NaN; it is never filled from a later month. Rows
    whose return month is after `through` are excluded; the default keeps the
    holdout sealed, and only the Phase 5 builder passes a later month (D050).
    """
    out = signals.rename(columns={"yyyymm": "signal_month"})
    out.insert(1, "yyyymm", next_month(out["signal_month"]))
    out = out[out["yyyymm"] <= through]
    target = returns[["permno", "yyyymm", "ret_excess"]].rename(columns={"ret_excess": "target"})
    return out.merge(target, on=["permno", "yyyymm"], how="left", validate="one_to_one")
