"""Pair each signal month t with the excess return in month t+1.

This join is the most likely place for lookahead in the whole project, so it
lives in one function with its own tests (D008).
"""

import pandas as pd


def next_month(yyyymm: pd.Series) -> pd.Series:
    """201512 -> 201601. Integer month arithmetic, no calendar ambiguity."""
    year, month = yyyymm // 100, yyyymm % 100
    return (year + month // 12) * 100 + month % 12 + 1


def align(signals: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Left-join `returns.ret_excess` at t+1 onto `signals` at t as `ret_excess_next`.

    A missing t+1 return stays NaN; it is never filled from a later month.
    """
    nxt = returns[["permno", "yyyymm", "ret_excess"]].rename(
        columns={"yyyymm": "label_month", "ret_excess": "ret_excess_next"}
    )
    out = signals.assign(label_month=next_month(signals["yyyymm"]))
    return out.merge(nxt, on=["permno", "label_month"], how="left", validate="one_to_one").drop(
        columns="label_month"
    )
