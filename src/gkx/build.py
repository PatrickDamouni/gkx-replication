"""Assemble the modeling panel: characteristics at t, macro at t, excess return at t+1.

Order matters and is fixed here, in one place:
  1. universe membership is decided at the signal month t, from CRSP rows that exist at t
     (deciding it at t+1 would use knowledge of which firms survive);
  2. characteristics are rank-normalized within that universe, month by month;
  3. rows are re-keyed to the return month t+1 with the target attached (D014);
  4. macro predictors join on the signal month (D021).
ME at t is carried as `me_lag` for value weights.
"""

import pandas as pd

from gkx.align import align
from gkx.preprocess import rank_normalize


def build_panel(
    chars: pd.DataFrame, crsp: pd.DataFrame, macro: pd.DataFrame, char_cols: list[str]
) -> pd.DataFrame:
    at_t = crsp[["permno", "yyyymm", "me", "shrcd"]].rename(columns={"me": "me_lag"})
    signals = chars[["permno", "yyyymm", *char_cols]].merge(
        at_t, on=["permno", "yyyymm"], how="inner", validate="one_to_one"
    )
    signals = rank_normalize(signals, char_cols)
    panel = align(signals, crsp[["permno", "yyyymm", "ret_excess"]])
    return panel.merge(
        macro.rename(columns={"yyyymm": "signal_month"}),
        on="signal_month",
        how="left",
        validate="many_to_one",
    )
