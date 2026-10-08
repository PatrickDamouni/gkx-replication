"""Assemble the modeling panel: characteristics at t, macro at t, excess return at t+1.

Order matters and is fixed here, in one place:
  1. universe membership is decided at the signal month t, from CRSP rows that exist at t
     (deciding it at t+1 would use knowledge of which firms survive);
  2. characteristics are rank-normalized within that universe, month by month;
  3. rows are re-keyed to the return month t+1 with the target attached (D014);
  4. macro predictors join on the signal month (D021), prefixed `m_` because Welch-Goyal's
     bm/ep share names with characteristics (council 4).
ME at t is carried as `me_lag` for value weights.
"""

import numpy as np
import pandas as pd

from gkx.align import align
from gkx.preprocess import rank_normalize


def build_panel(
    chars: pd.DataFrame,
    crsp: pd.DataFrame,
    macro: pd.DataFrame,
    char_cols: list[str],
    keep: list[str] | None = None,
) -> pd.DataFrame:
    """`keep`: signal-month columns carried through unranked (e.g. sic2 for industry dummies)."""
    at_t = crsp[["permno", "yyyymm", "me", "shrcd"]].rename(columns={"me": "me_lag"})
    signals = chars[["permno", "yyyymm", *char_cols, *(keep or [])]].merge(
        at_t, on=["permno", "yyyymm"], how="inner", validate="one_to_one"
    )
    signals = rank_normalize(signals, char_cols)
    panel = align(signals, crsp[["permno", "yyyymm", "ret_excess"]])
    for col in (
        "target",
        "me_lag",
    ):  # nullable Float64 -> numpy float64 (to_numpy is object otherwise)
        panel[col] = panel[col].to_numpy(dtype="float64", na_value=np.nan)
    m = macro.rename(columns=lambda c: c if c == "yyyymm" else f"m_{c}")
    overlap = (set(m.columns) - {"yyyymm"}) & set(panel.columns)
    if overlap:
        raise ValueError(f"macro columns collide with panel columns: {sorted(overlap)}")
    return panel.merge(
        m.rename(columns={"yyyymm": "signal_month"}),
        on="signal_month",
        how="left",
        validate="many_to_one",
    )
