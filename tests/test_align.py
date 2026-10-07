"""Seam 2: signals dated t are paired with the return in t+1 (D008).

OSAP convention, verified from openassetpricing 0.0.2's own code: STreversal at
yyyymm t is minus the return *in* month t, so a signal dated t uses information
through the end of t and may only predict returns in t+1.
"""

import numpy as np
import pandas as pd

from gkx.align import align


def test_signal_at_t_is_paired_with_return_at_t_plus_1_across_year_end():
    signals = pd.DataFrame({"permno": [1, 1], "yyyymm": [201511, 201512], "BM": [0.4, 0.5]})
    returns = pd.DataFrame(
        {"permno": [1, 1, 1], "yyyymm": [201511, 201512, 201601], "ret_excess": [0.01, 0.02, 0.03]}
    )
    out = align(signals, returns).set_index("yyyymm")
    assert out.loc[201511, "ret_excess_next"] == 0.02
    assert out.loc[201512, "ret_excess_next"] == 0.03  # December rolls to January, not "201513"


def test_missing_next_month_gives_nan_not_a_later_return():
    signals = pd.DataFrame({"permno": [1], "yyyymm": [201503], "BM": [0.4]})
    returns = pd.DataFrame({"permno": [1], "yyyymm": [201505], "ret_excess": [0.09]})  # 201504 missing
    out = align(signals, returns)
    assert np.isnan(out["ret_excess_next"].iloc[0])


def test_signal_is_never_paired_with_its_own_month_return():
    signals = pd.DataFrame({"permno": [7], "yyyymm": [200006], "BM": [1.0]})
    returns = pd.DataFrame({"permno": [7, 7], "yyyymm": [200006, 200007], "ret_excess": [0.50, -0.10]})
    assert align(signals, returns)["ret_excess_next"].iloc[0] == -0.10
