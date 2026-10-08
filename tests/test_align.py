"""Seam 2: signals dated t are paired with the return in t+1; rows are keyed on the return month (D014).

OSAP convention, verified from openassetpricing 0.0.2's own code: STreversal at
yyyymm t is minus the return *in* month t, so a signal dated t uses information
through the end of t and may only predict returns in t+1.

Rows are keyed on the *return* month because GKX's sample windows (1957-03,
1987-2016) are return months; `signal_month` is kept as metadata.
"""

import numpy as np
import pandas as pd

from gkx.align import align


def test_row_is_keyed_on_return_month_and_carries_the_signal_month():
    signals = pd.DataFrame({"permno": [1, 1], "yyyymm": [201511, 201512], "BM": [0.4, 0.5]})
    returns = pd.DataFrame(
        {
            "permno": [1, 1, 1],
            "yyyymm": [201511, 201512, 201601],
            "ret_excess": [0.01, 0.02, 0.03],
        }
    )
    out = align(signals, returns).set_index("yyyymm")
    assert out.loc[201512, "target"] == 0.02 and out.loc[201512, "signal_month"] == 201511
    assert (
        out.loc[201601, "target"] == 0.03 and out.loc[201601, "BM"] == 0.5
    )  # Dec -> Jan, not "201513"


def test_missing_next_month_gives_nan_not_a_later_return():
    signals = pd.DataFrame({"permno": [1], "yyyymm": [201503], "BM": [0.4]})
    returns = pd.DataFrame(
        {"permno": [1], "yyyymm": [201505], "ret_excess": [0.09]}
    )  # 201504 missing
    out = align(signals, returns)
    assert out["yyyymm"].iloc[0] == 201504 and np.isnan(out["target"].iloc[0])


def test_signal_is_never_paired_with_its_own_month_return():
    signals = pd.DataFrame({"permno": [7], "yyyymm": [200006], "BM": [1.0]})
    returns = pd.DataFrame(
        {"permno": [7, 7], "yyyymm": [200006, 200007], "ret_excess": [0.50, -0.10]}
    )
    assert align(signals, returns)["target"].iloc[0] == -0.10


def test_rows_whose_return_month_is_in_the_holdout_are_excluded():
    """The Dec-2016 signal predicts a sealed Jan-2017 return; that row belongs to the holdout."""
    signals = pd.DataFrame({"permno": [1, 1], "yyyymm": [201611, 201612], "BM": [0.1, 0.2]})
    returns = pd.DataFrame({"permno": [1], "yyyymm": [201612], "ret_excess": [0.05]})
    out = align(signals, returns)
    assert out["yyyymm"].tolist() == [201612]


def test_a_t_plus_2_misalignment_would_be_caught():
    """Negative control: pairing t with t+2 must give a different target than the real align."""
    signals = pd.DataFrame({"permno": [1], "yyyymm": [201001], "BM": [0.3]})
    returns = pd.DataFrame(
        {"permno": [1, 1], "yyyymm": [201002, 201003], "ret_excess": [0.02, 0.07]}
    )
    assert align(signals, returns)["target"].iloc[0] == 0.02  # 0.07 would mean t+2
