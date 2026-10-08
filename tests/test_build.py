"""Phase 1b: assemble the modeling panel from characteristics, CRSP and macro (D014, D020, D021)."""

import numpy as np
import pandas as pd

from gkx.build import build_panel
from gkx.invariance import assert_future_invariant


def _crsp(rows):
    return pd.DataFrame(
        rows, columns=["permno", "yyyymm", "ret_excess", "me", "me_lag", "shrcd", "exchcd"]
    )


MACRO = pd.DataFrame({"yyyymm": [200001, 200002], "dp": [-3.5, -3.6], "tbl": [0.05, 0.051]})


def test_universe_membership_is_decided_at_the_signal_month():
    chars = pd.DataFrame({"permno": [1, 2], "yyyymm": [200001, 200001], "BM": [0.1, 0.2]})
    crsp = _crsp(
        [
            (1, 200001, 0.0, 10.0, 9.0, 10, 1),
            (1, 200002, 0.03, 11.0, 10.0, 10, 1),  # firm 1: in universe at t, return at t+1
            (2, 200002, 0.05, 5.0, np.nan, 10, 3),  # firm 2: no CRSP row at t (e.g. just listed)
        ]
    )
    out = build_panel(chars, crsp, MACRO, char_cols=["BM"])
    assert out["permno"].tolist() == [1]
    row = out.iloc[0]
    assert (row["yyyymm"], row["signal_month"], row["target"]) == (200002, 200001, 0.03)
    assert row["me_lag"] == 10.0  # ME at the signal month, for value weights


def test_ranks_are_computed_only_among_universe_firms():
    chars = pd.DataFrame(
        {"permno": [1, 2, 3], "yyyymm": [200001] * 3, "BM": [1.0, 2.0, 99.0]}  # firm 3 not in CRSP
    )
    crsp = _crsp(
        [
            (1, 200001, 0.0, 1.0, 1.0, 10, 1),
            (2, 200001, 0.0, 1.0, 1.0, 10, 1),
            (1, 200002, 0.01, 1.0, 1.0, 10, 1),
            (2, 200002, 0.02, 1.0, 1.0, 10, 1),
        ]
    )
    out = build_panel(chars, crsp, MACRO, char_cols=["BM"]).set_index("permno")
    assert (
        out.loc[1, "BM"] == -1.0 and out.loc[2, "BM"] == 1.0
    )  # firm 3's 99 never entered the ranks


def test_macro_joins_on_the_signal_month():
    chars = pd.DataFrame({"permno": [1], "yyyymm": [200001], "BM": [0.1]})
    crsp = _crsp([(1, 200001, 0.0, 1.0, 1.0, 10, 1), (1, 200002, 0.01, 1.0, 1.0, 10, 1)])
    row = build_panel(chars, crsp, MACRO, char_cols=["BM"]).iloc[0]
    assert (row["m_dp"], row["m_tbl"]) == (-3.5, 0.05)  # 200001 values, not 200002


def test_panel_up_to_a_return_month_ignores_everything_later():
    rng = np.random.default_rng(0)
    months = [200001 + m for m in range(12)]
    firms = range(20)
    chars = pd.DataFrame(
        [(f, m, rng.normal()) for m in months for f in firms], columns=["permno", "yyyymm", "BM"]
    )
    crsp = _crsp([(f, m, rng.normal(), 1.0, 1.0, 10, 1) for m in months for f in firms])
    macro = pd.DataFrame({"yyyymm": months, "dp": rng.normal(size=12)})

    def from_crsp(c: pd.DataFrame) -> pd.DataFrame:
        return build_panel(chars, c, macro, char_cols=["BM"])

    assert_future_invariant(from_crsp, crsp, cutoff=200006)


def test_passthrough_columns_are_carried_unranked_from_the_signal_month():
    chars = pd.DataFrame(
        {"permno": [1, 2], "yyyymm": [200001] * 2, "BM": [0.1, 0.2], "sic2": [35, 60]}
    )
    crsp = _crsp(
        [
            (1, 200001, 0.0, 1.0, 1.0, 10, 1),
            (2, 200001, 0.0, 1.0, 1.0, 10, 1),
            (1, 200002, 0.01, 1.0, 1.0, 10, 1),
            (2, 200002, 0.02, 1.0, 1.0, 10, 1),
        ]
    )
    out = build_panel(chars, crsp, MACRO, char_cols=["BM"], keep=["sic2"]).set_index("permno")
    assert out.loc[1, "sic2"] == 35 and out.loc[2, "sic2"] == 60


def test_macro_predictors_are_prefixed_so_they_never_collide_with_characteristics():
    """datashare has chars bm/ep and Welch-Goyal has macro bm/ep (council 4, all three members)."""
    chars = pd.DataFrame({"permno": [1], "yyyymm": [200001], "bm": [0.7], "ep": [0.1]})
    crsp = _crsp([(1, 200001, 0.0, 1.0, 1.0, 10, 1), (1, 200002, 0.01, 1.0, 1.0, 10, 1)])
    macro = pd.DataFrame({"yyyymm": [200001], "bm": [0.5], "ep": [-2.8], "dp": [-3.5]})
    out = build_panel(chars, crsp, macro, char_cols=["bm", "ep"])
    assert {"bm", "ep", "m_bm", "m_ep", "m_dp"} <= set(out.columns)
    assert not any(c.endswith(("_x", "_y")) for c in out.columns)
    assert out["m_bm"].iloc[0] == 0.5  # the market's B/M, distinct from the stock's


def test_nullable_input_dtypes_come_out_as_plain_numpy_floats():
    """CRSP loads as pandas Float64/Int64; to_numpy() on those gives dtype=object (council 4)."""
    chars = pd.DataFrame({"permno": [1], "yyyymm": [200001], "bm": [0.7]})
    crsp = _crsp([(1, 200001, 0.0, 1.0, 1.0, 10, 1), (1, 200002, 0.01, 2.0, 1.0, 10, 1)]).astype(
        {"ret_excess": "Float64", "me": "Float64", "shrcd": "Int64"}
    )
    out = build_panel(chars, crsp, MACRO, char_cols=["bm"])
    assert out[["target", "me_lag", "bm"]].to_numpy().dtype == np.float64
