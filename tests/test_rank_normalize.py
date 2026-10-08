"""Seam 2: GKX per-month rank normalization, and the future-invariance property (D008).

GKX (2020) §2: characteristics are ranked cross-sectionally each month and
mapped to [-1, 1]; missing values are set to the cross-sectional median, which
is 0 after the mapping.
"""

import numpy as np
import pandas as pd
import pytest

from gkx.align import align
from gkx.invariance import FutureLeak, assert_future_invariant
from gkx.preprocess import rank_normalize


def test_ranks_map_to_minus_one_to_one_and_missing_goes_to_zero():
    panel = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4],
            "yyyymm": [200001] * 4,
            "BM": [10.0, 30.0, 20.0, np.nan],
        }
    )
    out = rank_normalize(panel, ["BM"])
    assert out["BM"].tolist() == [-1.0, 1.0, 0.0, 0.0]


def test_each_month_is_ranked_on_its_own():
    panel = pd.DataFrame(
        {
            "permno": [1, 2, 1, 2],
            "yyyymm": [200001, 200001, 200002, 200002],
            "BM": [1.0, 2.0, 100.0, 50.0],
        }
    )
    assert rank_normalize(panel, ["BM"])["BM"].tolist() == [-1.0, 1.0, 1.0, -1.0]


def _random_panel(seed: int = 0, firms: int = 30, months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yyyymm = np.array([200001 + (m // 12) * 100 + m % 12 for m in range(months)])
    df = pd.DataFrame(
        {
            "permno": np.tile(np.arange(firms), months),
            "yyyymm": np.repeat(yyyymm, firms),
            "BM": rng.normal(size=firms * months),
            "Mom": rng.normal(size=firms * months),
        }
    )
    df.loc[rng.random(len(df)) < 0.1, "BM"] = np.nan
    return df


def test_rank_normalize_does_not_use_future_months():
    assert_future_invariant(
        lambda p: rank_normalize(p, ["BM", "Mom"]), _random_panel(), cutoff=200012
    )


def test_invariance_check_catches_a_leaky_normalizer():
    """Negative control: ranking over the pooled panel must be flagged."""

    def leaky(p: pd.DataFrame) -> pd.DataFrame:
        return p.assign(BM=p["BM"].rank(pct=True))

    with pytest.raises(FutureLeak):
        assert_future_invariant(leaky, _random_panel(), cutoff=200012)


def test_align_targets_up_to_a_month_do_not_depend_on_later_returns():
    signals = _random_panel()[["permno", "yyyymm", "BM"]]
    returns = _random_panel(seed=1).rename(columns={"Mom": "ret_excess"})[
        ["permno", "yyyymm", "ret_excess"]
    ]
    assert_future_invariant(lambda r: align(signals, r), returns, cutoff=200012)


def test_integer_columns_do_not_raise_false_alarms():
    panel = _random_panel().assign(exchcd=1)  # int64, like CRSP codes
    assert_future_invariant(lambda p: p, panel, cutoff=200012)


def test_invariance_check_catches_a_filter_on_which_future_rows_exist():
    """Negative control: survivorship ("keep firms alive in the last month") leaks via row presence."""

    def survivors_only(p: pd.DataFrame) -> pd.DataFrame:
        alive = p.loc[p["yyyymm"] == p["yyyymm"].max(), "permno"]
        return p[p["permno"].isin(alive)]

    panel = _random_panel()
    panel = panel[~((panel["permno"] == 3) & (panel["yyyymm"] > 200012))]  # firm 3 delists in 2001
    with pytest.raises(FutureLeak):
        assert_future_invariant(survivors_only, panel, cutoff=200012)


def test_leak_report_names_the_first_changed_cell():
    def leaky(p: pd.DataFrame) -> pd.DataFrame:
        return p.assign(BM=p["BM"].rank(pct=True))

    with pytest.raises(FutureLeak, match=r"column 'BM'.*permno \d+, yyyymm \d{6}"):
        assert_future_invariant(leaky, _random_panel(), cutoff=200012)


def test_float32_stays_float32_and_unranked_columns_are_untouched():
    """Memory: a float64 upcast doubles the ~3.7 GB panel (council 2, measured)."""
    panel = _random_panel().astype({"BM": "float32", "Mom": "float32"})
    out = rank_normalize(panel, ["BM"])
    assert out["BM"].dtype == "float32"
    assert out["Mom"].equals(panel["Mom"])


def test_ties_share_a_rank_and_a_single_firm_month_maps_to_zero():
    panel = pd.DataFrame(
        {
            "permno": [1, 2, 3, 9],
            "yyyymm": [200001, 200001, 200001, 200002],
            "BM": [5.0, 5.0, 9.0, 3.0],
        }
    )
    assert rank_normalize(panel, ["BM"])["BM"].tolist() == [-0.5, -0.5, 1.0, 0.0]
