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
        {"permno": [1, 2, 3, 4], "yyyymm": [200001] * 4, "BM": [10.0, 30.0, 20.0, np.nan]}
    )
    out = rank_normalize(panel, ["BM"])
    assert out["BM"].tolist() == [-1.0, 1.0, 0.0, 0.0]


def test_each_month_is_ranked_on_its_own():
    panel = pd.DataFrame(
        {"permno": [1, 2, 1, 2], "yyyymm": [200001, 200001, 200002, 200002], "BM": [1.0, 2.0, 100.0, 50.0]}
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
    assert_future_invariant(lambda p: rank_normalize(p, ["BM", "Mom"]), _random_panel(), cutoff=200012)


def test_invariance_check_catches_a_leaky_normalizer():
    """Negative control: ranking over the pooled panel must be flagged."""

    def leaky(p: pd.DataFrame) -> pd.DataFrame:
        return p.assign(BM=p["BM"].rank(pct=True))

    with pytest.raises(FutureLeak):
        assert_future_invariant(leaky, _random_panel(), cutoff=200012)


def test_align_labels_for_early_months_do_not_depend_on_later_returns():
    signals = _random_panel()[["permno", "yyyymm", "BM"]]
    returns = _random_panel(seed=1).rename(columns={"Mom": "ret_excess"})[["permno", "yyyymm", "ret_excess"]]
    # Signals through 200011 may see returns through 200012 (their t+1) and nothing later.
    assert_future_invariant(
        lambda r: align(signals, r), returns, cutoff=200012, keep_rows=lambda o: o["yyyymm"] <= 200011
    )
