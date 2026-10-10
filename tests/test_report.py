"""Pre-registered final-analysis helpers (D005, D015, D030, D036). Hand-computed expectations."""

import numpy as np
import pandas as pd
import pytest

from gkx.report import CRASH_MONTHS, demeaned_r2, r2_excluding, sharpe_ci, size_masks


def _df():
    return pd.DataFrame(
        {
            "yyyymm": [200001, 200001, 200002, 200002],
            "permno": [1, 2, 1, 2],
            "target": [0.03, 0.01, -0.02, 0.04],
            "pred": [0.02, 0.00, 0.01, 0.01],
            "me_lag": [5.0, 1.0, 5.0, 1.0],
        }
    )


def test_demeaned_r2_by_hand():
    # Month 1: target dm (+0.01, −0.01), pred dm (+0.01, −0.01) → errors 0, 0.
    # Month 2: target dm (−0.03, +0.03), pred dm (0, 0) → errors −0.03, +0.03.
    # 1 − (0.0009·2) / (0.0001·2 + 0.0009·2) = 1 − 0.0018/0.0020 = 0.1
    assert demeaned_r2(_df(), "pred") == pytest.approx(0.1)


def test_r2_excluding_the_crash_months():
    df = _df().assign(yyyymm=[198711, 198711, 200002, 200002])
    assert CRASH_MONTHS == (198711, 200811)
    # only month 200002 remains: SSE = 0.03² + 0.03² = 0.0018; Σr² = 0.0004 + 0.0016 = 0.002
    assert r2_excluding(df, "pred", CRASH_MONTHS) == pytest.approx(1 - 0.0018 / 0.002)


def test_size_masks_pick_largest_and_smallest_by_formation_me_each_month():
    df = pd.DataFrame(
        {"yyyymm": [1] * 5 + [2] * 3, "permno": range(8), "me_lag": [5, 1, 4, 2, 3, 9, 8, 7.0]}
    )
    top, bottom = size_masks(df, n=2)
    assert list(df.loc[top, "permno"]) == [0, 2, 5, 6]
    assert list(df.loc[bottom, "permno"]) == [1, 3, 6, 7]


def test_sharpe_ci_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    s = pd.Series(0.01 + rng.normal(0, 0.04, 240))
    point, lo, hi = sharpe_ci(s, n_boot=2000)
    assert lo < point < hi
    assert sharpe_ci(s, n_boot=2000) == (point, lo, hi)
