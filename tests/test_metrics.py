"""Seam 4: out-of-sample metrics and tests (D005, D010, D015).

Inputs are prediction frames with columns yyyymm, permno, target, <pred cols>.
Expected values are worked by hand, not recomputed the way the code does.
"""

import numpy as np
import pandas as pd
import pytest

from gkx.metrics import clark_west, diebold_mariano, hac_mean_test, r2_oos


def test_r2_oos_hand_example_with_non_demeaned_denominator():
    df = pd.DataFrame(
        {"yyyymm": [200001] * 3, "permno": [1, 2, 3], "target": [0.1, -0.2, 0.3]}
    ).assign(zero=0.0, perfect=lambda d: d["target"], flat=0.1)
    assert r2_oos(df, "zero") == 0.0
    assert r2_oos(df, "perfect") == 1.0
    # SSE = 0 + 0.09 + 0.04 = 0.13; Σr² = 0.01 + 0.04 + 0.09 = 0.14
    assert r2_oos(df, "flat") == pytest.approx(1 - 0.13 / 0.14)


def test_r2_oos_pools_stock_months_rather_than_averaging_months():
    # Month A: one stock, SSE 1, Σr² 1. Month B: three stocks predicted perfectly, Σr² 3.
    df = pd.DataFrame(
        {
            "yyyymm": [200001, 200002, 200002, 200002],
            "permno": [1, 1, 2, 3],
            "target": [1.0, 1.0, 1.0, 1.0],
            "pred": [0.0, 1.0, 1.0, 1.0],
        }
    )
    assert r2_oos(df, "pred") == pytest.approx(0.75)  # mean of monthly R² would be 0.5


def test_hac_with_zero_lags_is_the_plain_t_statistic():
    # mean 2.5, population variance 1.25, se = sqrt(1.25 / 4)
    t, _ = hac_mean_test(pd.Series([1.0, 2.0, 3.0, 4.0]), lags=0)
    assert t == pytest.approx(2.5 / np.sqrt(1.25 / 4))


def _panel(rng: np.random.Generator, months: int = 120, firms: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "yyyymm": np.repeat(np.arange(months) + 190001, firms),
            "permno": np.tile(np.arange(firms), months),
            "x": rng.normal(size=months * firms),
        }
    )


def test_diebold_mariano_is_positive_when_the_candidate_is_better():
    rng = np.random.default_rng(1)
    df = _panel(rng)
    df["target"] = 0.3 * df["x"] + rng.normal(size=len(df))
    df["zero"], df["good"] = 0.0, 0.3 * df["x"]
    t, p = diebold_mariano(df, base="zero", candidate="good")
    assert t > 3 and p < 0.01


def test_clark_west_is_correctly_sized_where_dm_is_not_for_nested_models():
    """Null: the big model only adds estimated noise to the small one (zero forecast nested in it).

    Clark & West (2007): DM is undersized here because the big model's MSE is inflated by
    estimation noise even when it is the true model's equal. CW should reject ~5%.
    """
    rng = np.random.default_rng(7)
    reps, dm_rej, cw_rej = 400, 0, 0
    for _ in range(reps):
        df = _panel(rng, months=60, firms=20)
        df["target"] = rng.normal(size=len(df))
        df["small"] = 0.0
        df["big"] = 0.05 * rng.normal(size=len(df))  # fitted noise, no signal
        dm_rej += diebold_mariano(df, base="small", candidate="big")[1] < 0.05
        cw_rej += clark_west(df, small="small", big="big")[1] < 0.05
    assert dm_rej / reps < 0.02
    assert 0.02 < cw_rej / reps < 0.10
