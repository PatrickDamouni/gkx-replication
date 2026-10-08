"""Phase 1b: CRSP return construction (D019), on hand-built rows. Expected values worked by hand."""

import numpy as np
import pandas as pd
import pytest

from gkx.crsp import QUERY_MSF, build_returns
from gkx.panel import HOLDOUT_START


def _msf(rows):
    df = pd.DataFrame(rows, columns=["permno", "date", "ret", "prc", "shrout"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _names(rows):
    df = pd.DataFrame(rows, columns=["permno", "namedt", "nameendt", "shrcd", "exchcd"])
    df[["namedt", "nameendt"]] = df[["namedt", "nameendt"]].apply(pd.to_datetime)
    return df


def _delist(rows):
    df = pd.DataFrame(rows, columns=["permno", "dlstdt", "dlstcd", "dlret"])
    df["dlstdt"] = pd.to_datetime(df["dlstdt"])
    return df


RF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01", "2000-05-01"]),
        "rf": [0.004, 0.005, 0.006, 0.007],
    }
)
NYSE_COMMON = ("1990-01-01", "2010-12-31", 10, 1)


def _one(out, permno, yyyymm, col):
    return out.set_index(["permno", "yyyymm"]).loc[(permno, yyyymm), col]


def test_excess_return_and_market_equity_in_millions():
    msf = _msf([(1, "2000-01-31", 0.02, -10.0, 1000), (1, "2000-02-29", 0.03, 11.0, 1000)])
    out = build_returns(msf, _delist([]), _names([(1, *NYSE_COMMON)]), RF)
    assert _one(out, 1, 200002, "ret_excess") == pytest.approx(0.03 - 0.005)
    assert _one(out, 1, 200001, "me") == pytest.approx(
        10.0
    )  # |−10| × 1000 / 1000; bid-ask midpoint sign
    assert _one(out, 1, 200002, "me_lag") == pytest.approx(10.0)  # ME at the signal month


def test_me_lag_is_nan_across_a_gap_not_the_last_available_month():
    msf = _msf([(1, "2000-03-31", 0.01, 10.0, 1000), (1, "2000-05-31", 0.01, 12.0, 1000)])
    out = build_returns(msf, _delist([]), _names([(1, *NYSE_COMMON)]), RF)
    assert np.isnan(_one(out, 1, 200005, "me_lag"))


def test_delisting_return_is_compounded_into_the_delisting_month():
    msf = _msf([(3, "2000-02-29", 0.01, 5.0, 100)])
    out = build_returns(
        msf, _delist([(3, "2000-02-15", 231, 0.05)]), _names([(3, *NYSE_COMMON)]), RF
    )
    assert _one(out, 3, 200002, "ret") == pytest.approx(1.01 * 1.05 - 1)


def test_missing_performance_dlret_is_imputed_by_exchange():
    nasdaq = ("1990-01-01", "2010-12-31", 11, 3)
    msf = _msf([(2, "2000-02-29", 0.10, 2.0, 100), (7, "2000-02-29", 0.10, 2.0, 100)])
    delist = _delist([(2, "2000-02-15", 560, np.nan), (7, "2000-02-15", 552, np.nan)])
    out = build_returns(msf, delist, _names([(2, *nasdaq), (7, *NYSE_COMMON)]), RF)
    assert _one(out, 2, 200002, "ret") == pytest.approx(1.10 * (1 - 0.55) - 1)  # NASDAQ: −55%
    assert _one(out, 7, 200002, "ret") == pytest.approx(1.10 * (1 - 0.30) - 1)  # NYSE/AMEX: −30%


def test_missing_regular_return_takes_the_delisting_return_alone():
    msf = _msf([(5, "2000-02-29", np.nan, np.nan, 100)])
    out = build_returns(
        msf, _delist([(5, "2000-02-10", 580, -0.2)]), _names([(5, *NYSE_COMMON)]), RF
    )
    assert _one(out, 5, 200002, "ret") == pytest.approx(-0.2)


def test_non_performance_delisting_with_missing_dlret_is_not_imputed():
    msf = _msf([(8, "2000-02-29", 0.02, 5.0, 100)])
    out = build_returns(
        msf, _delist([(8, "2000-02-15", 231, np.nan)]), _names([(8, *NYSE_COMMON)]), RF
    )
    assert _one(out, 8, 200002, "ret") == pytest.approx(0.02)


def test_universe_is_common_shares_on_nyse_amex_nasdaq_as_of_each_month():
    msf = _msf(
        [
            (4, "2000-01-31", 0.01, 5.0, 100),  # shrcd 12: not common stock
            (6, "2000-01-31", 0.01, 5.0, 100),
            (6, "2000-03-31", 0.01, 5.0, 100),
        ]
    )
    names = _names(
        [
            (4, "1990-01-01", "2010-12-31", 12, 1),
            (6, "1990-01-01", "2000-02-15", 10, 1),
            (6, "2000-02-16", "2010-12-31", 10, 4),
        ]  # moves off-exchange
    )
    out = build_returns(msf, _delist([]), names, RF)
    assert set(zip(out["permno"], out["yyyymm"], strict=True)) == {(6, 200001)}


def test_the_sql_seals_the_holdout_server_side():
    year = HOLDOUT_START // 100
    assert f"date < '{year}-01-01'" in QUERY_MSF
