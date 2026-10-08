"""CRSP-rebuilt characteristics (D024 rule 1). Windows were identified by reproducing datashare
(results/crsp_chars_validation.md); these tests pin them with hand-placed returns.

All windows are relative to the signal month s (characteristics known at end of s).
"""

import numpy as np
import pandas as pd
import pytest

from gkx.chars_crsp import crsp_characteristics

S = 200312  # signal month under test


def _firm(returns_at: dict[int, float], me: float = 50.0) -> pd.DataFrame:
    """One firm, 48 consecutive months ending at S; zero return except where placed (offset from S)."""
    months = [(S // 100 - 4) * 100 + 12]  # 199912
    while months[-1] != S:
        y, m = divmod(months[-1], 100)
        months.append(y * 100 + m + 1 if m < 12 else (y + 1) * 100 + 1)
    offsets = {m: len(months) - 1 - i for i, m in enumerate(months)}  # S -> 0, S-1 -> 1, ...
    return pd.DataFrame(
        {
            "permno": 1,
            "yyyymm": months,
            "ret": [returns_at.get(offsets[m], 0.0) for m in months],
            "me": me,
        }
    )


@pytest.fixture
def at_s() -> pd.Series:
    placed = {0: 0.10, 1: 0.20, 6: 0.30, 11: 0.70, 12: 0.50}  # offset k means month S-k
    out = crsp_characteristics(_firm(placed))
    return out.set_index("yyyymm").loc[S]


def test_mom1m_is_the_return_of_the_signal_month(at_s):
    assert at_s["mom1m"] == pytest.approx(0.10)


def test_mom6m_covers_s_minus_5_to_s_minus_1(at_s):
    assert at_s["mom6m"] == pytest.approx(0.20)  # excludes S (0.10) and S-6 (0.30)


def test_mom12m_covers_s_minus_11_to_s_minus_1(at_s):
    assert at_s["mom12m"] == pytest.approx(1.2 * 1.3 * 1.7 - 1)  # S-1, S-6, S-11; not S-12


def test_mom36m_covers_s_minus_35_to_s_minus_12(at_s):
    assert at_s["mom36m"] == pytest.approx(0.50)  # S-12 only; S-11 (0.70) excluded


def test_chmom_includes_month_s_unlike_mom6m(at_s):
    """GHZ quirk reproduced by datashare: (S-5..S) − (S-11..S-6)."""
    assert at_s["chmom"] == pytest.approx((1.1 * 1.2 - 1) - (1.3 * 1.7 - 1))


def test_mvel1_is_market_equity_in_thousands(at_s):
    assert at_s["mvel1"] == pytest.approx(50_000.0)  # me is $ millions in our CRSP build


def test_a_missing_month_inside_a_window_gives_nan_not_a_shorter_window():
    firm = _firm({1: 0.20})
    firm = firm[firm["yyyymm"] != 200309]  # S-3 missing
    row = crsp_characteristics(firm).set_index("yyyymm").loc[S]
    assert np.isnan(row["mom6m"]) and row["mom1m"] == 0.0
