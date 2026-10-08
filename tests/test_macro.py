"""Phase 1b: GKX's eight Welch-Goyal macro predictors (D021). Expected values worked by hand."""

import numpy as np
import pandas as pd
import pytest

from gkx.macro import MACRO_COLS, build_macro


def _raw(**overrides):
    row = {
        "yyyymm": 200001,
        "Index": 100.0,
        "D12": 2.0,
        "E12": 5.0,
        "b/m": 0.5,
        "tbl": 0.05,
        "AAA": 0.06,
        "BAA": 0.08,
        "lty": 0.07,
        "ntis": 0.01,
        "svar": 0.002,
    }
    return pd.DataFrame([row | overrides])


def test_eight_predictors_follow_welch_goyal_definitions():
    out = build_macro(_raw()).iloc[0]
    assert list(MACRO_COLS) == ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
    assert out["dp"] == pytest.approx(np.log(0.02))  # log D12 − log Index
    assert out["ep"] == pytest.approx(np.log(0.05))  # log E12 − log Index
    assert out["tms"] == pytest.approx(0.02)  # lty − tbl
    assert out["dfy"] == pytest.approx(0.02)  # BAA − AAA
    assert (out["bm"], out["ntis"], out["tbl"], out["svar"]) == (0.5, 0.01, 0.05, 0.002)


def test_output_is_keyed_on_the_month_the_values_are_known():
    """Macro at month t is joined on the SIGNAL month t, like the characteristics (D014)."""
    assert build_macro(_raw(yyyymm=198612))["yyyymm"].tolist() == [198612]
