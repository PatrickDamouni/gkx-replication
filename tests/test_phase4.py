"""Phase 4 outputs are always labelled exploratory and never reach past 2016 (D049)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/phase4_diagnostics.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase4", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_output_carries_the_exploratory_banner_and_the_seal_month():
    mod = _load()
    assert mod.exploratory("x").startswith("> **EXPLORATORY (D049).**")
    assert mod.LAST_MONTH == 201612
    assert "sealed" not in SCRIPT.read_text().replace("seal)", "")


def test_nyse_quintiles_use_nyse_breakpoints_only():
    mod = _load()
    df = pd.DataFrame(
        {
            "yyyymm": [1] * 7,
            "exchcd": [1, 1, 1, 1, 1, 3, 3],
            "me_lag": [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 100.0],
        }
    )
    q = mod.nyse_quintiles(df)
    assert q[5] == 0 and q[6] == 4  # a tiny and a huge NASDAQ stock fall in the NYSE extremes
    assert sorted(q[:5]) == [0, 1, 2, 3, 4]


def test_months_helper_sums_by_month():
    mod = _load()
    mo = mod.Months(np.array([3, 1, 3, 2]))
    assert mo.sum(np.array([1.0, 2.0, 3.0, 4.0])).tolist() == [2.0, 4.0, 4.0]
    assert mo.n.tolist() == [1, 1, 2]
