"""Phase 4 outputs are always labelled exploratory and never reach past 2016 (D049)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "exploratory/phase4_diagnostics.py"


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


def test_sse_split_reconstructs_total():
    mod = _load()
    y = np.array([0.1, -0.2, 0.3, 0.05])
    base = pd.DataFrame({"a": [0.0, 0.1, 0.0, 0.2], "train_mean": 0.0, "NN3": 0.0})
    mo = mod.Months(np.array([1, 1, 2, 2]))
    df, _ = mod.level_vs_ranking(base, ["a", "train_mean", "NN3"], mo, y, (y**2).sum())
    row = df.set_index("model").loc["a"]
    total = 100 * ((y - base["a"]) ** 2).sum() / (y**2).sum()
    assert np.isclose(row.iloc[1] + row.iloc[2], total)
    # residuals y - a = [0.1, -0.3 | 0.3, -0.15]; month means -0.1 and 0.075, two rows each
    between = 100 * (2 * 0.1**2 + 2 * 0.075**2) / (y**2).sum()
    assert np.isclose(row.iloc[1], between)
