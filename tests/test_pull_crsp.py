"""scripts/pull_crsp_holdout.py's pure parts (D050 §3, council 15): no WRDS connection here."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pull_crsp_holdout.py"


def _load():
    spec = importlib.util.spec_from_file_location("pull_crsp_holdout", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(months, ret=0.01):
    df = pd.DataFrame([(p, m) for m in months for p in (1, 2)], columns=["permno", "yyyymm"])
    return df.assign(ret=ret, ret_excess=ret - 0.001, me=5.0, shrcd=10, exchcd=1)


def test_compare_is_exact_and_names_what_differs():
    mod = _load()
    a = _rows([201611, 201612])
    assert mod.compare(a, a.copy(), [201612]) == []
    b = a.copy()
    b.loc[b["yyyymm"] == 201612, "ret"] = 0.02
    assert mod.compare(a, b, [201612]) == ["ret: 2 rows differ"]
    assert "row sets differ" in mod.compare(a, a[a["permno"] == 1], [201612])[0]
    nan = a.assign(me=np.nan)
    assert mod.compare(nan, nan.copy(), [201611, 201612]) == []


def test_holdout_months_must_start_in_january_2017_and_be_contiguous():
    mod = _load()
    assert mod.holdout_months_ok([201701, 201702, 201703])
    assert mod.holdout_months_ok([*(201700 + m for m in range(1, 13)), 201801])
    assert not mod.holdout_months_ok([])
    assert not mod.holdout_months_ok([201702, 201703])
    assert not mod.holdout_months_ok([201701, 201703])


def test_fetch_binds_dates_as_parameters():
    mod = _load()
    calls = []

    class FakeDB:
        def raw_sql(self, sql, date_cols=None, params=None):
            calls.append((sql, params))
            return pd.DataFrame()

    mod.build_returns = lambda *frames: "built"
    assert mod.fetch(FakeDB(), "2016-12-01", "2100-01-01") == "built"
    bounded = [c for c in calls if c[1] is not None]
    assert len(bounded) == 3 and all("2016-12-01" not in sql for sql, _ in calls)
    assert bounded[0][1]["start"].isoformat() == "2016-12-01"
    with pytest.raises(ValueError):
        mod.fetch(FakeDB(), "2016-12-01'; drop table x; --", "2100-01-01")
