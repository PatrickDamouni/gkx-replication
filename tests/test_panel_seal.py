"""Seam 1: the 2017+ holdout is sealed at load time (PREREGISTRATION.md §Splits, D008)."""

import pandas as pd
import pytest

from gkx.panel import SealedHoldoutError, load_panel


@pytest.fixture
def straddling_parquet(tmp_path):
    """Three months either side of the seal: 2016-11, 2016-12, 2017-01."""
    df = pd.DataFrame(
        {
            "permno": [10001, 10001, 10001, 10002],
            "yyyymm": [201611, 201612, 201701, 201612],
            "BM": [0.5, 0.6, 0.7, 1.1],
        }
    )
    path = tmp_path / "panel.parquet"
    df.to_parquet(path)
    return path


def test_load_panel_never_returns_holdout_months(straddling_parquet):
    df = load_panel(straddling_parquet)
    assert df["yyyymm"].max() == 201612
    assert len(df) == 3


def test_requesting_a_holdout_month_raises(straddling_parquet):
    with pytest.raises(SealedHoldoutError, match="201701"):
        load_panel(straddling_parquet, end=201701)


def test_split_at_ingest_partitions_rows_and_fingerprints_the_holdout(straddling_parquet, tmp_path):
    import hashlib

    from gkx.panel import split_holdout

    pre, holdout = tmp_path / "pre.parquet", tmp_path / "holdout.parquet"
    digest = split_holdout(straddling_parquet, pre, holdout)

    assert sorted(pd.read_parquet(pre)["yyyymm"]) == [201611, 201612, 201612]
    assert list(pd.read_parquet(holdout)["yyyymm"]) == [201701]
    assert digest == hashlib.sha256(holdout.read_bytes()).hexdigest()


def test_split_refuses_to_lose_rows(tmp_path):
    """A null yyyymm matches neither side of the split; that must fail loudly, not vanish."""
    from gkx.panel import split_holdout

    src = tmp_path / "src.parquet"
    pd.DataFrame(
        {
            "permno": [1, 2],
            "yyyymm": pd.array([201612, None], dtype="Int64"),
            "BM": [0.1, 0.2],
        }
    ).to_parquet(src)
    with pytest.raises(ValueError, match="1 of 2 rows"):
        split_holdout(src, tmp_path / "pre.parquet", tmp_path / "hold.parquet")
