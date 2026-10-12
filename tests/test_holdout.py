"""Phase 5 (D050): seal-keeping defaults and the opt-in holdout paths."""

import pandas as pd
import pytest

from gkx.align import align
from gkx.panel import SealedHoldoutError
from gkx.walk_forward import walk_forward


def test_holdout_folds_only_when_explicitly_allowed():
    """D050: the Phase 5 runner opts in; the default still seals."""
    with pytest.raises(SealedHoldoutError):
        walk_forward(first_test_year=2016, last_test_year=2018)
    folds = walk_forward(first_test_year=2016, last_test_year=2018, allow_holdout=True)
    assert [f.refit for f in folds] == [2016, 2017, 2018]
    assert folds[1].train == (195703, 200412) and folds[1].val == (200501, 201612)
    assert folds[0] == walk_forward()[-1]  # the 2016 canary fold is the frozen 2016 fold


def test_align_keeps_the_seal_by_default_and_opens_only_through_a_given_month():
    sig = pd.DataFrame(
        {"permno": [1, 1, 1], "yyyymm": [201611, 201612, 201701], "x": [1.0, 2.0, 3.0]}
    )
    ret = pd.DataFrame(
        {"permno": [1, 1, 1], "yyyymm": [201612, 201701, 201702], "ret_excess": [0.1, 0.2, 0.3]}
    )
    assert align(sig, ret)["yyyymm"].tolist() == [201612]
    opened = align(sig, ret, through=201702)
    assert opened["yyyymm"].tolist() == [201612, 201701, 201702]
    assert opened["target"].tolist() == [0.1, 0.2, 0.3]


# --- gkx.holdout: the builder and the merged loader, on synthetic data only -----------------------

from pathlib import Path

import numpy as np

from gkx.build import build_panel
from gkx.holdout import (
    HashMismatchError,
    build_holdout_panel,
    build_panel_through,
    load_merged,
    sha256,
    verify,
)

CHARS = ["x1", "x2"]


def _chars(months, seed):
    rng = np.random.default_rng(seed)
    rows = [(p, m) for m in months for p in (1, 2, 3, 4)]
    df = pd.DataFrame(rows, columns=["permno", "yyyymm"]).astype("int32")
    for c in CHARS:
        df[c] = rng.normal(size=len(df)).astype("float32")
    df["sic2"] = pd.array([10, 20, 30, 40] * len(months), dtype="Int8")
    return df


def _crsp(months, me):
    rows = [(p, m) for m in months for p in (1, 2, 3, 4)]
    df = pd.DataFrame(rows, columns=["permno", "yyyymm"])
    df["ret_excess"] = 0.01 * df["permno"] + df["yyyymm"] % 100 / 1000
    df["me"] = me * df["permno"].astype(float)
    df["shrcd"] = 10
    return df


def _macro(months):
    return pd.DataFrame({"yyyymm": months, "dp": np.linspace(0.1, 0.2, len(months))})


PRE, HOLD = [201610, 201611, 201612], [201701, 201702, 201703]


def test_build_panel_through_equals_the_frozen_builder_up_to_the_seal():
    chars, crsp, macro = _chars(PRE, 0), _crsp(PRE, 1.0), _macro(PRE)
    frozen = build_panel(chars, crsp, macro, char_cols=CHARS, keep=["sic2"])
    mirror = build_panel_through(chars, crsp, macro, CHARS, ["sic2"], through=201612)
    pd.testing.assert_frame_equal(frozen, mirror)


def test_holdout_panel_starts_in_january_2017_from_the_pre_2017_signal_month():
    ds_pre, ds_hold = _chars(PRE, 0), _chars(HOLD, 1)
    crsp_pre = _crsp(PRE, 1.0)
    crsp_hold = _crsp([201612, *HOLD, 201704], 99.0)  # the pull starts at 2016-12 (D050 §3)
    out = build_holdout_panel(
        ds_pre, ds_hold, crsp_pre, crsp_hold, _macro(PRE), _macro(HOLD), CHARS
    )
    assert out["yyyymm"].min() == 201701 and out["yyyymm"].max() == 201704
    jan = out[out["yyyymm"] == 201701]
    assert (jan["signal_month"] == 201612).all() and len(jan) == 4
    # December-2016 market equity comes from the frozen file, never from the new pull
    assert jan["me_lag"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out[out["yyyymm"] == 201702]["me_lag"].tolist() == [99.0, 198.0, 297.0, 396.0]
    assert out[["yyyymm", "permno"]].equals(
        out[["yyyymm", "permno"]].sort_values(["yyyymm", "permno"]).reset_index(drop=True)
    )


def test_holdout_ends_at_the_last_month_with_returns_and_macro():
    ds_pre, ds_hold = _chars(PRE, 0), _chars(HOLD, 1)
    short_crsp = build_holdout_panel(
        ds_pre, ds_hold, _crsp(PRE, 1.0), _crsp([201612, 201701, 201702], 9.0),
        _macro(PRE), _macro(HOLD), CHARS,
    )  # fmt: skip
    assert short_crsp["yyyymm"].max() == 201702  # a partial final year is kept, not an error
    short_macro = build_holdout_panel(
        ds_pre, ds_hold, _crsp(PRE, 1.0), _crsp([201612, *HOLD, 201704], 9.0),
        _macro(PRE), _macro([201701]), CHARS,
    )  # fmt: skip
    assert short_macro["yyyymm"].max() == 201702


def test_sha_checks_fail_closed(tmp_path):
    f = tmp_path / "a.parquet"
    f.write_bytes(b"abc")
    verify(f, sha256(f))
    with pytest.raises(HashMismatchError):
        verify(f, "0" * 64)


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_parquet(path, index=False)
    return path


def test_load_merged_appends_the_locked_holdout_after_the_frozen_rows(tmp_path):
    pre = _write(pd.DataFrame({"permno": [1, 2], "yyyymm": [201612, 201612], "x": [0.1, 0.2]}),
                 tmp_path / "pre.parquet")  # fmt: skip
    post = _write(pd.DataFrame({"permno": [1], "yyyymm": [201701], "x": [0.3]}),
                  tmp_path / "post.parquet")  # fmt: skip
    lock = tmp_path / "lock"
    lock.write_text(sha256(post) + "\n")
    out = load_merged(pre, post, lock, frozen_sha=sha256(pre))
    assert out["yyyymm"].tolist() == [201612, 201612, 201701]
    lock.write_text("0" * 64)
    with pytest.raises(HashMismatchError):
        load_merged(pre, post, lock, frozen_sha=sha256(pre))
    with pytest.raises(HashMismatchError):
        load_merged(pre, post, lock, frozen_sha="0" * 64)


def test_load_merged_refuses_a_schema_change_or_pre_2017_rows(tmp_path):
    pre = _write(
        pd.DataFrame({"permno": [1], "yyyymm": [201612], "x": [0.1]}), tmp_path / "pre.parquet"
    )
    bad = _write(
        pd.DataFrame({"permno": [1], "yyyymm": [201701], "y": [0.3]}), tmp_path / "bad.parquet"
    )
    lock = tmp_path / "lock"
    lock.write_text(sha256(bad))
    with pytest.raises(ValueError, match="schema"):
        load_merged(pre, bad, lock, frozen_sha=sha256(pre))
    early = _write(
        pd.DataFrame({"permno": [1], "yyyymm": [201612], "x": [0.3]}), tmp_path / "e.parquet"
    )
    lock.write_text(sha256(early))
    with pytest.raises(ValueError, match="pre-2017"):
        load_merged(pre, early, lock, frozen_sha=sha256(pre))


def test_conform_matches_the_frozen_column_order_and_dtypes(tmp_path):
    from gkx.holdout import conform

    frozen = _write(
        pd.DataFrame(
            {"permno": pd.array([1], dtype="int32"), "yyyymm": pd.array([201612], dtype="int32"),
             "x": pd.array([0.5], dtype="float32"), "sic2": pd.array([10], dtype="Int8")}
        ),
        tmp_path / "frozen.parquet",
    )  # fmt: skip
    new = pd.DataFrame({"sic2": [20], "x": [0.25], "yyyymm": [201701], "permno": [7]})
    out = conform(new, frozen)
    assert list(out.columns) == ["permno", "yyyymm", "x", "sic2"]
    assert out.dtypes.to_dict() == pd.read_parquet(frozen).dtypes.to_dict()
    with pytest.raises(ValueError, match="missing"):
        conform(new.drop(columns="x"), frozen)
