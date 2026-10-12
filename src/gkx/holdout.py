"""Phase 5 (D050): the only code that opens the sealed holdout.

Every sealed input is sha256-checked before it is read. The holdout panel holds return months
>= 201701 only and is appended to the frozen panel, so every pre-2017 row reaches the models
byte-for-byte as in freeze-v1 and in the same order (row order drives NN batching and tree
binning). January 2017's row is built from signal month 201612, whose characteristics, market
equity and macro values live in the *pre-2017* files (datashare splits on signal month).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gkx.align import align, next_month
from gkx.panel import HOLDOUT_START, load_panel
from gkx.preprocess import rank_normalize

LAST_PRE = HOLDOUT_START - 100 + 11  # 201612: the signal month of the first holdout return month
FROZEN_PANEL = Path("data/processed/panel_datashare.parquet")
FROZEN_PANEL_SHA = "dfffda9aada2ddddbd9a74ecf1acdabaa91ea74f40f38f039a83b796ec1d56fc"  # D044
HOLDOUT_PANEL = Path("data/processed/panel_holdout.parquet")
PANEL_LOCK = Path("results/holdout_panel.sha256")
SEALED = {
    "datashare": (
        Path("data/sealed/datashare_holdout.parquet"),
        "9a87fbf7153c2a4d6d61ac93196df24f59a6f4d590566d2e780a0965e5a1881b",  # D023
    ),
    "macro": (
        Path("data/sealed/goyal_macro_holdout.parquet"),
        "4d2640cc569645deaa94cab27bd51ac32e3469eb72371660a3fd07110360448e",  # D021
    ),
}
CRSP_HOLDOUT = Path("data/sealed/crsp_returns_holdout.parquet")
# Set from the one-time pull (scripts/pull_crsp_holdout.py --pull) before freeze-v2; D050 §3.
CRSP_HOLDOUT_SHA = "unset: pull not yet run"


class HashMismatchError(Exception):
    """A file's sha256 differs from the one recorded before the holdout was opened."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise HashMismatchError(f"{path}: sha256 {actual} != recorded {expected}")


def build_panel_through(
    chars: pd.DataFrame,
    crsp: pd.DataFrame,
    macro: pd.DataFrame,
    char_cols: list[str],
    keep: list[str] | None,
    through: int,
) -> pd.DataFrame:
    """`gkx.build.build_panel` line for line, except that `align` keeps return months through
    `through`. tests/test_holdout.py checks it equals build_panel when through = 201612."""
    at_t = crsp[["permno", "yyyymm", "me", "shrcd"]].rename(columns={"me": "me_lag"})
    signals = chars[["permno", "yyyymm", *char_cols, *(keep or [])]].merge(
        at_t, on=["permno", "yyyymm"], how="inner", validate="one_to_one"
    )
    signals = rank_normalize(signals, char_cols)
    panel = align(signals, crsp[["permno", "yyyymm", "ret_excess"]], through=through)
    for col in ("target", "me_lag"):
        panel[col] = panel[col].to_numpy(dtype="float64", na_value=np.nan)
    m = macro.rename(columns=lambda c: c if c == "yyyymm" else f"m_{c}")
    overlap = (set(m.columns) - {"yyyymm"}) & set(panel.columns)
    if overlap:
        raise ValueError(f"macro columns collide with panel columns: {sorted(overlap)}")
    return panel.merge(
        m.rename(columns={"yyyymm": "signal_month"}),
        on="signal_month",
        how="left",
        validate="many_to_one",
    )


def build_holdout_panel(
    ds_pre: pd.DataFrame,
    ds_hold: pd.DataFrame,
    crsp_pre: pd.DataFrame,
    crsp_hold: pd.DataFrame,
    macro_pre: pd.DataFrame,
    macro_hold: pd.DataFrame,
    char_cols: list[str],
) -> pd.DataFrame:
    """Rows for return months 201701 → the last month with characteristics, macro and CRSP returns.

    Signal month 201612 comes from the pre-2017 files; later signal months from the sealed ones.
    Pre-2017 CRSP (December-2016 market equity) always comes from the frozen file (D050 §3).
    """
    chars = pd.concat([ds_pre[ds_pre["yyyymm"] == LAST_PRE], ds_hold], ignore_index=True)
    crsp = pd.concat(
        [crsp_pre[crsp_pre["yyyymm"] == LAST_PRE], crsp_hold[crsp_hold["yyyymm"] >= HOLDOUT_START]],
        ignore_index=True,
    )
    macro = pd.concat([macro_pre[macro_pre["yyyymm"] == LAST_PRE], macro_hold], ignore_index=True)
    last_signal = min(int(chars["yyyymm"].max()), int(macro["yyyymm"].max()))
    through = min(int(next_month(pd.Series([last_signal]))[0]), int(crsp["yyyymm"].max()))
    panel = build_panel_through(chars, crsp, macro, char_cols, ["sic2"], through)
    panel = panel[panel["yyyymm"] >= HOLDOUT_START]
    return panel.sort_values(["yyyymm", "permno"]).reset_index(drop=True)


def conform(panel: pd.DataFrame, frozen: Path = FROZEN_PANEL) -> pd.DataFrame:
    """Same columns, order and dtypes as the frozen panel, or fail (D050 §7)."""
    template = pd.read_parquet(frozen, filters=[("yyyymm", "==", 201612)])
    if set(panel.columns) != set(template.columns):
        raise ValueError(
            f"holdout columns differ: missing {sorted(set(template.columns) - set(panel.columns))}, "
            f"extra {sorted(set(panel.columns) - set(template.columns))}"
        )
    return panel[list(template.columns)].astype(template.dtypes.to_dict())


def load_merged(
    frozen: Path = FROZEN_PANEL,
    holdout: Path = HOLDOUT_PANEL,
    lock: Path = PANEL_LOCK,
    frozen_sha: str = FROZEN_PANEL_SHA,
) -> pd.DataFrame:
    """The frozen panel exactly as freeze-v1 read it, followed by the locked holdout panel."""
    verify(frozen, frozen_sha)
    verify(holdout, lock.read_text().split()[0])
    if pq.read_schema(frozen).remove_metadata() != pq.read_schema(holdout).remove_metadata():
        raise ValueError("holdout panel schema differs from the frozen panel")
    pre = load_panel(frozen)  # the same call scripts/run_test.py makes
    post = pd.read_parquet(holdout)
    if post["yyyymm"].min() < HOLDOUT_START:
        raise ValueError("holdout panel contains pre-2017 rows")
    return pd.concat([pre, post], ignore_index=True)
