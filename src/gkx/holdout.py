"""Phase 5 (D050): the code that opens the sealed holdout, with the scripts that call it.

Every sealed input is sha256-checked before it is read. The holdout panel holds return months
>= 201701 only and is appended to the frozen panel, so every pre-2017 row reaches the models
byte-for-byte as in freeze-v1 and in the same order (row order drives NN batching and tree
binning). January 2017's row is built from signal month 201612, whose characteristics, market
equity and macro values live in the *pre-2017* files (datashare splits on signal month).
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gkx.align import align, next_month
from gkx.datashare import prev_month
from gkx.panel import HOLDOUT_START, load_panel
from gkx.preprocess import rank_normalize

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


REPO = Path(__file__).resolve().parents[2]
ALLOWED_TAGS = ("freeze-v2", "freeze-v2.1")  # D050 §7: one fix at most
# Annotated tag whose message is the holdout panel sha256; pushed before any holdout fit.
LOCK_TAG = "phase5-panel"


class FreezeError(Exception):
    """Holdout code run from anything but a clean, tagged freeze-v2 checkout."""


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise FreezeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def require_freeze_v2() -> str:
    """The tag on HEAD (freeze-v2 or freeze-v2.1) with a clean tree outside results/, or raise.

    Called before anything reads sealed data or holdout predictions (D050 §6-7, council 15)."""
    tags = [t for t in _git("tag", "--points-at", "HEAD").split() if t in ALLOWED_TAGS]
    if not tags:
        raise FreezeError(f"HEAD carries none of {ALLOWED_TAGS}")
    if _git("status", "--porcelain", "--", ".", ":!results"):
        raise FreezeError("uncommitted changes outside results/")
    return max(tags)


def require_published_lock(lock: Path = PANEL_LOCK) -> str:
    """The lock's sha256, which must equal the message of the pushed annotated tag phase5-panel."""
    digest = lock.read_text().split()[0]
    message = _git("tag", "-l", "--format=%(contents)", LOCK_TAG).strip()
    if digest not in message.split():
        raise FreezeError(f"tag {LOCK_TAG} does not carry the holdout panel sha256 {digest}")
    if not _git("ls-remote", "--tags", "origin", f"refs/tags/{LOCK_TAG}"):
        raise FreezeError(f"tag {LOCK_TAG} is not pushed to origin")
    return digest


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


def _months(start: int, end: int) -> list[int]:
    out = [start]
    while out[-1] < end:
        out.append(int(next_month(pd.Series([out[-1]]))[0]))
    return out


def build_holdout_panel(
    ds_pre: pd.DataFrame,
    ds_hold: pd.DataFrame,
    crsp_pre: pd.DataFrame,
    crsp_hold: pd.DataFrame,
    macro_pre: pd.DataFrame,
    macro_hold: pd.DataFrame,
    char_cols: list[str],
    first_return: int = HOLDOUT_START,
) -> pd.DataFrame:
    """Return months `first_return` → the last month with characteristics, macro and a CRSP return.

    The signal month before `first_return` comes from the "pre" files (for the real holdout: the
    pre-2017 files, including December-2016 market equity, D050 §3); later months from the "hold"
    files. `first_return` exists so the same code can be rehearsed on real pre-2017 data with 2016
    playing the holdout (scripts/build_panel_holdout.py --verify-mirror). Fails rather than build
    an empty, gappy or NaN-macro panel.
    """
    last_pre = int(prev_month(pd.Series([first_return]))[0])
    for name, df in (("characteristics", ds_pre), ("CRSP", crsp_pre), ("macro", macro_pre)):
        if not (df["yyyymm"] == last_pre).any():
            raise ValueError(f"pre-holdout {name} file has no rows for signal month {last_pre}")
    chars = pd.concat(
        [ds_pre[ds_pre["yyyymm"] == last_pre], ds_hold[ds_hold["yyyymm"] > last_pre]],
        ignore_index=True,
    )
    crsp = pd.concat(
        [crsp_pre[crsp_pre["yyyymm"] == last_pre], crsp_hold[crsp_hold["yyyymm"] > last_pre]],
        ignore_index=True,
    )
    macro = pd.concat(
        [macro_pre[macro_pre["yyyymm"] == last_pre], macro_hold[macro_hold["yyyymm"] > last_pre]],
        ignore_index=True,
    )
    with_return = crsp.loc[(crsp["yyyymm"] > last_pre) & crsp["ret_excess"].notna(), "yyyymm"]
    if with_return.empty:
        raise ValueError("no CRSP returns after the boundary")
    last_signal = min(int(chars["yyyymm"].max()), int(macro["yyyymm"].max()))
    through = min(int(next_month(pd.Series([last_signal]))[0]), int(with_return.max()))
    panel = build_panel_through(chars, crsp, macro, char_cols, ["sic2"], through)
    panel = panel[panel["yyyymm"] >= first_return]
    months = sorted(int(m) for m in panel["yyyymm"].unique())
    if not months or months != _months(first_return, through):
        raise ValueError(f"holdout months are not contiguous from {first_return} to {through}")
    macro_cols = [c for c in panel.columns if c.startswith("m_")]
    if panel[macro_cols].isna().any().any():
        raise ValueError("holdout rows have missing macro predictors")
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
