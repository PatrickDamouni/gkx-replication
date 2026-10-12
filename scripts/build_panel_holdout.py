"""The Phase 5 holdout panel (D050 §7).

  --verify-mirror  pre-2017 only: run the exact unseal function, gkx.holdout.build_holdout_panel,
                   on the real pre-2017 files with 2016 playing the holdout (boundary 201601), and
                   require the result to equal the frozen panel's 2016 rows exactly. Prints only
                   counts and column names (never values: the data are licensed).
  --build          at unseal, once, from a clean freeze-v2 checkout: sha-check every input, build
                   return months 201701+, conform to the frozen schema, write
                   data/processed/panel_holdout.parquet and results/holdout_panel.sha256 (both
                   read-only), log the unseal in the ledger. Prints counts, months and the hash.

Usage: uv run --locked python scripts/build_panel_holdout.py --verify-mirror|--build
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime

import pandas as pd
import pyarrow.parquet as pq

from gkx.evaluate import LEDGER
from gkx.holdout import (
    CRSP_HOLDOUT,
    CRSP_HOLDOUT_SHA,
    FROZEN_PANEL,
    FROZEN_PANEL_SHA,
    HOLDOUT_PANEL,
    LOCK_TAG,
    PANEL_LOCK,
    SEALED,
    build_holdout_panel,
    conform,
    require_freeze_v2,
    sha256,
    verify,
)
from gkx.panel import load_panel

DS = "data/raw/datashare_pre2017.parquet"
CRSP = "data/raw/crsp_returns_pre2017.parquet"
MACRO = "data/raw/goyal_macro_pre2017.parquet"
CRSP_COLS = ["permno", "yyyymm", "ret_excess", "me", "shrcd"]


def _char_cols() -> list[str]:
    return [c for c in pq.read_schema(DS).names if c not in ("permno", "yyyymm", "sic2")]


def _mismatch(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Shape, column and per-column difference counts only — no values."""
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return [
            f"shape {a.shape} vs {b.shape}; columns equal: {list(a.columns) == list(b.columns)}"
        ]
    out = []
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype != y.dtype:
            out.append(f"{c}: dtype {x.dtype} vs {y.dtype}")
        elif not x.equals(y):
            out.append(f"{c}: {int((x != y).sum())} rows differ")
    return out


def verify_mirror() -> int:
    verify(FROZEN_PANEL, FROZEN_PANEL_SHA)
    boundary = 201601
    ds, crsp, macro = load_panel(DS), load_panel(CRSP, columns=CRSP_COLS), load_panel(MACRO)
    ds, crsp, macro = (d[d["yyyymm"] <= 201611] for d in (ds, crsp, macro))  # signal months
    crsp_all = load_panel(CRSP, columns=CRSP_COLS)  # returns through 201612
    rebuilt = build_holdout_panel(
        ds[ds["yyyymm"] < boundary],
        ds[ds["yyyymm"] >= boundary],
        crsp[crsp["yyyymm"] < boundary],
        crsp_all[crsp_all["yyyymm"] >= boundary],
        macro[macro["yyyymm"] < boundary],
        macro[macro["yyyymm"] >= boundary],
        _char_cols(),
        first_return=boundary,
    )
    rebuilt = conform(rebuilt)
    frozen = load_panel(FROZEN_PANEL)
    frozen = frozen[frozen["yyyymm"].between(boundary, 201612)].reset_index(drop=True)
    problems = _mismatch(rebuilt, frozen)
    if problems:
        print("MIRROR MISMATCH on 2016: " + "; ".join(problems[:10]))
        return 1
    print(
        "the unseal builder, run with 2016 as the holdout, reproduces the frozen panel's 2016 rows "
        f"exactly ({len(frozen):,} rows, {frozen['yyyymm'].nunique()} months)"
    )
    return 0


def build() -> int:
    tag = require_freeze_v2()
    if PANEL_LOCK.exists() or HOLDOUT_PANEL.exists():
        print(f"refusing: {HOLDOUT_PANEL} or {PANEL_LOCK} exists (built once, D050 §7)")
        return 1
    verify(FROZEN_PANEL, FROZEN_PANEL_SHA)
    for path, digest in SEALED.values():
        verify(path, digest)
    verify(CRSP_HOLDOUT, CRSP_HOLDOUT_SHA)
    panel = conform(
        build_holdout_panel(
            load_panel(DS),
            pd.read_parquet(SEALED["datashare"][0]),
            load_panel(CRSP, columns=CRSP_COLS),
            pd.read_parquet(CRSP_HOLDOUT, columns=CRSP_COLS),
            load_panel(MACRO),
            pd.read_parquet(SEALED["macro"][0]),
            _char_cols(),
        )
    )
    tmp = HOLDOUT_PANEL.with_suffix(".partial")
    HOLDOUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(tmp, compression="zstd", index=False)
    if pq.read_schema(tmp).remove_metadata() != pq.read_schema(FROZEN_PANEL).remove_metadata():
        tmp.unlink()
        print("refusing: holdout panel schema differs from the frozen panel")
        return 1
    os.replace(tmp, HOLDOUT_PANEL)
    HOLDOUT_PANEL.chmod(0o444)
    digest = sha256(HOLDOUT_PANEL)
    PANEL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    PANEL_LOCK.write_text(f"{digest}  {HOLDOUT_PANEL}\n")
    PANEL_LOCK.chmod(0o444)
    with LEDGER.open("a") as f:
        f.write(
            json.dumps(
                {
                    "event": "phase5_unseal",
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "tag": tag,
                    "holdout_panel_sha256": digest,
                    "rows": len(panel),
                    "months": [int(panel["yyyymm"].min()), int(panel["yyyymm"].max())],
                }
            )
            + "\n"
        )
    print(
        f"{len(panel):,} rows, {panel['permno'].nunique():,} permnos, return months "
        f"{panel['yyyymm'].min()}..{panel['yyyymm'].max()} -> {HOLDOUT_PANEL}\nsha256 {digest}\n"
        f"next: git tag -a {LOCK_TAG} -m {digest} && git push origin {LOCK_TAG}  (before any fit)"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-mirror", action="store_true")
    group.add_argument("--build", action="store_true")
    return verify_mirror() if ap.parse_args().verify_mirror else build()


if __name__ == "__main__":
    sys.exit(main())
