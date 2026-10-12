"""The Phase 5 holdout panel (D050 §7).

  --verify-mirror  pre-2017 only: rebuild return months 2016-01..2016-12 with
                   gkx.holdout.build_panel_through from the pre-2017 files and require them to equal
                   the frozen panel's 2016 rows exactly (real-data test of the mirror builder).
  --build          at unseal, once: sha-check every sealed input, build return months 201701+,
                   conform to the frozen schema, write data/processed/panel_holdout.parquet and its
                   sha256 to results/holdout_panel.sha256. Prints counts and months only.

Usage: uv run --locked python scripts/build_panel_holdout.py --verify-mirror|--build
"""

import argparse
import sys

import pandas as pd
import pyarrow.parquet as pq

from gkx.holdout import (
    CRSP_HOLDOUT,
    CRSP_HOLDOUT_SHA,
    FROZEN_PANEL,
    FROZEN_PANEL_SHA,
    HOLDOUT_PANEL,
    PANEL_LOCK,
    SEALED,
    build_holdout_panel,
    build_panel_through,
    conform,
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


def verify_mirror() -> int:
    verify(FROZEN_PANEL, FROZEN_PANEL_SHA)
    chars = _char_cols()
    sig = (201512, 201611)  # signal months of return months 2016-01..2016-12
    ds = load_panel(DS)
    ds = ds[ds["yyyymm"].between(*sig)]
    crsp = load_panel(CRSP, columns=CRSP_COLS)
    crsp = crsp[crsp["yyyymm"].between(sig[0], 201612)]
    macro = load_panel(MACRO)
    rebuilt = build_panel_through(ds, crsp, macro, chars, ["sic2"], through=201612)
    rebuilt = rebuilt[rebuilt["yyyymm"].between(201601, 201612)]
    rebuilt = conform(rebuilt.sort_values(["yyyymm", "permno"]).reset_index(drop=True))
    frozen = load_panel(FROZEN_PANEL)
    frozen = frozen[frozen["yyyymm"].between(201601, 201612)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(rebuilt, frozen, check_exact=True)
    except AssertionError as e:
        print(f"MIRROR MISMATCH on 2016:\n{e}")
        return 1
    print(f"mirror builder reproduces the frozen panel's 2016 rows exactly ({len(frozen):,} rows)")
    return 0


def build() -> int:
    if PANEL_LOCK.exists() or HOLDOUT_PANEL.exists():
        print(f"refusing: {HOLDOUT_PANEL} or {PANEL_LOCK} exists (built once, D050 §7)")
        return 1
    verify(FROZEN_PANEL, FROZEN_PANEL_SHA)
    for path, digest in SEALED.values():
        verify(path, digest)
    verify(CRSP_HOLDOUT, CRSP_HOLDOUT_SHA)
    panel = build_holdout_panel(
        load_panel(DS),
        pd.read_parquet(SEALED["datashare"][0]),
        load_panel(CRSP, columns=CRSP_COLS),
        pd.read_parquet(CRSP_HOLDOUT, columns=CRSP_COLS),
        load_panel(MACRO),
        pd.read_parquet(SEALED["macro"][0]),
        _char_cols(),
    )
    panel = conform(panel)
    HOLDOUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(HOLDOUT_PANEL, compression="zstd", index=False)
    if (
        pq.read_schema(HOLDOUT_PANEL).remove_metadata()
        != pq.read_schema(FROZEN_PANEL).remove_metadata()
    ):
        HOLDOUT_PANEL.unlink()
        print("refusing: holdout panel schema differs from the frozen panel")
        return 1
    PANEL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    PANEL_LOCK.write_text(f"{sha256(HOLDOUT_PANEL)}  {HOLDOUT_PANEL}\n")
    print(
        f"{len(panel):,} rows, {panel['permno'].nunique():,} permnos, return months "
        f"{panel['yyyymm'].min()}..{panel['yyyymm'].max()} -> {HOLDOUT_PANEL}\n"
        f"locked: {PANEL_LOCK.read_text().strip()}"
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
