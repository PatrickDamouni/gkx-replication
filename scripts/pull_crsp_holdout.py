"""CRSP for Phase 5 (D050 §3), through `gkx.crsp.fetch` (same queries and build_returns as D019).

  --check  re-pull 2015-2016 and require it to equal the frozen crsp_returns_pre2017.parquet
           (proves the legacy tables still serve the same data). Pre-2017 only.
  --pull   pull 2016-12 onward once, compare its 2016-12 rows with the frozen file, and write the
           rest to data/sealed/crsp_returns_holdout.parquet (read-only). Prints counts, the month
           range and the sha256 only; never a return statistic. Refuses if the file exists.

Usage: uv run --locked python scripts/pull_crsp_holdout.py --check|--pull
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.crsp import fetch
from gkx.holdout import CRSP_HOLDOUT, sha256
from gkx.panel import load_panel

FROZEN = Path("data/raw/crsp_returns_pre2017.parquet")
KEYS = ["permno", "yyyymm"]
COLS = ["ret", "ret_excess", "me", "shrcd", "exchcd"]


def _connect():
    import wrds

    return wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME", "patrickdamouni"))


def compare(new: pd.DataFrame, frozen: pd.DataFrame, months: list[int]) -> list[str]:
    """Differences between a re-pull and the frozen file on `months` (empty list = identical)."""
    a = new[new["yyyymm"].isin(months)].sort_values(KEYS).reset_index(drop=True)
    b = frozen[frozen["yyyymm"].isin(months)].sort_values(KEYS).reset_index(drop=True)
    if not a[KEYS].equals(b[KEYS].astype(a[KEYS].dtypes.to_dict())):
        return [f"row sets differ: {len(a)} new vs {len(b)} frozen"]
    problems = []
    for c in COLS:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        if not np.array_equal(x, y, equal_nan=True):
            problems.append(
                f"{c}: {int((~np.isclose(x, y, equal_nan=True, rtol=0, atol=0)).sum())} rows differ"
            )
    return problems


def check() -> int:
    db = _connect()
    new = fetch(db, "2015-01-01", "2017-01-01")
    db.close()
    frozen = load_panel(FROZEN)
    months = sorted(m for m in new["yyyymm"].unique() if 201501 <= m <= 201612)
    problems = compare(new, frozen, months)
    print(
        f"re-pulled {len(new):,} rows, {months[0]}..{months[-1]}: "
        + ("; ".join(problems) or "identical")
    )
    return 1 if problems else 0


def pull() -> int:
    if CRSP_HOLDOUT.exists():
        print(f"refusing: {CRSP_HOLDOUT} exists (pulled once, D050 §3)")
        return 1
    db = _connect()
    new = fetch(db, "2016-12-01", "2100-01-01")
    db.close()
    problems = compare(new, load_panel(FROZEN), [201612])
    if problems:
        print("refusing: 2016-12 rows differ from the frozen file: " + "; ".join(problems))
        return 1
    out = new[new["yyyymm"] >= 201701].astype({"shrcd": "int8", "exchcd": "int8"})
    CRSP_HOLDOUT.parent.chmod(0o755)
    out.to_parquet(CRSP_HOLDOUT, compression="zstd", index=False)
    CRSP_HOLDOUT.chmod(0o444)
    CRSP_HOLDOUT.parent.chmod(0o555)
    print(
        f"{len(out):,} rows, {out['permno'].nunique():,} permnos, months "
        f"{out['yyyymm'].min()}..{out['yyyymm'].max()} -> {CRSP_HOLDOUT}\nsha256 {sha256(CRSP_HOLDOUT)}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--pull", action="store_true")
    args = ap.parse_args()
    return check() if args.check else pull()


if __name__ == "__main__":
    sys.exit(main())
