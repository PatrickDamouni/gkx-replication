"""CRSP for Phase 5 (D050 §3): the frozen D019 queries with date bounds, through the frozen
`gkx.crsp.build_returns` (gkx.crsp itself is byte-identical to freeze-v1).

  --check  re-pull 2015-2016 and require it to equal the frozen crsp_returns_pre2017.parquet
           (proves the legacy tables still serve the same data). Pre-2017 only.
  --pull   pull 2016-12 onward once, compare its 2016-12 rows with the frozen file, require
           contiguous months from 201701, and write the rest atomically to
           data/sealed/crsp_returns_holdout.parquet (read-only). Prints counts, the month range and
           the sha256 only; never a return statistic. Refuses if the file exists.

Usage: uv run --locked python scripts/pull_crsp_holdout.py --check|--pull
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.crsp import QUERY_NAMES, build_returns
from gkx.holdout import CRSP_HOLDOUT, sha256
from gkx.panel import load_panel

FROZEN = Path("data/raw/crsp_returns_pre2017.parquet")
KEYS = ["permno", "yyyymm"]
COLS = ["ret", "ret_excess", "me", "shrcd", "exchcd"]
MSF = (
    "select permno, date, ret, prc, shrout from crsp.msf where date >= %(start)s and date < %(end)s"
)
DELIST = (
    "select permno, dlstdt, dlstcd, dlret from crsp.msedelist "
    "where dlstdt >= %(start)s and dlstdt < %(end)s"
)
RF = "select date, rf from ff.factors_monthly where date >= %(start)s and date < %(end)s"


def fetch(db, start: str, end: str) -> pd.DataFrame:
    """Returns for dates in [start, end): bound parameters, ISO dates only."""
    params = {"start": date.fromisoformat(start), "end": date.fromisoformat(end)}
    msf = db.raw_sql(MSF, date_cols=["date"], params=params)
    delist = db.raw_sql(DELIST, date_cols=["dlstdt"], params=params)
    names = db.raw_sql(QUERY_NAMES, date_cols=["namedt", "nameendt"])
    rf = db.raw_sql(RF, date_cols=["date"], params=params)
    return build_returns(msf, delist, names, rf)


def _connect():
    import wrds

    return wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME", "patrickdamouni"))


def compare(new: pd.DataFrame, frozen: pd.DataFrame, months: list[int]) -> list[str]:
    """Differences between a re-pull and the frozen file on `months` (empty list = identical)."""
    a = new[new["yyyymm"].isin(months)].sort_values(KEYS).reset_index(drop=True)
    b = frozen[frozen["yyyymm"].isin(months)].sort_values(KEYS).reset_index(drop=True)
    if len(a) != len(b) or not np.array_equal(a[KEYS].to_numpy(), b[KEYS].to_numpy()):
        return [f"row sets differ: {len(a)} new vs {len(b)} frozen"]
    problems = []
    for c in COLS:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        n = int(np.sum(~((x == y) | (np.isnan(x) & np.isnan(y)))))
        if n:
            problems.append(f"{c}: {n} rows differ")
    return problems


def holdout_months_ok(months: list[int]) -> bool:
    """Non-empty, starting at 201701, no missing month."""
    if not months or months[0] != 201701:
        return False
    year, month = 2017, 1
    for m in months:
        if m != year * 100 + month:
            return False
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return True


def check() -> int:
    db = _connect()
    new = fetch(db, "2015-01-01", "2017-01-01")
    db.close()
    months = sorted(int(m) for m in new["yyyymm"].unique() if 201501 <= m <= 201612)
    problems = compare(new, load_panel(FROZEN), months)
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
    out = new[new["yyyymm"] >= 201701].astype({"shrcd": "int8", "exchcd": "int8"})
    months = sorted(int(m) for m in out["yyyymm"].unique())
    if not holdout_months_ok(months):
        problems.append(
            "2017+ months missing, late or not contiguous (legacy tables: D050 §3 says stop)"
        )
    if problems:
        print("refusing: " + "; ".join(problems))
        return 1
    tmp = CRSP_HOLDOUT.with_suffix(".partial")
    CRSP_HOLDOUT.parent.chmod(0o755)
    try:
        out.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, CRSP_HOLDOUT)
        CRSP_HOLDOUT.chmod(0o444)
    finally:
        tmp.unlink(missing_ok=True)
        CRSP_HOLDOUT.parent.chmod(0o555)
    print(
        f"{len(out):,} rows, {out['permno'].nunique():,} permnos, months {months[0]}..{months[-1]}"
        f" -> {CRSP_HOLDOUT}\nsha256 {sha256(CRSP_HOLDOUT)}"
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
