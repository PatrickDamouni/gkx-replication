"""Build the Arm A modeling panel (D025): datashare characteristics + CRSP target + macro.

Return-month keyed (D014), all NYSE/AMEX/NASDAQ securities (D020), characteristics ranked
within the signal-month universe (build_panel), sic2 carried for industry dummies.
Writes data/processed/panel_datashare.parquet sorted by (yyyymm, permno). Reads pre-2017
files only, so no holdout row can enter.

Usage: uv run python scripts/build_panel_datashare.py
"""

import resource
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from gkx.build import build_panel
from gkx.panel import load_panel

DS = "data/raw/datashare_pre2017.parquet"
OUT = Path("data/processed/panel_datashare.parquet")


def main() -> None:
    t0 = time.time()
    chars = [c for c in pq.read_schema(DS).names if c not in ("permno", "yyyymm", "sic2")]
    ds = load_panel(DS)
    crsp = load_panel(
        "data/raw/crsp_returns_pre2017.parquet",
        columns=["permno", "yyyymm", "ret_excess", "me", "shrcd"],
    )
    macro = load_panel("data/raw/goyal_macro_pre2017.parquet")
    panel = build_panel(ds, crsp, macro, char_cols=chars, keep=["sic2"])
    panel = panel[panel["yyyymm"] >= 195703].sort_values(["yyyymm", "permno"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, compression="zstd", index=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = rss / 1e9 if sys.platform == "darwin" else rss / 1e6
    print(
        f"{len(panel):,} rows, {panel['permno'].nunique():,} permnos, return months "
        f"{panel['yyyymm'].min()}..{panel['yyyymm'].max()}, target NaN "
        f"{panel['target'].isna().mean():.2%}, peak RSS {peak:.1f} GB, "
        f"{time.time() - t0:.0f}s -> {OUT} ({OUT.stat().st_size / 1e9:.2f} GB)"
    )


if __name__ == "__main__":
    main()
