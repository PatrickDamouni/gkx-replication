"""GKX's eight macro predictors from the Welch-Goyal data (D021).

Values at month t are treated as known at the end of t, so they join on the
signal month, like the characteristics.

Usage: uv run python -m gkx.macro   (ingests data/raw/goyal_download/predictors_2025.xlsx)
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.panel import split_holdout

MACRO_COLS = ("dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar")
XLSX = Path("data/raw/goyal_download/predictors_2025.xlsx")
FULL = Path("data/raw/goyal_macro_full.parquet")  # transient: deleted after split
PRE = Path("data/raw/goyal_macro_pre2017.parquet")
HOLDOUT = Path("data/sealed/goyal_macro_holdout.parquet")


def build_macro(monthly: pd.DataFrame) -> pd.DataFrame:
    """Welch & Goyal (2008) definitions, as used by GKX (2020) §2."""
    return pd.DataFrame(
        {
            "yyyymm": monthly["yyyymm"].astype("int64"),
            "dp": np.log(monthly["D12"]) - np.log(monthly["Index"]),
            "ep": np.log(monthly["E12"]) - np.log(monthly["Index"]),
            "bm": monthly["b/m"],
            "ntis": monthly["ntis"],
            "tbl": monthly["tbl"],
            "tms": monthly["lty"] - monthly["tbl"],
            "dfy": monthly["BAA"] - monthly["AAA"],
            "svar": monthly["svar"],
        }
    )


def main() -> int:
    xlsx_sha = hashlib.sha256(XLSX.read_bytes()).hexdigest()
    build_macro(pd.read_excel(XLSX, sheet_name="Monthly")).to_parquet(FULL)
    HOLDOUT.parent.chmod(0o755)
    digest = split_holdout(FULL, PRE, HOLDOUT)
    HOLDOUT.chmod(0o444)
    HOLDOUT.parent.chmod(0o555)
    FULL.unlink()
    XLSX.unlink()  # an unsealed copy (D009)
    XLSX.parent.rmdir()
    print(f"source xlsx sha256 {xlsx_sha}\npre-2017 -> {PRE}\nholdout sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
