"""Evidence for D024 rule 1: CRSP-rebuilt characteristics vs GKX datashare (pre-2017 only).

Accept a rebuild only if Spearman ρ ≥ 0.99 and ≥ 95% of values equal within 1e-4.

Usage: uv run python scripts/validate_crsp_chars.py   (writes results/crsp_chars_validation.md)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gkx.chars_crsp import crsp_characteristics
from gkx.panel import load_panel

NAMES = ["mvel1", "mom1m", "mom6m", "mom12m", "mom36m", "chmom"]
RHO_MIN, EXACT_MIN = 0.99, 0.95


def main() -> None:
    crsp = load_panel(
        "data/raw/crsp_returns_pre2017.parquet", columns=["permno", "yyyymm", "ret", "me"]
    )
    mine = crsp_characteristics(crsp)
    ds = load_panel("data/raw/datashare_pre2017.parquet", columns=["permno", "yyyymm", *NAMES])
    both = ds.merge(mine, on=["permno", "yyyymm"], suffixes=("_ds", "_crsp"))
    rows = []
    for n in NAMES:
        pair = both[[f"{n}_ds", f"{n}_crsp"]].dropna()
        a, b = pair[f"{n}_ds"], pair[f"{n}_crsp"]
        rel = np.isclose(a, b, rtol=1e-4, atol=1e-4).mean()
        rho = a.rank().corr(b.rank())
        ok = rho >= RHO_MIN and rel >= EXACT_MIN
        coverage = len(pair) / ds[n].notna().sum()
        rows.append(
            {
                "characteristic": n,
                "spearman": round(rho, 4),
                "equal": f"{rel:.1%}",
                "coverage of datashare": f"{coverage:.1%}",
                "rows": f"{len(pair):,}",
                "accepted": "yes" if ok else "NO",
            }
        )
    table = pd.DataFrame(rows)
    Path("results/crsp_chars_validation.md").write_text(
        "# CRSP-rebuilt characteristics vs GKX datashare (pre-2017)\n\n"
        f"Rule (D024): Spearman ≥ {RHO_MIN} and ≥ {EXACT_MIN:.0%} equal (rtol/atol 1e-4). "
        "Same (permno, signal month).\n\n" + table.to_markdown(index=False) + "\n"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
