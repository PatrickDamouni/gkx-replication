"""Peak memory and wall time of the real data path, in a fresh process (council 2, replaces seam 6).

load pre-2017 OSAP (94 signal columns, matching D013's target width) -> rank_normalize ->
take the largest training window (return months <= 2003-12) as a float32 matrix.
Columns are the first 94 signals as a stand-in until the GKX feature map (D013) exists.

Usage: uv run python scripts/measure_memory.py   (writes results/memory.md)
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from gkx.panel import load_panel
from gkx.preprocess import rank_normalize

PATH = "data/raw/osap_firm_char_202510_pre2017.parquet"
WIDTH = 94


def peak_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6  # bytes on macOS, KB on Linux


def main() -> None:
    cols = [c for c in pq.read_schema(PATH).names if c not in ("permno", "yyyymm")][:WIDTH]
    steps = []
    t0 = time.time()
    df = load_panel(PATH, columns=["permno", "yyyymm", *cols])
    steps.append(("load", time.time() - t0, peak_gb(), len(df)))
    t0 = time.time()
    df = rank_normalize(df, cols)
    steps.append(("rank_normalize", time.time() - t0, peak_gb(), len(df)))
    t0 = time.time()
    X = df.loc[df["yyyymm"] <= 200312, cols].to_numpy(dtype=np.float32)
    steps.append(("largest train window -> float32 matrix", time.time() - t0, peak_gb(), len(X)))

    lines = [
        "# Memory and time of the real data path\n",
        f"{WIDTH} signal columns, fresh process, peak RSS is cumulative. Machine: 26 GB RAM.\n",
        "| step | seconds | peak RSS so far (GB) | rows |",
        "|---|---|---|---|",
        *(f"| {s} | {t:.0f} | {m:.2f} | {n:,} |" for s, t, m, n in steps),
    ]
    out = Path("results/memory.md")
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
