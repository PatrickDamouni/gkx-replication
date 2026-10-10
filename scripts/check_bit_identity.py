"""Compare the latest rehearsal run of each model (frozen runner) with its previous finished run."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
rows = [json.loads(l) for l in (R / "trials.jsonl").read_text().splitlines() if l.strip()]
started = {r["run_id"]: r for r in rows if r["event"] == "started"}
fin = [
    (started[r["run_id"]], r) for r in rows if r["event"] == "finished" and r["run_id"] in started
]
by = {}
for st, f in fin:
    if st["name"].endswith("-rehearsal"):
        by.setdefault(st["name"], []).append((st, f))
want = sys.argv[1]
for name, runs in by.items():
    new = [x for x in runs if x[0]["git_sha"].startswith(want)]
    old = [x for x in runs if not x[0]["git_sha"].startswith(want)]
    if not new:
        continue
    if not old:
        print(f"{name:22s} no earlier run")
        continue
    a = pd.read_parquet(R / new[-1][1]["preds_path"])
    b = pd.read_parquet(R / old[-1][1]["preds_path"])
    cols = [c for c in a.columns if c.startswith("pred")]
    same = (
        list(a.columns) == list(b.columns)
        and len(a) == len(b)
        and all(np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in cols)
    )
    diff = (
        max(float(np.nanmax(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in cols)
        if len(a) == len(b)
        else float("nan")
    )
    print(
        f"{name:22s} {'IDENTICAL' if same else 'DIFFERS'}  max|Δ|={diff:.3g}  vs {old[-1][0]['git_sha'][:7]}"
    )
