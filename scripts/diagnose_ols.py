"""Rehearsal-only diagnostics for the linear baselines (D027: never the test window).

1. Fast Huber IRLS: wall time, and agreement with the earlier slow-path OLS+H predictions.
2. Where OLS-all's rehearsal explosion comes from: full design vs characteristics + industry
   dummies without macro interactions.

Usage: uv run python scripts/diagnose_ols.py
"""

import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gkx.evaluate import evaluate
from gkx.linear import LinearGKX
from gkx.metrics import r2_oos
from gkx.panel import load_panel
from gkx.walk_forward import walk_forward

PANEL = "data/processed/panel_datashare.parquet"


def _previous_preds(name: str) -> pd.DataFrame:
    with open("results/trials.jsonl") as f:
        rows = [json.loads(line) for line in f]
    started = {r["run_id"]: r for r in rows if r["event"] == "started"}
    done = [r for r in rows if r["event"] == "finished" and started[r["run_id"]]["name"] == name]
    return pd.read_parquet("results/" + done[-1]["preds_path"])


def main() -> None:
    folds = walk_forward(first_test_year=1975, last_test_year=1986)
    names = pq.read_schema(PANEL).names
    macro = [c for c in names if c.startswith("m_")]
    meta = {"permno", "yyyymm", "signal_month", "target", "me_lag", "shrcd", "sic2", *macro}
    chars = [c for c in names if c not in meta]
    panel = load_panel(PANEL, end=198612)
    full = [*chars, *macro, "sic2"]

    t0 = time.time()
    fast = evaluate(
        lambda s: LinearGKX(chars, macro, "sic2", loss="huber"),
        panel,
        folds,
        full,
        name="OLS+H-rehearsal-fast",
    )
    slow = _previous_preds("OLS+H-rehearsal")
    j = fast.merge(slow, on=["permno", "yyyymm"], suffixes=("_fast", "_slow"))
    rel = np.abs(j.pred_fast - j.pred_slow).max() / np.abs(j.pred_slow).max()
    print(
        f"OLS+H fast: {time.time() - t0:.0f}s (unconverged slow path was 2378s); "
        f"R² {100 * r2_oos(fast, 'pred'):+.3f}%; max |Δpred| / max |pred| vs slow = {rel:.1e}"
    )

    nomacro = evaluate(
        lambda s: LinearGKX(chars, [], "sic2"),
        panel,
        folds,
        [*chars, "sic2"],
        name="OLS-nomacro-rehearsal",
    )
    withmacro = _previous_preds("OLS-rehearsal")
    for label, p in (
        ("OLS full (x ⊗ c + dummies)", withmacro),
        ("OLS chars + dummies only", nomacro),
    ):
        by_year = (
            p.assign(y=p.yyyymm // 100)
            .groupby("y")
            .apply(
                lambda d: 100 * (1 - ((d.target - d.pred) ** 2).sum() / (d.target**2).sum()),
                include_groups=False,
            )
        )
        print(
            f"{label:28s} R² {100 * r2_oos(p, 'pred'):+10.3f}%  | by year: "
            + " ".join(f"{v:+.1f}" for v in by_year)
        )


if __name__ == "__main__":
    main()
