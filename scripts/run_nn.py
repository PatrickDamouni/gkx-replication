"""NN1–NN3 rehearsal runner (D029, D040, D043). Imports torch only — never LightGBM (deadlock).

Usage: uv run python scripts/run_nn.py MODEL [MODEL ...]   e.g. NN1 NN1-dm NN3
Default schedule: the 1975-1986 REHEARSAL (D027). Five seeds per model (D006, D029).
"""

import json
import sys
import time

import pyarrow.parquet as pq

from gkx.evaluate import LEDGER, evaluate
from gkx.metrics import r2_oos
from gkx.nn import NN
from gkx.panel import load_panel
from gkx.walk_forward import walk_forward

PANEL = "data/processed/panel_datashare.parquet"
SEEDS = range(5)
HIDDEN = {"NN1": (32,), "NN2": (32, 16), "NN3": (32, 16, 8)}


def main() -> None:
    folds = walk_forward(first_test_year=1975, last_test_year=1986)
    names = pq.read_schema(PANEL).names
    macro = [c for c in names if c.startswith("m_")]
    meta = {"permno", "yyyymm", "signal_month", "target", "me_lag", "shrcd", "sic2", *macro}
    chars = [c for c in names if c not in meta]
    full = [*chars, *macro, "sic2"]
    panel = load_panel(PANEL, end=198612)

    for name in sys.argv[1:]:
        arch, _, arm = name.partition("-")
        target = "demeaned" if arm == "dm" else "raw"  # D043 arms
        hidden = HIDDEN[arch]
        t0 = time.time()
        preds = evaluate(
            lambda seed, h=hidden: NN(chars, macro, "sic2", hidden=h, seed=seed),
            panel,
            folds,
            full,
            seeds=SEEDS,
            name=f"{name}-rehearsal",
            counts_as_trial=False,
            target=target,
        )
        rows = [json.loads(line) for line in LEDGER.read_text().splitlines()]
        chosen = [s["chosen"] for s in rows[-1].get("selection", [])]
        n_const = sum(c == "constant" for c in chosen)
        print(
            f"{name:7s} rehearsal: R²_oos {100 * r2_oos(preds, 'pred'):+.3f}%  "
            f"({time.time() - t0:.0f}s)  constant chosen in {n_const}/{len(chosen)} seed-folds",
            flush=True,
        )


if __name__ == "__main__":
    main()
