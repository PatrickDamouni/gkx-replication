"""Phase 3 models on Arm A (D029, D037). Default: the 1975-1986 REHEARSAL schedule (D027).

Usage: uv run python scripts/run_models.py MODEL [MODEL ...]      e.g. PCR PLS
The test window is not reachable from this script; the single frozen run gets its own script.
"""

import json
import sys
import time

import pyarrow.parquet as pq

from gkx.enet import ENet
from gkx.evaluate import LEDGER, evaluate
from gkx.glm import GLM
from gkx.metrics import r2_oos
from gkx.panel import load_panel
from gkx.reduction import PCR, PLS
from gkx.trees import GBRT, RF
from gkx.walk_forward import walk_forward

PANEL = "data/processed/panel_datashare.parquet"


def main() -> None:
    folds = walk_forward(first_test_year=1975, last_test_year=1986)
    names = pq.read_schema(PANEL).names
    macro = [c for c in names if c.startswith("m_")]
    meta = {"permno", "yyyymm", "signal_month", "target", "me_lag", "shrcd", "sic2", *macro}
    chars = [c for c in names if c not in meta]
    full = [*chars, *macro, "sic2"]
    panel = load_panel(PANEL, end=198612)

    specs = {
        "PCR": (full, lambda s: PCR(chars, macro, "sic2")),
        "PLS": (full, lambda s: PLS(chars, macro, "sic2")),
        "ENet+H": (full, lambda s: ENet(chars, macro, "sic2", loss="huber")),
        "GLM+H": (full, lambda s: GLM(chars, macro, "sic2", loss="huber")),
        "GBRT+H": (full, lambda s: GBRT(chars, macro, "sic2")),
        "RF": (full, lambda s: RF(chars, macro, "sic2")),
        "GBRT+H-dm": (full, lambda s: GBRT(chars, macro, "sic2"), "demeaned"),
        "RF-dm": (full, lambda s: RF(chars, macro, "sic2"), "demeaned"),
    }
    for name in sys.argv[1:]:
        features, make, *arm = specs[name]
        target = arm[0] if arm else "raw"  # D041/D043 arms
        t0 = time.time()
        preds = evaluate(
            make,
            panel,
            folds,
            features,
            name=f"{name}-rehearsal",
            counts_as_trial=False,
            target=target,
        )
        rows = [json.loads(line) for line in LEDGER.read_text().splitlines()]
        chosen = [s["chosen"] for s in rows[-1].get("selection", [])]
        print(
            f"{name:6s} rehearsal: R²_oos {100 * r2_oos(preds, 'pred'):+.3f}%  "
            f"({time.time() - t0:.0f}s)  chosen by fold: {chosen}",
            flush=True,
        )


if __name__ == "__main__":
    main()
