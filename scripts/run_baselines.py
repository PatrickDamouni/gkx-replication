"""Pre-registered OLS baselines (D011, D030) on Arm A. Default: the 1975-1986 REHEARSAL schedule (D027).

The 1987-2016 test window is only reachable with --test, which evaluate() further refuses
unless the git tree is clean (D027). Baselines never count as deflated-Sharpe trials (D028).

Usage: uv run python scripts/run_baselines.py [--test]
"""

import sys
import time

import pyarrow.parquet as pq

from gkx.evaluate import evaluate
from gkx.linear import LinearGKX
from gkx.metrics import r2_oos
from gkx.panel import load_panel
from gkx.walk_forward import walk_forward

PANEL = "data/processed/panel_datashare.parquet"
OLS3 = ["mvel1", "bm", "mom12m"]


def main() -> None:
    test = "--test" in sys.argv
    folds = walk_forward() if test else walk_forward(first_test_year=1975, last_test_year=1986)
    names = pq.read_schema(PANEL).names
    macro = [c for c in names if c.startswith("m_")]
    meta = {"permno", "yyyymm", "signal_month", "target", "me_lag", "shrcd", "sic2", *macro}
    chars = [c for c in names if c not in meta]
    assert len(chars) == 94 and len(macro) == 8, (len(chars), len(macro))
    panel = load_panel(PANEL) if test else load_panel(PANEL, end=198612)  # data-layer lock

    specs = {
        "OLS-3": (OLS3, lambda s: LinearGKX(OLS3, [], None, intercept=True)),
        "OLS-3+H": (OLS3, lambda s: LinearGKX(OLS3, [], None, intercept=True, loss="huber")),
        "OLS": ([*chars, *macro, "sic2"], lambda s: LinearGKX(chars, macro, "sic2")),
        "OLS+H": (
            [*chars, *macro, "sic2"],
            lambda s: LinearGKX(chars, macro, "sic2", loss="huber"),
        ),
    }
    tag = "test" if test else "rehearsal"
    for name, (features, make) in specs.items():
        t0 = time.time()
        preds = evaluate(make, panel, folds, features, name=f"{name}-{tag}", unlock_test=test)
        print(
            f"{name:8s} {tag}: R²_oos {100 * r2_oos(preds, 'pred'):+.3f}%  "
            f"({len(preds):,} predictions, {time.time() - t0:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
