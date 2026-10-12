"""Phase 5 runs (D050): the frozen 18 specifications of scripts/run_test.py, unchanged.

  --check2016  pre-tag: the 2016 fold only, on the frozen panel, run names `-check2016`
               (counts_as_trial=False). Must equal the freeze-v1 2016 predictions exactly.
  --holdout    from tag freeze-v2: folds 2016 (canary) → the last holdout year, on the frozen
               panel followed by the locked holdout panel; run names `-holdout`.

Prints no metric. Every number comes from scripts/phase5_report.py after all runs finish.
Usage: uv run --locked python scripts/run_holdout.py --check2016|--holdout --family F [--only A,B]
"""

import argparse
import importlib.util
import time
from pathlib import Path

from gkx.evaluate import evaluate
from gkx.holdout import FROZEN_PANEL, load_merged
from gkx.panel import load_panel
from gkx.walk_forward import walk_forward

REPO = Path(__file__).resolve().parents[1]


def frozen_specs(family: str):
    """The spec list exactly as freeze-v1 defined it (scripts/run_test.py is byte-identical)."""
    spec = importlib.util.spec_from_file_location("run_test", REPO / "scripts/run_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.specs(family)


def plan(mode: str):
    """(panel, folds, tag) for a mode."""
    if mode == "check2016":
        return load_panel(FROZEN_PANEL), walk_forward(first_test_year=2016), "check2016"
    panel = load_merged()
    last_year = int(panel["yyyymm"].max()) // 100
    folds = walk_forward(first_test_year=2016, last_test_year=last_year, allow_holdout=True)
    return panel, folds, "holdout"


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check2016", action="store_true")
    mode.add_argument("--holdout", action="store_true")
    ap.add_argument("--family", required=True, choices=["linear", "trees", "nn"])
    ap.add_argument("--only", default="", help="comma-separated spec names within the family")
    args = ap.parse_args()

    chosen = frozen_specs(args.family)
    if args.only:
        only = args.only.split(",")
        unknown = set(only) - {s[0] for s in chosen}
        if unknown:
            raise SystemExit(f"--only names not in family {args.family}: {sorted(unknown)}")
        chosen = [s for s in chosen if s[0] in only]

    panel, folds, tag = plan("check2016" if args.check2016 else "holdout")
    for name, features, make, seeds, target, trial in chosen:
        t0 = time.time()
        evaluate(
            make,
            panel,
            folds,
            features,
            seeds=seeds,
            name=f"{name}-{tag}",
            unlock_test=True,
            counts_as_trial=trial and tag == "holdout",
            target=target,
        )
        print(
            f"{time.strftime('%H:%M:%S')} done {name}-{tag} ({time.time() - t0:.0f}s)", flush=True
        )


if __name__ == "__main__":
    main()
