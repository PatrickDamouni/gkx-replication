"""The frozen model list and the single run (D027). One model family per process (LightGBM and
torch deadlock in one process), so the launcher calls this script three times.

Usage:
    uv run python scripts/run_test.py --family linear|trees|nn [--only A,B] [--rehearsal]

--only restricts the family to the named specs (D046: the raw and -dm NN arms run as two processes).

--rehearsal runs the identical specs on the 1975-1986 schedule (dry run). Without it, the folds are
GKX's 1987-2016 test years and evaluate() requires a clean git tree (D027). In test mode nothing
prints a metric: predictions are written and hashed, and every number is computed afterwards by
scripts/final_report.py (council 5: no per-model looks while later models still run).
"""

import argparse
import importlib
import time

import pyarrow.parquet as pq

from gkx.evaluate import evaluate
from gkx.panel import load_panel
from gkx.walk_forward import walk_forward

PANEL = "data/processed/panel_datashare.parquet"
NN_SEEDS = range(5)  # D006, D029


def columns(names: list[str] | None = None):
    names = pq.read_schema(PANEL).names if names is None else names
    macro = [c for c in names if c.startswith("m_")]
    meta = {"permno", "yyyymm", "signal_month", "target", "me_lag", "shrcd", "sic2", *macro}
    chars = [c for c in names if c not in meta]
    return chars, macro, [*chars, *macro, "sic2"]


def _model(module: str, cls: str, *args, **kwargs):
    """Import at fit time, so listing specs never loads lightgbm or torch (process isolation)."""
    return getattr(importlib.import_module(f"gkx.{module}"), cls)(*args, **kwargs)


def LinearGKX(*a, **k):
    return _model("linear", "LinearGKX", *a, **k)


def PCR(*a, **k):
    return _model("reduction", "PCR", *a, **k)


def PLS(*a, **k):
    return _model("reduction", "PLS", *a, **k)


def ENet(*a, **k):
    return _model("enet", "ENet", *a, **k)


def GLM(*a, **k):
    return _model("glm", "GLM", *a, **k)


def GBRT(*a, **k):
    return _model("trees", "GBRT", *a, **k)


def RF(*a, **k):
    return _model("trees", "RF", *a, **k)


def NN(*a, **k):
    return _model("nn", "NN", *a, **k)


def specs(family: str, names: list[str] | None = None):
    """(name, features, make_model, seeds, target, counts_as_trial) — frozen at the freeze tag."""
    chars, macro, full = columns(names)
    ols3 = ["mvel1", "bm", "mom12m"]
    if family == "linear":
        return [  # baselines are reported, not trials (D028/D034)
            ("OLS-3", ols3, lambda s: LinearGKX(ols3, [], None), (0,), "raw", False),
            (
                "OLS-3+H",
                ols3,
                lambda s: LinearGKX(ols3, [], None, loss="huber"),
                (0,),
                "raw",
                False,
            ),
            ("OLS", full, lambda s: LinearGKX(chars, macro, "sic2"), (0,), "raw", False),
            (
                "OLS+H",
                full,
                lambda s: LinearGKX(chars, macro, "sic2", loss="huber"),
                (0,),
                "raw",
                False,
            ),
            ("PCR", full, lambda s: PCR(chars, macro, "sic2"), (0,), "raw", True),
            ("PLS", full, lambda s: PLS(chars, macro, "sic2"), (0,), "raw", True),
            ("ENet+H", full, lambda s: ENet(chars, macro, "sic2"), (0,), "raw", True),
            ("GLM+H", full, lambda s: GLM(chars, macro, "sic2"), (0,), "raw", True),
        ]
    if family == "trees":
        return [  # D041 two arms, D042 inputs
            ("GBRT+H", full, lambda s: GBRT(chars, macro, "sic2"), (0,), "raw", True),
            ("GBRT+H-dm", full, lambda s: GBRT(chars, macro, "sic2"), (0,), "demeaned", True),
            ("RF", full, lambda s: RF(chars, macro, "sic2"), (0,), "raw", True),
            ("RF-dm", full, lambda s: RF(chars, macro, "sic2"), (0,), "demeaned", True),
        ]
    if family == "nn":
        out = []
        for arch, hidden in (("NN1", (32,)), ("NN2", (32, 16)), ("NN3", (32, 16, 8))):
            for suffix, target in (("", "raw"), ("-dm", "demeaned")):  # D043
                make = lambda s, h=hidden: NN(chars, macro, "sic2", hidden=h, seed=s)
                out.append((arch + suffix, full, make, NN_SEEDS, target, True))
        return out
    raise ValueError(family)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["linear", "trees", "nn"])
    ap.add_argument("--only", default="", help="comma-separated spec names within the family")
    ap.add_argument("--rehearsal", action="store_true")
    args = ap.parse_args()
    chosen = specs(args.family)
    if args.only:
        only = args.only.split(",")
        unknown = set(only) - {s[0] for s in chosen}
        if unknown:
            raise SystemExit(f"--only names not in family {args.family}: {sorted(unknown)}")
        chosen = [s for s in chosen if s[0] in only]

    test = not args.rehearsal
    folds = walk_forward() if test else walk_forward(first_test_year=1975, last_test_year=1986)
    panel = load_panel(PANEL) if test else load_panel(PANEL, end=198612)
    tag = "test" if test else "rehearsal"
    for name, features, make, seeds, target, trial in chosen:
        t0 = time.time()
        evaluate(
            make,
            panel,
            folds,
            features,
            seeds=seeds,
            name=f"{name}-{tag}",
            unlock_test=test,
            counts_as_trial=trial,
            target=target,
        )
        print(
            f"{time.strftime('%H:%M:%S')} done {name}-{tag} ({time.time() - t0:.0f}s)", flush=True
        )


if __name__ == "__main__":
    main()
