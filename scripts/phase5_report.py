"""Phase 5 report (D050 §8-10): one confirmatory test (H1), everything else deciding nothing.

  --dry-run    on the freeze-v1 1987-2016 predictions: must reproduce H1 t = 4.569 and test 1
               t = 2.213 with the frozen functions (D050 §6). Writes nothing.
  --check2016  the pre-tag `-check2016` runs must equal the freeze-v1 2016 predictions exactly.
  --holdout    the single look: `-holdout` runs from freeze-v2, canary fold 2016 checked first,
               then results/phase5_report.md (H1 and its pre-written reading first).

Usage: uv run --locked python scripts/phase5_report.py --dry-run|--check2016|--holdout
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.evaluate import LEDGER
from gkx.holdout import FROZEN_PANEL, PANEL_LOCK
from gkx.metrics import BLOCK, clark_west, diebold_mariano, paired_r2_ci, r2_oos
from gkx.panel import HOLDOUT_START, load_panel
from gkx.report import demeaned_frame
from gkx.walk_forward import walk_forward

REPO = Path(__file__).resolve().parents[1]
FREEZE_V1 = "46c1584"  # the 1987-2016 runs, loaded by commit (D050 §6)
KEYS = ["yyyymm", "permno"]
H1 = ("NN3-dm", "train_mean")
EXPECTED_DRY = {"H1": 4.569, "test 1": 2.213}
OUT = REPO / "results/phase5_report.md"


def frozen_report():
    """MODELS, CONFIRMATORY and _training_mean exactly as freeze-v1 defined them."""
    spec = importlib.util.spec_from_file_location("final_report", REPO / "scripts/final_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def runs(tag: str, models: list[str], sha: str | None = None, ledger: Path = LEDGER) -> dict:
    """One finished run per model with name `<model>-<tag>` (and commit `sha` if given)."""
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    started = {r["run_id"]: r for r in rows if r["event"] == "started"}
    found: dict[str, list] = {m: [] for m in models}
    for r in rows:
        st = started.get(r.get("run_id"))
        if r["event"] != "finished" or st is None:
            continue
        model = st["name"].removesuffix(f"-{tag}")
        same_commit = sha is None or st["git_sha"].startswith(sha) or sha.startswith(st["git_sha"])
        if st["name"].endswith(f"-{tag}") and model in found and same_commit:
            found[model].append((st, r))
    missing = [m for m, v in found.items() if not v]
    if missing:
        raise SystemExit(f"missing finished -{tag} runs: {missing}")
    many = [m for m, v in found.items() if len({st["git_sha"] for st, _ in v}) > 1]
    if many:
        raise SystemExit(f"-{tag} runs from more than one commit: {many}")
    return {m: v[-1] for m, v in found.items()}


def predictions(selected: dict, ledger: Path = LEDGER) -> dict[str, pd.DataFrame]:
    out = {}
    for m, (_, fin) in selected.items():
        path = ledger.parent / fin["preds_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != fin["preds_sha256"]:
            raise SystemExit(f"{m}: {path} does not match its logged sha256")
        out[m] = pd.read_parquet(path)
    return out


def canary(new: dict[str, pd.DataFrame], frozen: dict[str, pd.DataFrame]) -> list[str]:
    """Models whose 2016 predictions differ (any key or pred* value) from freeze-v1's."""
    bad = []
    for m, p in new.items():
        a = p[p["yyyymm"] // 100 == 2016].reset_index(drop=True)
        b = frozen[m][frozen[m]["yyyymm"] // 100 == 2016].reset_index(drop=True)
        cols = sorted(c for c in b.columns if c.startswith("pred"))
        same = (
            len(a) == len(b)
            and sorted(c for c in a.columns if c.startswith("pred")) == cols
            and np.array_equal(a[KEYS].to_numpy(), b[KEYS].to_numpy())
            and all(np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in cols)
        )
        if not same:
            bad.append(m)
    return bad


def frame(preds: dict[str, pd.DataFrame], models: list[str]) -> pd.DataFrame:
    base = preds[models[0]][[*KEYS, "target", "me_lag"]].copy()
    for m in models:
        p = preds[m]
        if len(p) != len(base) or not np.array_equal(p[KEYS].to_numpy(), base[KEYS].to_numpy()):
            raise SystemExit(f"{m}: prediction rows do not line up")
        base[m] = p["pred"].to_numpy()
    base["zero"] = 0.0
    return base


def reading(p: float, months: int) -> str:
    """D050 §8: the two sentences fixed before the look, chosen mechanically."""
    if p <= 0.05:
        return (
            "On GKX's characteristics, the demeaned-target NN3's within-month edge over a constant "
            f"survived the first post-publication years (Arm A, {months} months). This is not "
            "evidence that machine learning beats linear models."
        )
    return (
        f"Not detected (Arm A, {months} months). With ≈ 30% power at half the 1987–2016 effect, "
        "this is weak evidence of decay, not evidence of absence."
    )


def h1_and_test1(base: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        "H1": diebold_mariano(base, base=H1[1], candidate=H1[0]),
        "test 1": diebold_mariano(base, base="train_mean", candidate="NN3"),
    }


def analyse(base: pd.DataFrame, models: list[str], confirmatory: list) -> dict[str, pd.DataFrame]:
    """Every holdout number: H1, per-model R², the seven D045 tests, two secondary tests."""
    per_model = []
    for m in [*models, "train_mean"]:
        d, lo, hi = paired_r2_ci(base, "zero", m)
        per_model.append(
            {
                "model": m,
                "R2 %": 100 * r2_oos(base, m),
                "R2 95% CI": f"[{100 * lo:.3f}, {100 * hi:.3f}]",
            }
        )
    tests = []
    for a, b, kind in confirmatory:
        t, p = (
            clark_west(base, small=b, big=a)
            if kind == "CW"
            else diebold_mariano(base, base=b, candidate=a)
        )
        d, lo, hi = paired_r2_ci(base, b, a)
        tests.append({"candidate": a, "vs": b, "test": kind, "t": t, "p (one-sided)": p,
                      "ΔR2 %": 100 * d, "95% CI": f"[{100 * lo:.3f}, {100 * hi:.3f}]"})  # fmt: skip
    t, p = diebold_mariano(
        demeaned_frame(base, ["NN3-dm", "ENet+H"]), base="ENet+H", candidate="NN3-dm"
    )
    t2, p2 = diebold_mariano(base, base="train_mean", candidate="ENet+H")
    secondary = pd.DataFrame(
        [
            {"test": "NN3-dm vs ENet+H, within-month errors", "t": t, "p (one-sided)": p},
            {"test": "ENet+H vs constant", "t": t2, "p (one-sided)": p2},
        ]
    )
    return {
        "per_model": pd.DataFrame(per_model),
        "tests": pd.DataFrame(tests),
        "secondary": secondary,
    }


def dry_run(rep) -> int:
    selected = runs("test", rep.MODELS, FREEZE_V1)
    base = frame(predictions(selected), rep.MODELS)
    panel = load_panel(FROZEN_PANEL, columns=["yyyymm", "target"])
    base["train_mean"] = rep._training_mean(base, panel, walk_forward()).to_numpy()
    got = {k: round(v[0], 3) for k, v in h1_and_test1(base).items()}
    ok = got == EXPECTED_DRY
    print(f"dry run on 1987-2016: {got}, expected {EXPECTED_DRY}: {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def check2016(rep) -> int:
    head = _git("rev-parse", "--short", "HEAD")
    selected = runs("check2016", rep.MODELS, head)
    dirty = [m for m, (st, _) in selected.items() if st["git_dirty"]]
    if dirty:
        raise SystemExit(f"dirty -check2016 runs: {dirty}")
    frozen = predictions(runs("test", rep.MODELS, FREEZE_V1))
    bad = canary(predictions(selected), frozen)
    print(
        f"2016 fold at {head}: {18 - len(bad)}/18 bit-identical to freeze-v1"
        + (f"; differ: {bad}" if bad else "")
    )
    return 1 if bad else 0


def holdout(rep) -> int:
    tags = [t for t in _git("tag", "--list", "freeze-v2*").split() if t]
    if not tags:
        raise SystemExit("no freeze-v2 tag")
    tag = max(tags, key=lambda t: [int(x) for x in t.removeprefix("freeze-v").split(".")])
    sha = _git("rev-parse", f"{tag}^{{commit}}")
    selected = runs("holdout", rep.MODELS, sha)
    bad_runs = [m for m, (st, _) in selected.items() if st["git_dirty"] or not st["test_window"]]
    changed = _git(
        "diff", "--name-only", sha, "HEAD", "--", "src", "scripts", "pyproject.toml", "uv.lock"
    )
    if bad_runs or changed:
        raise SystemExit(f"refused: dirty runs {bad_runs} / code changed since {tag}: {changed!r}")
    preds = predictions(selected)
    bad = canary(preds, predictions(runs("test", rep.MODELS, FREEZE_V1)))
    if bad:
        raise SystemExit(f"canary fold 2016 differs from freeze-v1 for {bad}: no report (D050 §5)")
    preds = {m: p[p["yyyymm"] >= HOLDOUT_START].reset_index(drop=True) for m, p in preds.items()}
    base = frame(preds, rep.MODELS)
    last_year = int(base["yyyymm"].max()) // 100
    folds = walk_forward(first_test_year=2017, last_test_year=last_year, allow_holdout=True)
    panel = load_panel(FROZEN_PANEL, columns=["yyyymm", "target"])  # training windows end <= 2008
    base["train_mean"] = rep._training_mean(base, panel, folds).to_numpy()
    months = base["yyyymm"].nunique()
    t, p = h1_and_test1(base)["H1"]
    res = analyse(base, rep.MODELS, rep.CONFIRMATORY)
    before = pd.read_csv(REPO / "results/final_report.csv").set_index("model")["R2 %"]
    res["per_model"]["1987–2016 R2 %"] = res["per_model"]["model"].map(before)
    before_t = pd.read_csv(REPO / "results/final_report_tests.csv")
    res["tests"]["1987–2016 p"] = before_t[before_t["role"] == "confirmatory"][
        "p (one-sided)"
    ].to_numpy()
    md = lambda df: df.round(4).to_markdown(index=False)
    OUT.write_text(
        "# Phase 5 report — sealed holdout, Arm A (D050)\n\n"
        f"Holdout predictions from `{tag}` (`{sha[:7]}`); analysis at `{_git('rev-parse', '--short', 'HEAD')}`; "
        f"holdout panel {PANEL_LOCK.read_text().split()[0][:12]}…; test months "
        f"{base['yyyymm'].min()}–{base['yyyymm'].max()} ({months}). Canary fold 2016: 18/18 "
        "bit-identical to freeze-v1.\n\n"
        f"## H1 — the one confirmatory test\n\nNN3-dm vs the constant, one-sided Diebold–Mariano: "
        f"t = {t:.3f}, p = {p:.4f} ({'reject' if p <= 0.05 else 'not rejected'} at α = 0.05).\n\n"
        f"**{reading(p, months)}**\n\n"
        "## Reported, deciding nothing\n\n"
        f"Bootstrap CIs use {BLOCK}-month blocks over {months} months, so they are coarse. No D048 "
        "label and no decay ratio are reported (D050 §9).\n\n### R² by model\n\n"
        + md(res["per_model"])
        + "\n\n### The seven D045 tests (test 1 had ≈ 23% power at 60 months)\n\n"
        + md(res["tests"])
        + "\n\n### Secondary\n\n"
        + md(res["secondary"])
        + "\n"
    )
    print(f"wrote {OUT}\nH1: t = {t:.3f}, p = {p:.4f}\n{reading(p, months)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check2016", action="store_true")
    mode.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    rep = frozen_report()
    if args.dry_run:
        return dry_run(rep)
    return check2016(rep) if args.check2016 else holdout(rep)


if __name__ == "__main__":
    sys.exit(main())
