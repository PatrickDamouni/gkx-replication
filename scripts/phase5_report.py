"""Phase 5 report (D050 §8-10): one confirmatory test (H1), everything else deciding nothing.

  --dry-run    rehearses the full report path on the freeze-v1 1987-2016 predictions (playing the
               holdout) and must reproduce H1 t = 4.569 and test 1 t = 2.213 with the frozen
               functions (D050 §6). Writes results/phase5_report_dryrun.md.
  --check2016  the pre-tag `-check2016` runs must equal the freeze-v1 2016 predictions exactly.
  --holdout    the single look, from a clean freeze-v2(.1) checkout: every input is checked first
               (tag, clean tree, published panel lock, one panel hash across all fits, -check2016
               evidence, canary fold 2016, no NaN); only then are holdout numbers computed, written
               once to results/phase5_report.md (refuses if it exists) and the look logged.

Usage: uv run --locked python scripts/phase5_report.py --dry-run|--check2016|--holdout
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.evaluate import LEDGER
from gkx.holdout import (
    ALLOWED_TAGS,
    FROZEN_PANEL,
    HOLDOUT_PANEL,
    require_freeze_v2,
    require_published_lock,
    sha256,
)
from gkx.metrics import BLOCK, clark_west, diebold_mariano, paired_r2_ci, r2_oos
from gkx.panel import HOLDOUT_START, load_panel
from gkx.report import demeaned_frame
from gkx.walk_forward import walk_forward

REPO = Path(__file__).resolve().parents[1]
# The 1987-2016 runs, loaded by commit (D050 §6). This is the commit's short sha as the ledger
# records it; `git rev-parse --short freeze-v1` prints the annotated tag object instead.
FREEZE_V1 = "46c1584"
KEYS = ["yyyymm", "permno"]
H1 = ("NN3-dm", "train_mean")
EXPECTED_DRY = {"H1": 4.569, "test 1": 2.213}
OUT = REPO / "results/phase5_report.md"
DRY_OUT = REPO / "results/phase5_report_dryrun.md"


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


def _rows(ledger: Path) -> list[dict]:
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def runs(tag: str, models: list[str], sha: str | None = None, ledger: Path = LEDGER) -> dict:
    """One finished run per model with name `<model>-<tag>` (and commit `sha` if given)."""
    rows = _rows(ledger)
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
    if base[["target", *models]].isna().any().any():
        raise SystemExit("NaN in targets or predictions: no report")
    base["zero"] = 0.0
    return base


def with_training_mean(base: pd.DataFrame, rep, folds) -> pd.DataFrame:
    if max(f.train[1] for f in folds) >= HOLDOUT_START:
        raise SystemExit("a training window reaches the holdout")
    panel = load_panel(FROZEN_PANEL, columns=["yyyymm", "target"])
    base["train_mean"] = rep._training_mean(base, panel, folds).to_numpy()
    if base["train_mean"].isna().any():
        raise SystemExit("training-mean benchmark missing for some rows")
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
    """Every number of the report: per-model R², the seven D045 tests, H1 and two secondary tests."""
    per_model = []
    for m in [*models, "train_mean"]:
        _, lo, hi = paired_r2_ci(base, "zero", m)
        per_model.append(
            {
                "model": m,
                "R2 %": 100 * r2_oos(base, m),
                "R2 95% CI": f"[{100 * lo:.3f}, {100 * hi:.3f}]",
            }
        )
    tests = []
    for a, b, kind in confirmatory:
        if kind == "CW":
            t, p = clark_west(base, small=b, big=a)
        else:
            t, p = diebold_mariano(base, base=b, candidate=a)
        d, lo, hi = paired_r2_ci(base, b, a)
        tests.append(
            {
                "candidate": a,
                "vs": b,
                "test": kind,
                "t": t,
                "p (one-sided)": p,
                "ΔR2 %": 100 * d,
                "95% CI": f"[{100 * lo:.3f}, {100 * hi:.3f}]",
            }
        )
    h1_t, h1_p = diebold_mariano(base, base=H1[1], candidate=H1[0])
    t, p = diebold_mariano(
        demeaned_frame(base, ["NN3-dm", "ENet+H"]), base="ENet+H", candidate="NN3-dm"
    )
    t2, p2 = diebold_mariano(base, base="train_mean", candidate="ENet+H")
    other = pd.DataFrame(
        [
            {"test": "H1: NN3-dm vs constant (confirmatory)", "t": h1_t, "p (one-sided)": h1_p},
            {"test": "NN3-dm vs ENet+H, within-month errors", "t": t, "p (one-sided)": p},
            {"test": "ENet+H vs constant", "t": t2, "p (one-sided)": p2},
        ]
    )
    tests = pd.DataFrame(tests)
    for df in (tests, other):
        if not np.isfinite(df[["t", "p (one-sided)"]].to_numpy(dtype=float)).all():
            raise SystemExit("non-finite test statistic: no report")
    return {"per_model": pd.DataFrame(per_model), "tests": tests, "other": other}


def render(now: dict, before: dict, months: int, header: str) -> tuple[str, float, float]:
    """The report text: H1 and its reading first, then every number beside its 1987–2016 value."""
    h1 = now["other"].iloc[0]
    t, p = float(h1["t"]), float(h1["p (one-sided)"])
    old = {"t": "1987–2016 t", "p (one-sided)": "1987–2016 p", "R2 %": "1987–2016 R2 %"}
    per_model = now["per_model"].merge(
        before["per_model"][["model", "R2 %"]].rename(columns=old), on="model"
    )
    tests = now["tests"].merge(
        before["tests"][["candidate", "vs", "t", "p (one-sided)"]].rename(columns=old),
        on=["candidate", "vs"],
    )
    other = now["other"].merge(before["other"].rename(columns=old), on="test")

    def md(df: pd.DataFrame) -> str:
        return df.round(4).to_markdown(index=False)

    before_t = float(before["other"].iloc[0]["t"])
    text = (
        header
        + "## H1 — the one confirmatory test\n\nNN3-dm vs the constant, one-sided Diebold–Mariano: "
        + f"t = {t:.3f}, p = {p:.4f} ({'reject' if p <= 0.05 else 'not rejected'} at α = 0.05); "
        + f"1987–2016: t = {before_t:.3f}.\n\n**{reading(p, months)}**\n\n"
        + "## Reported, deciding nothing\n\n"
        + f"Bootstrap CIs use {BLOCK}-month blocks over {months} months, so they are coarse. "
        + "No D048 label and no decay ratio are reported (D050 §9).\n\n### R² by model\n\n"
        + md(per_model)
        + "\n\n### The seven D045 tests (test 1 had ≈ 23% power at 60 months)\n\n"
        + md(tests)
        + "\n\n### H1 and secondary tests\n\n"
        + md(other)
        + "\n"
    )
    return text, t, p


def _frozen_side(rep) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    preds = predictions(runs("test", rep.MODELS, FREEZE_V1))
    return preds, with_training_mean(frame(preds, rep.MODELS), rep, walk_forward())


def _write_once(path: Path, text: str) -> None:
    tmp = path.with_suffix(".partial")
    tmp.write_text(text)
    os.replace(tmp, path)


def dry_run(rep) -> int:
    _, base = _frozen_side(rep)
    before = analyse(base, rep.MODELS, rep.CONFIRMATORY)
    header = (
        "# Phase 5 report — DRY RUN on 1987–2016 (not the holdout)\n\n"
        f"Code `{_git('rev-parse', '--short', 'HEAD')}`; the freeze-v1 test predictions play the "
        "holdout, so every 1987–2016 column equals its own value.\n\n"
    )
    text, _, _ = render(before, before, int(base["yyyymm"].nunique()), header)
    got = {k: round(v[0], 3) for k, v in h1_and_test1(base).items()}
    ok = got == EXPECTED_DRY
    _write_once(DRY_OUT, text)
    status = "OK" if ok else "MISMATCH"
    print(f"dry run on 1987-2016: {got}, expected {EXPECTED_DRY}: {status} -> {DRY_OUT}")
    return 0 if ok else 1


def check2016(rep) -> int:
    head = _git("rev-parse", "--short", "HEAD")
    selected = runs("check2016", rep.MODELS, head)
    dirty = [m for m, (st, _) in selected.items() if st["git_dirty"]]
    if dirty:
        raise SystemExit(f"dirty -check2016 runs: {dirty}")
    bad = canary(predictions(selected), predictions(runs("test", rep.MODELS, FREEZE_V1)))
    n = len(rep.MODELS)
    print(
        f"2016 fold at {head}: {n - len(bad)}/{n} bit-identical to freeze-v1"
        + (f"; differ: {bad}" if bad else "")
    )
    return 1 if bad else 0


def _check_inputs(rep, sha: str, digest: str) -> None:
    """Every holdout run comes from an allowed tag, and every fit used the published panel."""
    allowed = {
        _git("rev-parse", "--short", f"{t}^{{commit}}")
        for t in ALLOWED_TAGS
        if _git("tag", "-l", t)
    }
    rows = _rows(LEDGER)
    started = [r for r in rows if r["event"] == "started" and r["name"].endswith("-holdout")]
    stray = sorted({r["git_sha"] for r in started} - allowed)
    if stray:
        raise SystemExit(f"-holdout runs from commits outside {ALLOWED_TAGS}: {stray}")
    used = {
        r["holdout_panel_sha256"]
        for r in rows
        if r["event"] == "phase5_inputs" and r["git_sha"] == sha
    }
    if used != {digest}:
        raise SystemExit(f"fits used holdout panels {used}, not only the published {digest}")
    runs("check2016", rep.MODELS, sha)  # the pre-tag evidence exists at this exact commit


def _superseded(rep, preds: dict[str, pd.DataFrame]) -> list[str]:
    """Under freeze-v2.1 (D051): freeze-v2 holdout runs that the refit did not reproduce."""
    v2 = _git("rev-parse", "--short", "freeze-v2^{commit}")
    out = []
    for m in rep.MODELS:
        try:
            old = predictions(runs("holdout", [m], v2))[m]
        except SystemExit:
            continue
        if not old.equals(preds[m]):
            out.append(m)
    return out


def holdout(rep) -> int:
    # 1. Gates and inputs: nothing about the holdout is computed until all of these pass.
    tag = require_freeze_v2()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the look happens once (D050 §10)")
    digest = require_published_lock()
    if sha256(HOLDOUT_PANEL) != digest:
        raise SystemExit("holdout panel differs from its published lock")
    sha = _git("rev-parse", "--short", "HEAD")
    _check_inputs(rep, sha, digest)
    selected = runs("holdout", rep.MODELS, sha)
    bad_runs = [m for m, (st, _) in selected.items() if st["git_dirty"] or not st["test_window"]]
    if bad_runs:
        raise SystemExit(f"dirty or non-test holdout runs: {bad_runs}")
    preds = predictions(selected)
    frozen_preds, frozen_base = _frozen_side(rep)
    bad = canary(preds, frozen_preds)
    if bad:
        raise SystemExit(f"canary fold 2016 differs from freeze-v1 for {bad}: no report (D050 §5)")
    superseded = _superseded(rep, preds) if tag == "freeze-v2.1" else []
    preds = {m: p[p["yyyymm"] >= HOLDOUT_START].reset_index(drop=True) for m, p in preds.items()}
    base = frame(preds, rep.MODELS)
    last_year = int(base["yyyymm"].max()) // 100
    folds = walk_forward(first_test_year=2017, last_test_year=last_year, allow_holdout=True)
    base = with_training_mean(base, rep, folds)
    before = analyse(frozen_base, rep.MODELS, rep.CONFIRMATORY)
    months = int(base["yyyymm"].nunique())
    n = len(rep.MODELS)
    header = (
        "# Phase 5 report — sealed holdout, Arm A (D050)\n\n"
        f"Holdout predictions from `{tag}` (`{sha}`); holdout panel `{digest[:12]}…` (published "
        f"as tag phase5-panel); test months {base['yyyymm'].min()}–{base['yyyymm'].max()} "
        f"({months}). Canary fold 2016: {n}/{n} bit-identical to freeze-v1."
        + (f" freeze-v2 runs not reproduced by freeze-v2.1: {superseded}." if superseded else "")
        + "\n\n"
    )
    # 2. The look.
    now = analyse(base, rep.MODELS, rep.CONFIRMATORY)
    text, t, p = render(now, before, months, header)
    _write_once(OUT, text)
    look = {
        "event": "phase5_look",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "tag": tag,
        "git_sha": sha,
        "h1_t": t,
        "h1_p": p,
        "months": months,
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(look) + "\n")
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
