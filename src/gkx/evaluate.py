"""Walk-forward evaluation harness (seam 5).

The harness, not the model, owns the leak-prone loop: slicing each fold's windows,
handing the model only feature columns (D014), averaging seeds into an ensemble (D006),
and logging every call to the trial ledger.

Test-window discipline (D027): folds whose test months reach 1987-01 or later are refused
unless `unlock_test=True` from a clean git tree, and such runs must be logged. A `started`
row is written before any fitting, so a crash still leaves a record; a `finished` row adds
metrics and the sha256 of the persisted predictions, so no metric ever needs a re-run.
"""

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from gkx.metrics import decile_long_short, r2_oos, sharpe
from gkx.walk_forward import Fold, Window

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "results" / "trials.jsonl"  # absolute: never depends on the working directory
TEST_START = 198701  # first month of GKX's test window
MIN_SEEDS_STOCHASTIC = 5  # D006


class TestWindowLocked(RuntimeError):
    """Raised when code tries to evaluate on 1987-2016 without the D027 unlock."""

    __test__ = False  # not a pytest test class despite the name


class GitUnavailable(RuntimeError):
    """git could not be queried; the clean-tree guard must fail closed, not pass."""


class Model(Protocol):
    def fit(
        self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> "Model": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def _rows(panel: pd.DataFrame, w: Window) -> pd.DataFrame:
    """Rows whose return month is in the window and whose target is known."""
    return panel[panel["yyyymm"].between(*w) & panel["target"].notna()]


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise GitUnavailable(out.stderr.strip() or f"git {' '.join(args)} failed")
    return out.stdout.strip()


def _git_sha() -> str:
    return _git("rev-parse", "--short", "HEAD") or "unknown"


def _git_dirty() -> bool:
    """Uncommitted changes outside results/ (the ledger itself is written during the run)."""
    return bool(_git("status", "--porcelain", "--", ".", ":!results"))


def _append(ledger: Path, row: dict) -> None:
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _sharpes(preds: pd.DataFrame) -> tuple[float | None, float | None]:
    if "me_lag" not in preds.columns:
        return None, None
    ls = decile_long_short(preds, "pred")
    if len(ls) < 2:
        return None, None
    return sharpe(ls["ew"]), sharpe(ls["vw"])


def _plain(value):
    """JSON-safe version of a chosen config (numpy scalars, tuples, dicts)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _unfinished(ledger: Path, resume_key: str) -> tuple[str, dict] | None:
    """Latest run with this resume key that never finished, and its checkpointed folds."""
    if not ledger.exists():
        return None
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    runs = [
        r["run_id"] for r in rows if r["event"] == "started" and r.get("resume_key") == resume_key
    ]
    if not runs:
        return None
    run_id = runs[-1]
    if any(r["event"] == "finished" and r["run_id"] == run_id for r in rows):
        return None
    done = {r["refit"]: r for r in rows if r["event"] == "fold_done" and r["run_id"] == run_id}
    return run_id, done


def _run_id(name: str, features: list[str]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{name}-{hashlib.sha1(str(features).encode()).hexdigest()[:6]}"


def evaluate(
    make_model: Callable[[int], Model],
    panel: pd.DataFrame,
    folds: list[Fold],
    features: list[str],
    seeds: Iterable[int] = (0,),
    name: str = "unnamed",
    ledger: Path | None | str = "default",
    unlock_test: bool = False,
    counts_as_trial: bool = False,
    target: str = "raw",
) -> pd.DataFrame:
    """Fit on train (+ val for tuning), predict each fold's test year; return stacked predictions.

    target="demeaned" (D041/D043 arm 2): the harness demeans the training and validation targets
    within their own months, so models never need dates, and adds the training-window mean of the
    raw target back to the predictions.
    """
    if target not in ("raw", "demeaned"):
        raise ValueError(f"unknown target arm {target!r}")
    seeds = list(seeds)
    ledger = LEDGER if ledger == "default" else ledger
    if "target" in features:
        raise ValueError("'target' is in the feature list; models must never see the label (D014)")
    if getattr(make_model(seeds[0]), "stochastic", False) and len(seeds) < MIN_SEEDS_STOCHASTIC:
        raise ValueError(
            f"stochastic model needs >= {MIN_SEEDS_STOCHASTIC} seeds, got {len(seeds)} (D006)"
        )
    test_window = any(f.test[1] >= TEST_START for f in folds)
    dirty = _git_dirty()
    if test_window:
        if not unlock_test:
            raise TestWindowLocked(
                "folds reach the 1987-2016 test window; develop on the 1975-1986 rehearsal "
                "schedule and unlock only for the single frozen run (D027)"
            )
        if dirty:
            raise TestWindowLocked("test-window run refused: uncommitted changes in the git tree")
        if ledger is None:
            raise TestWindowLocked("test-window runs must be logged to the ledger (D027)")

    sha = _git_sha()
    resume_key = hashlib.sha1(
        json.dumps([name, sha, features, seeds, [f.refit for f in folds], target]).encode()
    ).hexdigest()
    run_id, done = _run_id(name, features), {}
    if ledger is not None:
        ledger = Path(ledger)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        previous = _unfinished(ledger, resume_key)
        if previous is not None:  # same code, inputs and folds: resume, don't start a new look
            run_id, done = previous
            _append(ledger, {"event": "resumed", "run_id": run_id, "timestamp": _now()})
        else:
            _append(
                ledger,
                {
                    "event": "started",
                    "run_id": run_id,
                    "resume_key": resume_key,
                    "timestamp": _now(),
                    "git_sha": sha,
                    "git_dirty": dirty,
                    "name": name,
                    "features": features,
                    "seeds": seeds,
                    "folds": [f.refit for f in folds],
                    "test_window": test_window,
                    "counts_as_trial": counts_as_trial,
                    "target": target,
                },
            )

    keep = [c for c in ("yyyymm", "permno", "target", "me_lag") if c in panel.columns]
    parts, selection = [], []
    for fold in folds:
        cached = done.get(fold.refit)
        if (
            cached is not None
            and _sha(ledger.parent / cached["fold_path"]) == cached["fold_sha256"]
        ):
            parts.append(pd.read_parquet(ledger.parent / cached["fold_path"]))
            selection += cached.get("selection", [])
            continue
        train, val, test = (_rows(panel, w) for w in (fold.train, fold.val, fold.test))
        out = test[keep].copy()
        fold_selection = []
        y_tr, y_va, level = train["target"], val["target"], 0.0
        if target == "demeaned":
            y_tr = y_tr - train.groupby("yyyymm")["target"].transform("mean")
            y_va = y_va - val.groupby("yyyymm")["target"].transform("mean")
            level = float(train["target"].mean())
        for seed in seeds:
            model = make_model(seed).fit(train[features], y_tr, val[features], y_va)
            out[f"pred_seed{seed}"] = model.predict(test[features]) + level
            if hasattr(model, "val_scores_"):  # grid points are scored on validation only
                fold_selection.append(
                    {
                        "refit": fold.refit,
                        "seed": seed,
                        "chosen": _plain(model.chosen_),
                        "val_scores": {str(k): float(v) for k, v in model.val_scores_.items()},
                    }
                    | (
                        {"converged": bool(model.converged_)}
                        if hasattr(model, "converged_")
                        else {}
                    )
                    | (
                        {"best_trees": {str(k): int(v) for k, v in model.best_trees_.items()}}
                        if hasattr(model, "best_trees_")
                        else {}
                    )
                )
        out["pred"] = out[[f"pred_seed{s}" for s in seeds]].mean(axis=1)
        parts.append(out)
        selection += fold_selection
        if ledger is not None:  # checkpoint: a crash later never loses this fold
            rel = Path("preds") / run_id / f"fold_{fold.refit}.parquet"
            (ledger.parent / rel).parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(ledger.parent / rel, compression="zstd", index=False)
            _append(
                ledger,
                {
                    "event": "fold_done",
                    "run_id": run_id,
                    "refit": fold.refit,
                    "timestamp": _now(),
                    "fold_path": str(rel),
                    "fold_sha256": _sha(ledger.parent / rel),
                }
                | ({"selection": fold_selection} if fold_selection else {}),
            )
    preds = pd.concat(parts, ignore_index=True)

    if ledger is not None:
        rel = Path("preds") / f"{run_id}.parquet"
        path = ledger.parent / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(path, compression="zstd", index=False)
        row = {
            "event": "finished",
            "run_id": run_id,
            "timestamp": _now(),
            "preds_path": str(rel),
            "preds_sha256": _sha(path),
        }
        if selection:
            row["selection"] = selection
        if not test_window:  # test-window metrics are computed only in the final step (D027)
            ew, vw = _sharpes(preds)
            row |= {"r2_oos": r2_oos(preds, "pred"), "sharpe_ew": ew, "sharpe_vw": vw}
        _append(ledger, row)
    return preds
