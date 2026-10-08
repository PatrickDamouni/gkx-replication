"""Walk-forward evaluation harness (seam 5).

The harness, not the model, owns the leak-prone loop: slicing each fold's
windows, handing the model only feature columns (D014), averaging seeds into
an ensemble (D006), and logging every call to the trial ledger so the deflated
Sharpe ratio gets an honest trial count.
"""

import json
import subprocess
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from gkx.metrics import r2_oos
from gkx.walk_forward import Fold, Window

LEDGER = Path("results/trials.jsonl")
MIN_SEEDS_STOCHASTIC = 5  # D006


class Model(Protocol):
    def fit(
        self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> "Model": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def _rows(panel: pd.DataFrame, w: Window) -> pd.DataFrame:
    """Rows whose return month is in the window and whose target is known."""
    return panel[panel["yyyymm"].between(*w) & panel["target"].notna()]


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def evaluate(
    make_model: Callable[[int], Model],
    panel: pd.DataFrame,
    folds: list[Fold],
    features: list[str],
    seeds: Iterable[int] = (0,),
    name: str = "unnamed",
    ledger: Path | None | str = "default",
) -> pd.DataFrame:
    """Fit on train (+ val for tuning), predict each fold's test year; return stacked predictions."""
    seeds = list(seeds)
    if "target" in features:
        raise ValueError("'target' is in the feature list; models must never see the label (D014)")
    if getattr(make_model(seeds[0]), "stochastic", False) and len(seeds) < MIN_SEEDS_STOCHASTIC:
        raise ValueError(
            f"stochastic model needs >= {MIN_SEEDS_STOCHASTIC} seeds, got {len(seeds)} (D006)"
        )

    parts = []
    for fold in folds:
        train, val, test = (_rows(panel, w) for w in (fold.train, fold.val, fold.test))
        out = test[["yyyymm", "permno", "target"]].copy()
        for seed in seeds:
            model = make_model(seed).fit(
                train[features], train["target"], val[features], val["target"]
            )
            out[f"pred_seed{seed}"] = model.predict(test[features])
        out["pred"] = out[[f"pred_seed{s}" for s in seeds]].mean(axis=1)
        parts.append(out)
    preds = pd.concat(parts, ignore_index=True)

    ledger = LEDGER if ledger == "default" else ledger
    if ledger is not None:
        ledger = Path(ledger)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "name": name,
            "features": features,
            "seeds": seeds,
            "folds": [f.refit for f in folds],
            "r2_oos": r2_oos(preds, "pred"),
        }
        with ledger.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return preds
