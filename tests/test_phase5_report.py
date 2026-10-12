"""scripts/phase5_report.py on synthetic inputs (D050 §8-10): readings, canary, run selection."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/phase5_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase5_report", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reading_is_chosen_mechanically_and_says_what_it_is_not():
    mod = _load()
    assert "survived" in mod.reading(
        0.05, 60
    ) and "not evidence that machine learning" in mod.reading(0.01, 60)
    assert mod.reading(0.0501, 48).startswith("Not detected (Arm A, 48 months)")
    assert "not evidence of absence" in mod.reading(0.5, 48)


def _preds(years, seed, value=None):
    months = [y * 100 + m for y in years for m in (1, 2)]
    df = pd.DataFrame([(m, p) for m in months for p in (1, 2)], columns=["yyyymm", "permno"])
    rng = np.random.default_rng(seed)
    df["pred_seed0"] = rng.normal(size=len(df)) if value is None else value
    df["pred"] = df["pred_seed0"]
    return df


def test_canary_requires_exact_equality_on_the_2016_fold():
    mod = _load()
    frozen = {"A": _preds([2015, 2016], 0)}
    same = {"A": frozen["A"][frozen["A"]["yyyymm"] // 100 == 2016].copy()}
    assert mod.canary(same, frozen) == []
    changed = {"A": same["A"].assign(pred_seed0=same["A"]["pred_seed0"] + 1e-12)}
    assert mod.canary(changed, frozen) == ["A"]
    shuffled = {"A": same["A"].iloc[::-1]}
    assert mod.canary(shuffled, frozen) == ["A"]


def _ledger(tmp_path: Path, runs: list[tuple[str, str, bool]]) -> Path:
    ledger = tmp_path / "trials.jsonl"
    lines = []
    for i, (name, sha, dirty) in enumerate(runs):
        rel = f"preds/{i}.parquet"
        (tmp_path / "preds").mkdir(exist_ok=True)
        _preds([2017], i).to_parquet(tmp_path / rel)
        digest = hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()
        lines += [
            {"event": "started", "run_id": str(i), "name": name, "git_sha": sha, "git_dirty": dirty,
             "test_window": True},
            {"event": "finished", "run_id": str(i), "preds_path": rel, "preds_sha256": digest},
        ]  # fmt: skip
    ledger.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    return ledger


def test_runs_selects_by_tag_and_commit_and_refuses_gaps_or_mixed_commits(tmp_path):
    mod = _load()
    ledger = _ledger(
        tmp_path,
        [
            ("A-holdout", "abc1234", False),
            ("B-holdout", "abc1234", False),
            ("A-test", "46c1584", False),
        ],
    )
    picked = mod.runs("holdout", ["A", "B"], "abc1234ffff", ledger)
    assert set(picked) == {"A", "B"}
    assert mod.runs("test", ["A"], "46c1584", ledger)["A"][0]["name"] == "A-test"
    with pytest.raises(SystemExit, match="missing"):
        mod.runs("holdout", ["A", "C"], None, ledger)
    mixed = _ledger(tmp_path, [("A-holdout", "abc1234", False), ("A-holdout", "def5678", False)])
    with pytest.raises(SystemExit, match="more than one commit"):
        mod.runs("holdout", ["A"], None, mixed)


def test_predictions_refuse_a_tampered_file(tmp_path):
    mod = _load()
    ledger = _ledger(tmp_path, [("A-holdout", "abc1234", False)])
    picked = mod.runs("holdout", ["A"], None, ledger)
    (tmp_path / "preds/0.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit, match="sha256"):
        mod.predictions(picked, ledger)


def test_analyse_reports_every_section_on_a_synthetic_holdout():
    mod = _load()
    rng = np.random.default_rng(1)
    months = [201700 + m for m in range(1, 13)] + [201800 + m for m in range(1, 13)]
    base = pd.DataFrame([(m, p) for m in months for p in range(50)], columns=["yyyymm", "permno"])
    signal = rng.normal(size=len(base))
    base["target"] = 0.01 * signal + rng.normal(scale=0.1, size=len(base))
    base["me_lag"] = 1.0
    models = ["NN3", "NN3-dm", "ENet+H", "PCR"]
    for m in models:
        base[m] = 0.01 * signal + rng.normal(scale=0.001, size=len(base))
    base["zero"], base["train_mean"] = 0.0, 0.0
    conf = [("NN3", "train_mean", "DM"), ("NN3", "PCR", "DM"), ("ENet+H", "PCR", "CW")]
    res = mod.analyse(base, models, conf)
    assert list(res["per_model"]["model"]) == [*models, "train_mean"]
    assert len(res["tests"]) == 3 and len(res["secondary"]) == 2
    t, p = mod.h1_and_test1(base)["H1"]
    assert t > 2 and p < 0.05  # a planted cross-sectional signal is detected
