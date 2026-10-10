"""The frozen run list (scripts/run_test.py) and the pre-specified analysis
(scripts/final_report.py) must name the same models (council 8)."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FAMILIES = ("linear", "trees", "nn")
NAMES = ["permno", "yyyymm", "signal_month", "target", "me_lag", "shrcd", "sic2",
         "m_dp", "m_ep", "mvel1", "bm", "mom12m", "chx"]  # fmt: skip


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _all_specs():
    run = _load("run_test")
    return [s for f in FAMILIES for s in run.specs(f, NAMES)]


def test_listing_specs_loads_neither_lightgbm_nor_torch():
    before = {"lightgbm", "torch"} & set(sys.modules)
    _all_specs()
    assert {"lightgbm", "torch"} & set(sys.modules) == before


def test_run_list_matches_report_models_and_trial_flags():
    rep = _load("final_report")
    specs = _all_specs()
    assert [s[0] for s in specs] == rep.MODELS
    assert {s[0] for s in specs if not s[5]} == rep.BASELINES
    assert all(s[4] == ("demeaned" if s[0].endswith("-dm") else "raw") for s in specs)
    assert all(len(list(s[3])) == 5 for s in specs if s[0].startswith("NN"))


def test_report_references_only_frozen_models():
    rep = _load("final_report")
    assert rep.HEADLINE in rep.MODELS and rep.HEADLINE not in rep.BASELINES
    for a, b, *_ in rep.PAIRS:
        assert a in rep.MODELS and b in [*rep.MODELS, "train_mean", "zero"]
    assert set(rep.GKX_COLUMN) <= set(rep.MODELS)
