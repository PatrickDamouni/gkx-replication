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


def test_confirmatory_family_is_the_declared_seven():
    rep = _load("final_report")
    assert len(rep.CONFIRMATORY) == 7 and rep.CONFIRMATORY[0] == ("NN3", "train_mean", "DM")
    assert not set(rep.CONFIRMATORY) & set(rep.SECONDARY)
    assert not any(a.endswith("-dm") or b.endswith("-dm") for a, b, _ in rep.CONFIRMATORY)


def test_provenance_requires_the_freeze_commit_and_clean_test_runs():
    rep = _load("final_report")
    good = {"NN3": ({"git_dirty": False, "test_window": True}, {})}
    assert rep.provenance_errors(good, "abc1234", "abc1234ffff", "", False) == []
    assert rep.provenance_errors(good, "abc1234", "def5678ffff", "", False)  # not freeze-v1
    assert rep.provenance_errors(good, "", "def5678ffff", "", False)
    stale = {"NN3": ({"git_dirty": False, "test_window": False}, {})}
    assert rep.provenance_errors(stale, "abc1234", "abc1234ffff", "", False)
    assert rep.provenance_errors(good, "abc1234", "abc1234ffff", "src/gkx/nn.py", False)
    assert rep.provenance_errors(good, "abc1234", "abc1234ffff", "", True)


def test_report_references_only_frozen_models():
    rep = _load("final_report")
    assert rep.HEADLINE in rep.MODELS and rep.HEADLINE not in rep.BASELINES
    for a, b, *_ in rep.PAIRS:
        assert a in rep.MODELS and b in [*rep.MODELS, "train_mean", "zero"]
    assert set(rep.GKX_COLUMN) <= set(rep.MODELS)
