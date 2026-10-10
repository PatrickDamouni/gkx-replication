"""Seam 5: evaluate() over walk-forward folds, tested with models whose answers are known (D008, D014),
plus the test-window lock and ledger guarantees (D027, D028).

All tests use the 1975–1986 rehearsal schedule; the real 1987–2016 window is locked.
"""

import hashlib
import json
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import gkx.evaluate as ev
from gkx.evaluate import TestWindowLocked, evaluate
from gkx.metrics import r2_oos
from gkx.synthetic import make_panel
from gkx.walk_forward import walk_forward

FOLDS = walk_forward(first_test_year=1975, last_test_year=1979)  # rehearsal, 5 folds
TEST_FOLDS = walk_forward(first_test_year=2015)  # real test window (locked)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel(seed=0, firms=200, beta=0.1, sigma=1.0).assign(me_lag=1.0)


class ZeroModel:
    def fit(self, X, y, X_val, y_val):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class OracleModel(ZeroModel):
    """Knows the planted coefficient; never fits."""

    def predict(self, X):
        return 0.1 * X["x"].to_numpy()


class SpyModel(ZeroModel):
    """Records exactly what the harness hands it."""

    def __init__(self):
        self.seen: list[tuple[str, pd.DataFrame]] = []

    def fit(self, X, y, X_val, y_val):
        self.seen += [("train", X), ("val", X_val)]
        return self

    def predict(self, X):
        self.seen.append(("test", X))
        return super().predict(X)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_generator_plants_the_coefficient_it_claims(panel):
    slope = np.polyfit(panel["x"], panel["target"], 1)[0]
    assert slope == pytest.approx(0.1, abs=0.01)


def test_zero_model_scores_exactly_zero(panel):
    preds = evaluate(lambda seed: ZeroModel(), panel, FOLDS, features=["x"])
    assert r2_oos(preds, "pred") == 0.0


def test_oracle_scores_the_analytic_r2(panel):
    """Mean-zero target: R²_oos → β²/(β² + σ²) = 0.01/1.01 ≈ 0.0099."""
    preds = evaluate(lambda seed: OracleModel(), panel, FOLDS, features=["x"])
    assert r2_oos(preds, "pred") == pytest.approx(0.01 / 1.01, abs=0.003)


def test_predictions_cover_exactly_the_test_months_and_keep_me_lag(panel):
    preds = evaluate(lambda seed: ZeroModel(), panel, FOLDS, features=["x"])
    assert preds["yyyymm"].min() == 197501 and preds["yyyymm"].max() == 197912
    assert preds["yyyymm"].nunique() == 60
    assert "me_lag" in preds.columns  # value-weighted Sharpe needs it (council 4)


def test_model_only_ever_sees_feature_columns_from_its_own_windows(panel):
    spy = SpyModel()
    evaluate(lambda seed: spy, panel, FOLDS[:1], features=["x"])
    fold = FOLDS[0]
    windows = {"train": fold.train, "val": fold.val, "test": fold.test}
    for split, X in spy.seen:
        assert list(X.columns) == ["x"]  # the label is structurally unreachable (D014)
        months = panel.loc[X.index, "yyyymm"]
        assert windows[split][0] <= months.min() and months.max() <= windows[split][1]


def test_label_in_feature_list_is_refused(panel):
    with pytest.raises(ValueError, match="target"):
        evaluate(lambda seed: ZeroModel(), panel, FOLDS, features=["x", "target"])


def test_seed_ensemble_averages_predictions_and_stochastic_models_need_five_seeds(panel):
    class SeedConstant(ZeroModel):
        stochastic = True

        def __init__(self, seed):
            self.seed = seed

        def predict(self, X):
            return np.full(len(X), float(self.seed))

    preds = evaluate(SeedConstant, panel, FOLDS[:1], features=["x"], seeds=range(5))
    assert (preds["pred"] == 2.0).all()  # mean of seeds 0..4 (D006)
    assert set(preds.columns) >= {f"pred_seed{s}" for s in range(5)}
    with pytest.raises(ValueError, match="5 seeds"):
        evaluate(SeedConstant, panel, FOLDS[:1], features=["x"], seeds=range(3))


# --- D027: the 1987-2016 window is locked ---------------------------------------------------


def test_test_window_is_locked_by_default(panel):
    with pytest.raises(TestWindowLocked, match="1987"):
        evaluate(lambda seed: ZeroModel(), panel, TEST_FOLDS, features=["x"])


def test_unlocking_requires_a_clean_git_tree(panel, monkeypatch):
    monkeypatch.setattr(ev, "_git_dirty", lambda: True)
    with pytest.raises(TestWindowLocked, match="uncommitted"):
        evaluate(lambda seed: ZeroModel(), panel, TEST_FOLDS, features=["x"], unlock_test=True)


def test_test_window_runs_must_be_logged(panel, monkeypatch):
    monkeypatch.setattr(ev, "_git_dirty", lambda: False)
    with pytest.raises(TestWindowLocked, match="ledger"):
        evaluate(
            lambda seed: ZeroModel(),
            panel,
            TEST_FOLDS,
            features=["x"],
            unlock_test=True,
            ledger=None,
        )


def test_unlocked_clean_run_is_allowed(panel, monkeypatch):
    monkeypatch.setattr(ev, "_git_dirty", lambda: False)
    preds = evaluate(lambda seed: ZeroModel(), panel, TEST_FOLDS, features=["x"], unlock_test=True)
    assert preds["yyyymm"].min() == 201501


# --- ledger guarantees -----------------------------------------------------------------------


def test_started_is_logged_before_fitting_so_a_crash_still_leaves_a_record(panel, tmp_path):
    class Crashes(ZeroModel):
        def fit(self, X, y, X_val, y_val):
            raise RuntimeError("boom")

    ledger = tmp_path / "trials.jsonl"
    with pytest.raises(RuntimeError):
        evaluate(lambda seed: Crashes(), panel, FOLDS, features=["x"], name="crash", ledger=ledger)
    rows = _rows(ledger)
    assert [r["event"] for r in rows] == ["started"] and rows[0]["name"] == "crash"


def test_finished_row_records_metrics_trial_flag_and_a_hashed_predictions_file(panel, tmp_path):
    ledger = tmp_path / "trials.jsonl"
    evaluate(
        lambda seed: OracleModel(),
        panel,
        FOLDS,
        features=["x"],
        name="oracle",
        ledger=ledger,
        counts_as_trial=True,
    )
    rows = _rows(ledger)
    started, finished = rows[0], rows[-1]
    assert [r["event"] for r in rows[1:-1]] == ["fold_done"] * len(FOLDS)  # per-fold checkpoints
    assert started["run_id"] == finished["run_id"] and finished["event"] == "finished"
    assert {"git_sha", "git_dirty", "counts_as_trial", "test_window"} <= started.keys()
    assert started["counts_as_trial"] is True
    assert {"r2_oos", "sharpe_ew", "sharpe_vw", "preds_path", "preds_sha256"} <= finished.keys()
    saved = tmp_path / finished["preds_path"]
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == finished["preds_sha256"]
    assert {"me_lag", "pred", "target"} <= set(pd.read_parquet(saved).columns)


# --- council 5 -------------------------------------------------------------------------------


def test_ledger_and_results_do_not_count_as_a_dirty_tree(monkeypatch):
    """The ledger is git-tracked; writing it must not lock the rest of the single test run."""
    calls = []
    monkeypatch.setattr(ev, "_git", lambda *args: calls.append(args) or "")
    ev._git_dirty()
    assert any(":!results" in " ".join(a) for a in calls)


def test_git_failure_fails_closed(monkeypatch):
    def broken(*args):
        raise ev.GitUnavailable("not a git repository")

    monkeypatch.setattr(ev, "_git", broken)
    with pytest.raises(ev.GitUnavailable):
        ev._git_dirty()


def test_test_window_runs_record_no_metrics_until_the_final_step(panel, monkeypatch, tmp_path):
    """D027 leak (council 5): per-model R² must not be visible while later models still run."""
    monkeypatch.setattr(ev, "_git_dirty", lambda: False)
    ledger = tmp_path / "trials.jsonl"
    evaluate(
        lambda seed: OracleModel(),
        panel,
        TEST_FOLDS,
        features=["x"],
        unlock_test=True,
        ledger=ledger,
    )
    finished = _rows(ledger)[-1]
    assert finished["event"] == "finished" and "preds_sha256" in finished
    assert not {"r2_oos", "sharpe_ew", "sharpe_vw"} & finished.keys()


def test_selection_details_are_logged_per_fold(panel, tmp_path):
    """Council 5: every grid point's validation score and the chosen config go to the ledger."""

    class Selects(ZeroModel):
        def fit(self, X, y, X_val, y_val):
            self.val_scores_, self.chosen_ = {"a": 0.1, "b": 0.3}, "b"
            return self

    ledger = tmp_path / "trials.jsonl"
    evaluate(lambda seed: Selects(), panel, FOLDS[:2], features=["x"], ledger=ledger)
    finished = _rows(ledger)[-1]
    assert finished["selection"] == [
        {"refit": 1975, "seed": 0, "chosen": "b", "val_scores": {"a": 0.1, "b": 0.3}},
        {"refit": 1976, "seed": 0, "chosen": "b", "val_scores": {"a": 0.1, "b": 0.3}},
    ]


# --- per-fold checkpoints and resume (council 6 Performance) ---------------------------------


class CrashOnce(ZeroModel):
    """Fails on the first fit of the given fold, then behaves; counts fits per process."""

    fits: ClassVar[list[int]] = []
    crashed = False

    def __init__(self, crash_at: int):
        self.crash_at = crash_at

    def fit(self, X, y, X_val, y_val):
        refit = int(X_val.index.max())  # identifies the fold within this test's fixture
        CrashOnce.fits.append(refit)
        if refit == self.crash_at and not CrashOnce.crashed:
            CrashOnce.crashed = True
            raise RuntimeError("simulated crash")
        return self


def test_a_crashed_run_resumes_under_the_same_run_id_and_skips_finished_folds(panel, tmp_path):
    ledger = tmp_path / "trials.jsonl"
    folds = FOLDS[:3]
    third_fold_val_max = int(panel.index[panel["yyyymm"].between(*folds[2].val)].max())
    CrashOnce.fits, CrashOnce.crashed = [], False
    with pytest.raises(RuntimeError):
        evaluate(lambda s: CrashOnce(third_fold_val_max), panel, folds, ["x"], ledger=ledger)
    fits_before = len(CrashOnce.fits)
    preds = evaluate(lambda s: CrashOnce(third_fold_val_max), panel, folds, ["x"], ledger=ledger)
    assert len(CrashOnce.fits) - fits_before == 1  # only the crashed third fold is refit
    events = [(r["event"], r.get("refit")) for r in _rows(ledger)]
    assert events == [
        ("started", None),
        ("fold_done", 1975),
        ("fold_done", 1976),
        ("resumed", None),
        ("fold_done", 1977),
        ("finished", None),
    ]
    assert len({r["run_id"] for r in _rows(ledger)}) == 1
    clean = evaluate(lambda s: ZeroModel(), panel, folds, ["x"], ledger=None)
    pd.testing.assert_frame_equal(preds, clean)


def test_a_tampered_fold_file_is_not_trusted_on_resume(panel, tmp_path):
    ledger = tmp_path / "trials.jsonl"
    folds = FOLDS[:2]
    second_val_max = int(panel.index[panel["yyyymm"].between(*folds[1].val)].max())
    CrashOnce.fits, CrashOnce.crashed = [], False
    with pytest.raises(RuntimeError):
        evaluate(lambda s: CrashOnce(second_val_max), panel, folds, ["x"], ledger=ledger)
    fold_file = tmp_path / _rows(ledger)[1]["fold_path"]
    fold_file.write_bytes(fold_file.read_bytes() + b"tampered")
    before = len(CrashOnce.fits)
    evaluate(lambda s: CrashOnce(second_val_max), panel, folds, ["x"], ledger=ledger)
    assert len(CrashOnce.fits) - before == 2  # both folds refit: the first file failed its hash


def test_a_different_git_commit_starts_a_new_run_instead_of_resuming(panel, tmp_path, monkeypatch):
    ledger = tmp_path / "trials.jsonl"
    folds = FOLDS[:2]
    second_val_max = int(panel.index[panel["yyyymm"].between(*folds[1].val)].max())
    CrashOnce.fits, CrashOnce.crashed = [], False
    with pytest.raises(RuntimeError):
        evaluate(lambda s: CrashOnce(second_val_max), panel, folds, ["x"], ledger=ledger)
    monkeypatch.setattr(ev, "_git_sha", lambda: "deadbee")
    evaluate(lambda s: ZeroModel(), panel, folds, ["x"], ledger=ledger)
    assert [r["event"] for r in _rows(ledger)].count("started") == 2
    assert "resumed" not in [r["event"] for r in _rows(ledger)]


# --- D041/D043: demeaned-target arm, applied by the harness ------------------------------------


class RecordsTargets(ZeroModel):
    seen: ClassVar[list] = []

    def fit(self, X, y, X_val, y_val):
        RecordsTargets.seen.append((X.index, y, X_val.index, y_val))
        return self


def test_demeaned_arm_gives_models_zero_mean_targets_per_month_and_adds_the_level_back(panel):
    RecordsTargets.seen = []
    preds = evaluate(lambda s: RecordsTargets(), panel, FOLDS[:2], ["x"], target="demeaned")
    for (idx, y, vidx, yv), fold in zip(RecordsTargets.seen, FOLDS[:2], strict=True):
        months = panel.loc[idx, "yyyymm"]
        assert y.groupby(months.to_numpy()).mean().abs().max() < 1e-12
        assert yv.groupby(panel.loc[vidx, "yyyymm"].to_numpy()).mean().abs().max() < 1e-12
        raw_mean = panel.loc[panel["yyyymm"].between(*fold.train), "target"].mean()
        got = preds.loc[preds["yyyymm"].between(*fold.test), "pred"]
        np.testing.assert_allclose(got, raw_mean)  # ZeroModel output + training-window level


def test_target_arm_is_recorded_in_the_ledger(panel, tmp_path):
    ledger = tmp_path / "trials.jsonl"
    evaluate(lambda s: ZeroModel(), panel, FOLDS[:1], ["x"], ledger=ledger, target="demeaned")
    assert _rows(ledger)[0]["target"] == "demeaned"


# --- council 8: a re-run at the same commit reuses finished runs (no second look) --------------


class CountsFits(ZeroModel):
    fits: ClassVar[list[int]] = []

    def fit(self, X, y, X_val, y_val):
        CountsFits.fits.append(1)
        return self


def test_a_finished_run_is_reused_not_refit_at_the_same_commit(panel, tmp_path):
    ledger = tmp_path / "trials.jsonl"
    CountsFits.fits = []
    first = evaluate(lambda s: CountsFits(), panel, FOLDS[:2], ["x"], ledger=ledger)
    n = len(CountsFits.fits)
    again = evaluate(lambda s: CountsFits(), panel, FOLDS[:2], ["x"], ledger=ledger)
    assert len(CountsFits.fits) == n  # nothing refit
    pd.testing.assert_frame_equal(first, again)
    events = [r["event"] for r in _rows(ledger)]
    assert events.count("started") == 1 and events[-1] == "reused"


def test_a_tampered_finished_predictions_file_is_not_reused(panel, tmp_path):
    ledger = tmp_path / "trials.jsonl"
    evaluate(lambda s: ZeroModel(), panel, FOLDS[:2], ["x"], ledger=ledger)
    finished = next(r for r in _rows(ledger) if r["event"] == "finished")
    path = tmp_path / finished["preds_path"]
    path.write_bytes(path.read_bytes() + b"x")
    evaluate(lambda s: ZeroModel(), panel, FOLDS[:2], ["x"], ledger=ledger)
    assert [r["event"] for r in _rows(ledger)].count("started") == 2
