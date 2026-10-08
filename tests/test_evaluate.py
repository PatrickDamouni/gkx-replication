"""Seam 5: evaluate() over walk-forward folds, tested with models whose answers are known (D008, D014),
plus the test-window lock and ledger guarantees (D027, D028).

All tests use the 1975–1986 rehearsal schedule; the real 1987–2016 window is locked.
"""

import hashlib
import json

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
    started, finished = _rows(ledger)
    assert started["run_id"] == finished["run_id"] and finished["event"] == "finished"
    assert {"git_sha", "git_dirty", "counts_as_trial", "test_window"} <= started.keys()
    assert started["counts_as_trial"] is True
    assert {"r2_oos", "sharpe_ew", "sharpe_vw", "preds_path", "preds_sha256"} <= finished.keys()
    saved = tmp_path / finished["preds_path"]
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == finished["preds_sha256"]
    assert {"me_lag", "pred", "target"} <= set(pd.read_parquet(saved).columns)
