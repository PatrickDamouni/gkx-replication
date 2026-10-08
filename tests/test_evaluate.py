"""Seam 5: evaluate() over walk-forward folds, tested with models whose answers are known (D008, D014)."""

import json

import numpy as np
import pandas as pd
import pytest

from gkx.evaluate import evaluate
from gkx.metrics import r2_oos
from gkx.synthetic import make_panel
from gkx.walk_forward import walk_forward

FOLDS = walk_forward(first_test_year=2012)  # 5 folds keeps the suite fast


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel(seed=0, firms=200, beta=0.1, sigma=1.0)


class ZeroModel:
    def fit(self, X, y, X_val, y_val):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class OracleModel:
    """Knows the planted coefficient; never fits."""

    def fit(self, X, y, X_val, y_val):
        return self

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


def test_predictions_cover_exactly_the_test_months(panel):
    preds = evaluate(lambda seed: ZeroModel(), panel, FOLDS, features=["x"])
    assert preds["yyyymm"].min() == 201201 and preds["yyyymm"].max() == 201612
    assert preds["yyyymm"].nunique() == 60


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


def test_every_call_is_appended_to_the_trial_ledger(panel, tmp_path):
    ledger = tmp_path / "trials.jsonl"
    evaluate(lambda seed: ZeroModel(), panel, FOLDS, features=["x"], name="zero", ledger=ledger)
    evaluate(lambda seed: OracleModel(), panel, FOLDS, features=["x"], name="oracle", ledger=ledger)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["name"] for r in rows] == ["zero", "oracle"]
    assert {"timestamp", "git_sha", "features", "seeds", "folds", "r2_oos"} <= rows[0].keys()
