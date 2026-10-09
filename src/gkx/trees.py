"""GBRT+H and RF with LightGBM (D012, D037).

Inputs: rank-normalized characteristics, macro predictors and sic2 (categorical) — no
characteristic × macro interactions (trees form interactions through splits; D012).
Both run once with fixed seeds and deterministic settings. Selection is by validation MSE
(`metric="l2"` set explicitly, so a Huber objective never selects on its own metric).
num_leaves = 2**max_depth so the depth grid is not capped by LightGBM's default of 31 leaves;
features per split are passed as a fraction of the inputs. One Dataset per fold, binned on
training rows only, reused across the whole grid.
GBRT+H: Huber objective with alpha = ξ (D037 shared rule); the number of trees (≤ max_trees)
is chosen on validation, like GKX's "#Trees = 1~1000".
RF: LightGBM `rf` boosting subsamples rows without replacement (bagging_fraction 0.632, the
expected unique share of a bootstrap) — a stated deviation from GKX's bootstrap.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from gkx.linear import fold_xi

COMMON = {
    "metric": "l2",
    "verbose": -1,
    "num_threads": 8,
    "deterministic": True,
    "force_row_wise": True,
    "seed": 0,
}


class _LGBM:
    def __init__(self, char_cols, macro_cols, ind_col="sic2"):
        self.char_cols, self.macro_cols, self.ind_col = char_cols, macro_cols, ind_col

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [*self.char_cols, *self.macro_cols]
        out = X[cols].astype("float32")
        if self.ind_col is not None:
            out[self.ind_col] = pd.Categorical(X[self.ind_col], categories=self.categories_)
        return out

    def _configs(self, n_inputs: int) -> list[tuple[tuple, dict, int]]:
        raise NotImplementedError

    def fit(self, X, y, X_val, y_val):
        self.categories_ = (
            sorted(X[self.ind_col].dropna().unique()) if self.ind_col is not None else []
        )
        self._prepare(X, y)
        F, Fv = self._frame(X), self._frame(X_val)
        cat = [self.ind_col] if self.ind_col is not None else "auto"
        train = lgb.Dataset(F, y.to_numpy(), categorical_feature=cat, free_raw_data=False)
        valid = lgb.Dataset(Fv, y_val.to_numpy(), reference=train)
        yv = y_val.to_numpy(dtype=float)
        self.params_, self.val_scores_, self.best_trees_, self.models_ = {}, {}, {}, {}
        for key, params, rounds in self._configs(F.shape[1]):
            history: dict = {}
            booster = lgb.train(
                params,
                train,
                num_boost_round=rounds,
                valid_sets=[valid],
                valid_names=["val"],
                callbacks=[lgb.record_evaluation(history)],
            )
            best = self._best_iteration(history["val"]["l2"], rounds)
            pv = booster.predict(Fv, num_iteration=best)
            self.params_[key], self.best_trees_[key], self.models_[key] = params, best, booster
            self.val_scores_[key] = float(1 - ((yv - pv) ** 2).sum() / (yv @ yv))
        self.chosen_ = max(self.val_scores_, key=self.val_scores_.get)
        return self

    def _prepare(self, X, y) -> None:
        pass

    def _best_iteration(self, val_mse: list[float], rounds: int) -> int:
        return rounds

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        booster = self.models_[self.chosen_]
        return booster.predict(self._frame(X), num_iteration=self.best_trees_[self.chosen_])


class GBRT(_LGBM):
    def __init__(
        self,
        char_cols,
        macro_cols,
        ind_col="sic2",
        depths=(1, 2),
        rates=(0.01, 0.1),
        max_trees=1000,
        xi=None,
    ):
        super().__init__(char_cols, macro_cols, ind_col)
        self.depths, self.rates, self.max_trees, self.xi = depths, rates, max_trees, xi

    def _prepare(self, X, y):
        if self.xi is None:
            self.xi = fold_xi(X, y, self.char_cols, self.macro_cols, self.ind_col)

    def _configs(self, n_inputs):
        return [
            (
                (d, lr),
                COMMON
                | {
                    "objective": "huber",
                    "alpha": self.xi,
                    "max_depth": d,
                    "num_leaves": 2**d,
                    "learning_rate": lr,
                },
                self.max_trees,
            )
            for d in self.depths
            for lr in self.rates
        ]

    def _best_iteration(self, val_mse, rounds):
        """Number of trees chosen on validation MSE over the full path (no patience rule); the
        per-iteration MSE is recorded by LightGBM during training (metric l2), not re-predicted."""
        return int(np.argmin(val_mse)) + 1


class RF(_LGBM):
    def __init__(
        self,
        char_cols,
        macro_cols,
        ind_col="sic2",
        depths=(1, 2, 3, 4, 5, 6),
        features_per_split=(3, 5, 10, 20, 30, 50),
        n_trees=300,
    ):
        super().__init__(char_cols, macro_cols, ind_col)
        self.depths, self.features_per_split, self.n_trees = depths, features_per_split, n_trees

    def _configs(self, n_inputs):
        return [
            (
                (d, k),
                COMMON
                | {
                    "objective": "regression",
                    "boosting": "rf",
                    "max_depth": d,
                    "num_leaves": 2**d,
                    "bagging_fraction": 0.632,
                    "bagging_freq": 1,
                    "feature_fraction_bynode": min(k / n_inputs, 1.0),
                },
                self.n_trees,
            )
            for d in self.depths
            for k in self.features_per_split
        ]
