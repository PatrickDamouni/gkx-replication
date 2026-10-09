"""PCR and PLS on GKX sufficient statistics (D037). Expected values come from scikit-learn run on
the explicitly materialized, demeaned design — an independent implementation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from gkx.reduction import PCR, PLS

CHARS, MACRO = ["c1", "c2", "c3", "c4"], ["m_a", "m_b"]


def _panel(seed: int, months: int = 30, firms: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = months * firms
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, len(CHARS))), columns=CHARS)
    X[MACRO] = np.repeat(rng.normal([3.0, -50.0], [0.5, 10.0], size=(months, 2)), firms, axis=0)
    y = pd.Series(0.01 + 0.03 * X["c1"] - 0.02 * X["c2"] * (X["m_a"] - 3) + rng.normal(0, 0.1, n))
    return X, y


def _z(X: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Materialized GKX design with training-standardized macro (no dummies in this fixture)."""
    x = np.column_stack([np.ones(len(X)), (X[MACRO].to_numpy() - mu) / sd])
    return np.einsum("ij,ik->ijk", x, X[CHARS].to_numpy()).reshape(len(X), -1)


@pytest.fixture(scope="module")
def data():
    X, y = _panel(0)
    Xv, yv = _panel(1)
    Xt, _ = _panel(2)
    # D040: raw macro is the default, so the reference design uses mu = 0, sd = 1
    return X, y, Xv, yv, Xt, np.zeros(len(MACRO)), np.ones(len(MACRO))


@pytest.mark.parametrize("k", [1, 3, 7])
def test_pcr_matches_sklearn_pca_then_ols_on_the_materialized_design(data, k):
    X, y, Xv, yv, Xt, mu, sd = data
    model = PCR(CHARS, MACRO, ind_col=None, k_grid=[k]).fit(X, y, Xv, yv)
    Z, Zt = _z(X, mu, sd), _z(Xt, mu, sd)
    pca = PCA(n_components=k).fit(Z)
    reg = LinearRegression().fit(pca.transform(Z), y)
    np.testing.assert_allclose(model.predict(Xt), reg.predict(pca.transform(Zt)), atol=1e-9)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_pls_matches_sklearn_pls_on_the_materialized_design(data, k):
    X, y, Xv, yv, Xt, mu, sd = data
    model = PLS(CHARS, MACRO, ind_col=None, k_grid=[k]).fit(X, y, Xv, yv)
    ref = PLSRegression(n_components=k, scale=False).fit(_z(X, mu, sd), y)
    np.testing.assert_allclose(model.predict(Xt), ref.predict(_z(Xt, mu, sd)).ravel(), atol=1e-9)


@pytest.mark.parametrize("cls", [PCR, PLS])
def test_k_zero_is_the_training_mean_forecast(data, cls):
    X, y, Xv, yv, Xt, *_ = data
    model = cls(CHARS, MACRO, ind_col=None, k_grid=[0]).fit(X, y, Xv, yv)
    np.testing.assert_allclose(model.predict(Xt), np.full(len(Xt), y.mean()), atol=1e-12)


@pytest.mark.parametrize("cls", [PCR, PLS])
def test_k_is_chosen_by_validation_r2_and_every_grid_point_is_recorded(data, cls):
    X, y, Xv, yv, *_ = data
    model = cls(CHARS, MACRO, ind_col=None, k_grid=range(13)).fit(X, y, Xv, yv)
    assert set(model.val_scores_) == set(range(13))
    assert model.chosen_ == max(model.val_scores_, key=model.val_scores_.get)
    # recompute the chosen K's validation R² from its own predictions, independently
    refit = cls(CHARS, MACRO, ind_col=None, k_grid=[model.chosen_]).fit(X, y, Xv, yv)
    pv = refit.predict(Xv)
    r2 = 1 - ((yv - pv) ** 2).sum() / (yv**2).sum()
    assert model.val_scores_[model.chosen_] == pytest.approx(r2, abs=1e-10)
