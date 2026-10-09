"""ENet+H on GKX sufficient statistics (D037). GKX eq. 8: φ = λ(1−ρ)Σ|θ| + ½λρΣθ², loss (1/n)Σ.

Independent references: scikit-learn ElasticNet (squared loss) with alpha = λ/2, l1_ratio = 1 − ρ —
derived by dividing the GKX objective by 2 — and KKT conditions on the materialized design (Huber).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet

from gkx.enet import ENet

CHARS, MACRO = ["c1", "c2", "c3", "c4"], ["m_a", "m_b"]


def _panel(seed: int, months: int = 30, firms: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = months * firms
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, len(CHARS))), columns=CHARS)
    X[MACRO] = np.repeat(rng.normal([3.0, -50.0], [0.5, 10.0], size=(months, 2)), firms, axis=0)
    y = pd.Series(
        0.01 + 0.03 * X["c1"] - 0.02 * X["c2"] * (X["m_a"] - 3) + rng.standard_t(3, n) * 0.08
    )
    return X, y


def _z(X, mu, sd):
    x = np.column_stack([np.ones(len(X)), (X[MACRO].to_numpy() - mu) / sd])
    return np.einsum("ij,ik->ijk", x, X[CHARS].to_numpy()).reshape(len(X), -1)


@pytest.fixture(scope="module")
def data():
    X, y = _panel(0)
    Xv, yv = _panel(1)
    m = X[MACRO].to_numpy()
    return X, y, Xv, yv, m.mean(0), m.std(0)


@pytest.mark.parametrize("lam", [1e-3, 1e-2])
def test_squared_loss_matches_sklearn_elastic_net_with_the_derived_mapping(data, lam):
    X, y, Xv, yv, mu, sd = data
    model = ENet(CHARS, MACRO, ind_col=None, lambdas=[lam], rho=0.5, loss="l2").fit(X, y, Xv, yv)
    ref = ElasticNet(alpha=lam / 2, l1_ratio=0.5, tol=1e-12, max_iter=200_000).fit(_z(X, mu, sd), y)
    np.testing.assert_allclose(model.predict(Xv), ref.predict(_z(Xv, mu, sd)), atol=1e-6)


def test_huge_penalty_gives_the_training_mean(data):
    X, y, Xv, yv, *_ = data
    model = ENet(CHARS, MACRO, ind_col=None, lambdas=[10.0], loss="l2").fit(X, y, Xv, yv)
    np.testing.assert_allclose(model.predict(Xv), np.full(len(Xv), y.mean()), atol=1e-10)


def test_huber_solution_satisfies_the_kkt_conditions(data):
    X, y, Xv, yv, mu, sd = data
    lam, rho, xi = 3e-3, 0.5, 0.12
    model = ENet(CHARS, MACRO, ind_col=None, lambdas=[lam], rho=rho, loss="huber", xi=xi).fit(
        X, y, Xv, yv
    )
    Z = _z(X, mu, sd)
    r = y.to_numpy() - model.predict(X)
    psi = np.clip(r, -xi, xi)
    theta = model.theta_  # penalized coefficients in the same column order as Z
    grad = -(2 / len(r)) * (Z.T @ psi) + lam * rho * theta  # smooth part of the subgradient
    l1 = lam * (1 - rho)
    nz = np.abs(theta) > 1e-10
    assert np.abs(grad[nz] + l1 * np.sign(theta[nz])).max() <= 1e-6
    assert np.abs(grad[~nz]).max() <= l1 + 1e-6
    assert abs(psi.sum()) <= 1e-6 * len(r)  # unpenalized intercept: its gradient is zero


def test_lambda_is_chosen_on_validation_and_every_value_is_recorded(data):
    X, y, Xv, yv, *_ = data
    grid = np.logspace(-1, -4, 10)
    model = ENet(CHARS, MACRO, ind_col=None, lambdas=grid, loss="huber").fit(X, y, Xv, yv)
    assert len(model.val_scores_) == 10
    assert model.chosen_ == max(model.val_scores_, key=model.val_scores_.get)
    single = ENet(CHARS, MACRO, ind_col=None, lambdas=[model.chosen_], loss="huber", xi=model.xi)
    pv = single.fit(X, y, Xv, yv).predict(Xv)
    r2 = 1 - ((yv - pv) ** 2).sum() / (yv**2).sum()
    assert model.val_scores_[model.chosen_] == pytest.approx(r2, abs=1e-8)
    assert model.converged_
