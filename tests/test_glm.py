"""GLM+H: order-2 splines on each characteristic, group lasso, Huber (D035, GKX eqs. 13–14).

Independent references: hand-computed basis values; numpy lstsq on the materialized basis (λ→0);
the group-lasso KKT conditions on the materialized basis (Huber).
"""

import numpy as np
import pandas as pd
import pytest

from gkx.glm import GLM, KNOTS, spline_basis

CHARS, MACRO = ["c1", "c2", "c3", "c4", "c5"], ["m_a"]


def _panel(seed: int, months: int = 30, firms: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = months * firms
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, len(CHARS))), columns=CHARS)
    X["m_a"] = np.repeat(rng.normal(3.0, 0.5, months), firms)
    X["sic2"] = rng.choice([10.0, 20.0, np.nan], size=n)
    # c1 enters nonlinearly (a kink at 0), c2 linearly, c3–c5 are noise
    y = 0.01 + 0.06 * np.maximum(X["c1"], 0) ** 2 + 0.02 * X["c2"] + rng.standard_t(3, n) * 0.08
    return X, pd.Series(y)


def _B(X: pd.DataFrame) -> np.ndarray:
    return np.column_stack([spline_basis(X[CHARS].to_numpy()), np.ones(len(X))])


@pytest.fixture(scope="module")
def data():
    X, y = _panel(0)
    Xv, yv = _panel(1)
    return X, y, Xv, yv


def _glm(**kw):
    return GLM(CHARS, MACRO, "sic2", **kw)


def test_basis_values_by_hand():
    assert KNOTS == (-0.5, 0.0, 0.5)
    np.testing.assert_allclose(spline_basis(np.array([[0.25]])), [[0.25, 0.5625, 0.0625, 0.0]])
    np.testing.assert_allclose(spline_basis(np.array([[-0.75]])), [[-0.75, 0.0, 0.0, 0.0]])


def test_tiny_penalty_with_squared_loss_is_ols_on_the_spline_basis(data):
    X, y, Xv, yv = data
    model = _glm(lambdas=[1e-12], loss="l2").fit(X, y, Xv, yv)
    beta = np.linalg.lstsq(_B(X), y.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(model.predict(Xv), _B(Xv) @ beta, atol=1e-6)


def test_huge_penalty_gives_the_training_mean(data):
    X, y, Xv, yv = data
    model = _glm(lambdas=[10.0], loss="l2").fit(X, y, Xv, yv)
    np.testing.assert_allclose(model.predict(Xv), np.full(len(Xv), y.mean()), atol=1e-10)


def test_huber_group_lasso_satisfies_the_kkt_conditions(data):
    X, y, Xv, yv = data
    lam, xi = 2e-3, 0.1
    model = _glm(lambdas=[lam], loss="huber", xi=xi).fit(X, y, Xv, yv)
    B = _B(X)
    psi = np.clip(y.to_numpy() - model.predict(X), -xi, xi)
    grad = -(2 / len(psi)) * (B.T @ psi)
    beta = model.beta_
    for g in range(len(CHARS)):
        idx = slice(4 * g, 4 * g + 4)
        norm = np.linalg.norm(beta[idx])
        if norm > 0:
            np.testing.assert_allclose(grad[idx] + lam * beta[idx] / norm, 0, atol=1e-6)
        else:
            assert np.linalg.norm(grad[idx]) <= lam + 1e-6
    assert abs(grad[-1]) <= 1e-6  # unpenalized intercept


def test_group_lasso_drops_whole_noise_groups_and_keeps_the_signal(data):
    X, y, Xv, yv = data
    model = _glm(lambdas=[1e-2], loss="huber").fit(X, y, Xv, yv)
    groups = model.beta_[:-1].reshape(len(CHARS), 4)
    norms = np.linalg.norm(groups, axis=1)
    assert norms[0] > 0 and norms[1] > 0  # c1 (nonlinear), c2 (linear)
    assert (norms[2:] == 0).all()  # whole groups exactly zero, not just small


def test_lambda_is_chosen_on_validation_and_recorded(data):
    X, y, Xv, yv = data
    model = _glm(loss="huber").fit(X, y, Xv, yv)
    assert len(model.val_scores_) == 10 and model.converged_
    assert model.chosen_ == max(model.val_scores_, key=model.val_scores_.get)
    pv = model.predict(Xv)
    assert model.val_scores_[model.chosen_] == pytest.approx(
        1 - ((yv - pv) ** 2).sum() / (yv**2).sum(), abs=1e-8
    )
