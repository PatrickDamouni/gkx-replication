"""Linear family via Kronecker sufficient statistics (D029, D030).

Every expected value comes from an independent computation on the explicitly materialized
GKX design z = [x ⊗ c, industry dummies]: numpy lstsq for OLS, scipy.optimize for Huber.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from gkx.linear import LinearGKX, huber_threshold

CHARS, MACRO = ["c1", "c2", "c3"], ["m_a", "m_b"]


def _panel(seed: int = 0, months: int = 24, firms: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = months * firms
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, 3)), columns=CHARS)
    macro = rng.normal(loc=[3.0, -50.0], scale=[0.5, 10.0], size=(months, 2))  # badly scaled
    X[MACRO] = np.repeat(macro, firms, axis=0)
    X["sic2"] = rng.choice([10, 20, 30, np.nan], size=n, p=[0.3, 0.3, 0.3, 0.1])
    y = pd.Series(0.02 * X["c1"] - 0.01 * X["c2"] * X["m_a"] + rng.standard_t(3, size=n) * 0.1)
    return X, y


def _design(X: pd.DataFrame, intercept: bool, industries: list[float]) -> np.ndarray:
    x = np.column_stack([np.ones(len(X)), X[MACRO].to_numpy()])
    c = X[CHARS].to_numpy()
    z = np.einsum("ij,ik->ijk", x, c).reshape(len(X), -1)
    d = np.column_stack([(X["sic2"] == s).to_numpy(float) for s in industries])
    parts = [z, d] + ([np.ones((len(X), 1))] if intercept else [])
    return np.column_stack(parts)


def _model(loss="l2", xi=None):
    return LinearGKX(char_cols=CHARS, macro_cols=MACRO, ind_col="sic2", loss=loss, xi=xi)


def test_ols_matches_brute_force_least_squares_on_the_materialized_design():
    X, y = _panel()
    Xt, _ = _panel(seed=1)
    pred = _model().fit(X, y, X, y).predict(Xt)
    Z = _design(X, intercept=False, industries=[10, 20, 30])
    theta = np.linalg.lstsq(Z, y.to_numpy(), rcond=None)[0]
    expected = _design(Xt, intercept=False, industries=[10, 20, 30]) @ theta
    np.testing.assert_allclose(pred, expected, atol=1e-9)


def test_ols3_style_model_with_intercept_and_no_macro():
    X, y = _panel()
    m = LinearGKX(char_cols=CHARS, macro_cols=[], ind_col=None, intercept=True).fit(X, y, X, y)
    Z = np.column_stack([X[CHARS].to_numpy(), np.ones(len(X))])
    theta = np.linalg.lstsq(Z, y.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(m.predict(X), Z @ theta, atol=1e-10)


def test_huber_threshold_is_the_999th_percentile_of_absolute_ols_residuals():
    X, y = _panel()
    ols = _model().fit(X, y, X, y)
    expected = np.quantile(np.abs(y.to_numpy() - ols.predict(X)), 0.999)
    assert huber_threshold(ols, X, y) == pytest.approx(expected)


def test_huber_solution_satisfies_the_first_order_optimality_condition():
    """Huber loss is convex, so the minimum is where Z'ψ(r) = 0 on the materialized design.

    (A first version compared against scipy BFGS, which stopped early with "precision loss";
    IRLS reached the lower objective. The optimality condition is the stronger check.)
    """
    X, y = _panel()
    xi = 0.15
    pred = _model(loss="huber", xi=xi).fit(X, y, X, y).predict(X)
    Z = _design(X, intercept=False, industries=[10, 20, 30])
    r = y.to_numpy() - pred
    psi = np.clip(r, -xi, xi)  # derivative of the Huber loss / 2
    grad = Z.T @ psi
    scale = np.abs(Z.T @ y.to_numpy()).max()
    assert np.abs(grad).max() <= 1e-8 * scale

    def objective(p):
        a = np.abs(y.to_numpy() - p)
        return np.where(a <= xi, a**2, 2 * xi * a - xi**2).sum()

    theta0 = np.linalg.lstsq(Z, y.to_numpy(), rcond=None)[0]
    bfgs = minimize(lambda t: objective(Z @ t), theta0, method="BFGS")
    assert objective(pred) <= objective(Z @ bfgs.x) + 1e-12


def test_huber_with_infinite_threshold_is_ols():
    X, y = _panel()
    a = _model().fit(X, y, X, y).predict(X)
    b = _model(loss="huber", xi=np.inf).fit(X, y, X, y).predict(X)
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_industry_unseen_in_training_contributes_nothing_and_does_not_crash():
    X, y = _panel()
    m = _model().fit(X, y, X, y)
    Xt = X.head(5).assign(sic2=99.0)
    Xn = X.head(5).assign(sic2=np.nan)
    np.testing.assert_allclose(m.predict(Xt), m.predict(Xn), atol=1e-12)


def test_unidentified_directions_get_zero_weight_minimum_norm():
    """A characteristic that is 0 for every training row (not yet available, median-filled) has
    no identified coefficient; its test values must not move predictions (D031)."""
    X, y = _panel()
    X["c3"] = 0.0  # unavailable throughout training
    m = _model().fit(X, y, X, y)
    Xt, _ = _panel(seed=1)  # c3 nonzero at test time
    np.testing.assert_allclose(m.predict(Xt), m.predict(Xt.assign(c3=0.0)), atol=1e-10)
    assert m.rank_ < len(m.coef_)


def test_minimum_norm_solution_matches_pinv_on_the_materialized_design():
    X, y = _panel()
    X["c3"] = 0.0
    Xt, _ = _panel(seed=1)
    theta = np.linalg.pinv(_design(X, intercept=False, industries=[10, 20, 30])) @ y.to_numpy()
    expected = _design(Xt, intercept=False, industries=[10, 20, 30]) @ theta
    np.testing.assert_allclose(_model().fit(X, y, X, y).predict(Xt), expected, atol=1e-9)
