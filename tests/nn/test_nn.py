"""NN1–NN3 (GKX Table A.5; D040 raw macro; D043). Run with: GKX_TORCH=1 uv run pytest tests/nn

Synthetic sets have ~8k rows, so tests use batch=500: with GKX's 10,000 an epoch would be a single
step (first version of these tests failed for exactly that reason, not a model bug).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from gkx.nn import NN, FactoredInput

CHARS, MACRO = ["c1", "c2", "c3"], ["m_a", "m_b"]
FEATS = [*CHARS, *MACRO, "sic2"]


def _panel(seed: int, n: int = 8000) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, 3)), columns=CHARS)
    X["m_a"] = np.repeat(rng.normal(0.5, 0.3, n // 100), 100)
    X["m_b"] = np.repeat(rng.normal(-3.0, 0.4, n // 100), 100)
    X["sic2"] = rng.choice([10.0, 20.0, np.nan], size=n)
    signal = 0.05 * np.tanh(3 * X["c1"]) + 0.04 * X["c2"] * X["m_a"]
    return X, pd.Series(signal + rng.normal(0, 0.03, n))


@pytest.fixture(scope="module")
def data():
    X, y = _panel(0)
    Xv, yv = _panel(1)
    return X, y, Xv, yv


def test_factored_first_layer_equals_the_materialized_z_times_weights():
    """h = z W with z = [x ⊗ c, sic2 dummies], x = (1, macro): computed without building z."""
    torch.manual_seed(0)
    layer = FactoredInput(n_chars=3, n_macro=2, n_industries=4, width=5)
    rng = np.random.default_rng(0)
    C = torch.tensor(rng.uniform(-1, 1, (7, 3)), dtype=torch.float32)
    M = torch.tensor(rng.normal(size=(7, 2)), dtype=torch.float32)
    ind = torch.tensor([0, 1, 2, 3, -1, 0, 2])
    x = torch.cat([torch.ones(7, 1), M], dim=1)
    z = torch.einsum("bj,bk->bjk", x, C).reshape(7, -1)
    D = torch.zeros(7, 4)
    D[ind >= 0, ind[ind >= 0]] = 1.0
    W = torch.cat([layer.w_z.reshape(-1, 5), layer.w_ind], dim=0)
    expected = torch.cat([z, D], dim=1) @ W + layer.bias
    torch.testing.assert_close(layer(C, M, ind), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("hidden", [(32,), (32, 16), (32, 16, 8)])
def test_learns_a_planted_nonlinear_interaction_signal(data, hidden):
    X, y, Xv, yv = data
    m = NN(
        CHARS, MACRO, "sic2", batch=500, hidden=hidden, seed=0, lrs=(0.01,), l1s=(1e-5,), epochs=60
    )
    pv = m.fit(X[FEATS], y, Xv[FEATS], yv).predict(Xv[FEATS])
    oracle = 0.05 * np.tanh(3 * Xv["c1"]) + 0.04 * Xv["c2"] * Xv["m_a"]
    r2 = lambda p: 1 - ((yv - p) ** 2).sum() / (yv**2).sum()
    assert r2(pv) >= 0.8 * r2(oracle)


def test_same_seed_is_bit_identical_and_seeds_differ(data):
    X, y, Xv, yv = data
    kw = {"hidden": (32, 16), "lrs": (0.01,), "l1s": (1e-5,), "epochs": 5}
    a = (
        NN(CHARS, MACRO, "sic2", batch=500, seed=1, **kw)
        .fit(X[FEATS], y, Xv[FEATS], yv)
        .predict(Xv[FEATS])
    )
    b = (
        NN(CHARS, MACRO, "sic2", batch=500, seed=1, **kw)
        .fit(X[FEATS], y, Xv[FEATS], yv)
        .predict(Xv[FEATS])
    )
    c = (
        NN(CHARS, MACRO, "sic2", batch=500, seed=2, **kw)
        .fit(X[FEATS], y, Xv[FEATS], yv)
        .predict(Xv[FEATS])
    )
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_grid_is_scored_on_validation_with_early_stopping(data):
    X, y, Xv, yv = data
    m = NN(
        CHARS,
        MACRO,
        "sic2",
        batch=500,
        hidden=(32,),
        seed=0,
        lrs=(0.001, 0.01),
        l1s=(1e-5, 1e-3),
        epochs=40,
    )
    m.fit(X[FEATS], y, Xv[FEATS], yv)
    assert set(m.val_scores_) == {
        (0.001, 1e-5),
        (0.001, 1e-3),
        (0.01, 1e-5),
        (0.01, 1e-3),
        "constant",
    }
    assert m.chosen_ == max(m.val_scores_, key=m.val_scores_.get)
    assert all(1 <= e <= 40 for e in m.best_epochs_.values())
    pv = m.predict(Xv[FEATS])
    assert m.val_scores_[m.chosen_] == pytest.approx(
        1 - ((yv - pv) ** 2).sum() / (yv**2).sum(), abs=1e-6
    )


def test_constant_fallback_on_pure_noise():
    rng = np.random.default_rng(3)
    X, _ = _panel(0, n=3000)
    Xv, _ = _panel(1, n=3000)
    y, yv = pd.Series(0.01 + rng.normal(0, 0.1, 3000)), pd.Series(0.01 + rng.normal(0, 0.1, 3000))
    m = NN(
        CHARS, MACRO, "sic2", batch=500, hidden=(32,), seed=0, lrs=(0.01,), l1s=(1e-5,), epochs=10
    )
    m.fit(X[FEATS], y, Xv[FEATS], yv)
    assert m.chosen_ == "constant"
    np.testing.assert_allclose(m.predict(Xv[FEATS]), np.full(3000, y.mean()), rtol=1e-6)
