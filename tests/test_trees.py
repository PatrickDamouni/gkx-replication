"""GBRT+H and RF via LightGBM (D012, D037). Small synthetic panels; grids shrunk for speed."""

import numpy as np
import pandas as pd
import pytest

from gkx.trees import GBRT, RF

CHARS, MACRO = ["c1", "c2", "c3"], ["m_a"]
FEATS = [*CHARS, *MACRO, "sic2"]


def _panel(seed: int, n: int = 6000) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, 3)), columns=CHARS)
    X["m_a"] = rng.normal(3.0, 0.5, n)
    X["sic2"] = rng.choice([10.0, 20.0, 30.0, np.nan], size=n)
    y = 0.05 * (X["c1"] > 0.3) - 0.04 * (X["c2"] < -0.5) + rng.normal(0, 0.05, n)
    return X, pd.Series(y)


@pytest.fixture(scope="module")
def data():
    X, y = _panel(0)
    Xv, yv = _panel(1)
    return X, y, Xv, yv


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / (y**2).sum()


@pytest.mark.parametrize(
    "make",
    [
        lambda: GBRT(CHARS, MACRO, "sic2", depths=(1, 2), rates=(0.1,), max_trees=200, xi=0.1),
        lambda: RF(CHARS, MACRO, "sic2", depths=(2, 4), features_per_split=(1, 3), n_trees=50),
    ],
)
def test_learns_a_step_signal_and_selection_is_consistent(data, make):
    X, y, Xv, yv = data
    model = make().fit(X[FEATS], y, Xv[FEATS], yv)
    assert model.chosen_ == max(model.val_scores_, key=model.val_scores_.get)
    pv = model.predict(Xv[FEATS])
    assert model.val_scores_[model.chosen_] == pytest.approx(_r2(yv, pv), abs=1e-9)
    # Benchmark: the true signal's own R² on these validation rows (the attainable maximum, ≈0.27;
    # a first version asserted > 0.3, which exceeds what the DGP allows).
    oracle = 0.05 * (Xv["c1"] > 0.3) - 0.04 * (Xv["c2"] < -0.5)
    assert _r2(yv, pv) >= 0.9 * _r2(yv, oracle)


def test_refitting_is_bit_identical(data):
    X, y, Xv, yv = data
    a = RF(CHARS, MACRO, "sic2", depths=(3,), features_per_split=(2,), n_trees=30)
    b = RF(CHARS, MACRO, "sic2", depths=(3,), features_per_split=(2,), n_trees=30)
    pa = a.fit(X[FEATS], y, Xv[FEATS], yv).predict(Xv[FEATS])
    pb = b.fit(X[FEATS], y, Xv[FEATS], yv).predict(Xv[FEATS])
    np.testing.assert_array_equal(pa, pb)


def test_depth_is_not_capped_by_lightgbm_default_leaves(data):
    """num_leaves must be 2**depth; LightGBM's default 31 would cap depth 6 (council 5)."""
    X, y, Xv, yv = data
    m = RF(CHARS, MACRO, "sic2", depths=(6,), features_per_split=(3,), n_trees=5)
    m.fit(X[FEATS], y, Xv[FEATS], yv)
    params = m.params_[(6, 3)]
    assert params["num_leaves"] == 64 and params["max_depth"] == 6


def test_features_per_split_is_a_fraction_of_the_inputs(data):
    X, y, Xv, yv = data
    m = RF(CHARS, MACRO, "sic2", depths=(2,), features_per_split=(2,), n_trees=5)
    m.fit(X[FEATS], y, Xv[FEATS], yv)
    assert m.params_[(2, 2)]["feature_fraction_bynode"] == pytest.approx(2 / (len(CHARS) + 1))


def test_gbrt_uses_huber_with_alpha_xi_and_picks_the_number_of_trees_on_validation(data):
    X, y, Xv, yv = data
    m = GBRT(CHARS, MACRO, "sic2", depths=(1,), rates=(0.1,), max_trees=150, xi=0.07)
    m.fit(X[FEATS], y, Xv[FEATS], yv)
    params = m.params_[(1, 0.1)]
    assert params["objective"] == "huber" and params["alpha"] == 0.07
    assert 1 <= m.best_trees_[(1, 0.1)] <= 150


# --- D041 / D042 ------------------------------------------------------------------------------


def test_constant_fallback_when_no_config_beats_the_training_mean():
    rng = np.random.default_rng(5)
    X, _ = _panel(0)
    Xv, _ = _panel(1)
    y = pd.Series(0.01 + rng.normal(0, 0.1, len(X)))  # pure noise around a level
    yv = pd.Series(0.01 + rng.normal(0, 0.1, len(Xv)))
    m = GBRT(CHARS, MACRO, "sic2", depths=(2,), rates=(0.1,), max_trees=50, xi=0.2)
    m.fit(X[FEATS], y, Xv[FEATS], yv)
    assert m.chosen_ == "constant"
    np.testing.assert_allclose(m.predict(Xv[FEATS]), np.full(len(Xv), y.mean()))


def test_macro_is_used_only_for_xi_never_as_a_split_input(data):
    X, y, Xv, yv = data
    m = GBRT(CHARS, MACRO, "sic2", depths=(1,), rates=(0.1,), max_trees=20)
    m.fit(X[FEATS], y, Xv[FEATS], yv)
    from gkx.linear import fold_xi

    assert m.xi == pytest.approx(fold_xi(X[FEATS], y, CHARS, MACRO, "sic2"))  # OLS-all incl. macro
    booster = next(iter(m.models_.values()))
    assert not any(f.startswith("m_") for f in booster.feature_name())
