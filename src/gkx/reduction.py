"""PCR and PLS for the GKX design, computed from sufficient statistics (D037).

Both work on demeaned Z (GKX footnote 12) built from the same Kronecker statistics as the
linear family (LinearGKX with an intercept): with A = Z̃'Z̃ and b = Z̃'y where Z̃ = [Z, 1],
    μ = A[:p, p]/n,  Σzz = A[:p, :p]/n − μμ',  Σzy = b[:p]/n − μ ȳ.
Every K in the grid comes from one decomposition; K is chosen by validation R² (MSE), computed
in closed form from validation statistics: SSE(β) = y'y − 2β'b_v + β'A_vβ. Estimates use the
training window only (no refit, D037). K = 0 is the training-mean forecast.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from gkx.linear import EIG_RTOL, LinearGKX


class _Reduction:
    def __init__(
        self,
        char_cols: list[str],
        macro_cols: list[str],
        ind_col: str | None = "sic2",
        k_grid: Iterable[int] = range(101),
        scale_macro: bool = False,  # D040: raw macro, as GKX eq. 21
    ):
        self.k_grid = sorted(set(k_grid))
        self.base = LinearGKX(
            char_cols, macro_cols, ind_col, intercept=True, scale_macro=scale_macro
        )

    # subclasses return {K: θ over the p non-intercept columns}
    def _paths(self, szz: np.ndarray, szy: np.ndarray) -> dict[int, np.ndarray]:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        b = self.base
        b._setup(X)

        yy = y.to_numpy(dtype=float)
        A, bz = b._normal_equations(X, yy, np.ones(len(yy)))
        n, p = len(yy), A.shape[0] - 1
        mu, ybar = A[:p, p] / n, bz[p] / n
        szz = A[:p, :p] / n - np.outer(mu, mu)
        szy = bz[:p] / n - mu * ybar

        yv = y_val.to_numpy(dtype=float)
        Av, bv = b._normal_equations(X_val, yv, np.ones(len(yv)))
        yyv = float(yv @ yv)

        self.coefs_, self.val_scores_ = {}, {}
        for k, theta in self._paths(szz, szy).items():
            beta = np.append(theta, ybar - mu @ theta)
            sse = yyv - 2 * beta @ bv + beta @ Av @ beta
            self.coefs_[k], self.val_scores_[k] = beta, 1 - sse / yyv
        self.chosen_ = max(self.val_scores_, key=self.val_scores_.get)
        b.coef_ = self.coefs_[self.chosen_]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base.predict(X)


class PCR(_Reduction):
    """Principal components of demeaned Z, then OLS on the first K components."""

    def _paths(self, szz, szy):
        vals, vecs = np.linalg.eigh(szz)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        rank = int((vals > EIG_RTOL * vals.max()).sum())
        proj = vecs.T @ szy
        return {k: vecs[:, :k] @ (proj[:k] / vals[:k]) for k in self.k_grid if k <= rank}


class PLS(_Reduction):
    """SIMPLS (de Jong 1993) for one response, run on covariances instead of rows."""

    def _paths(self, szz, szy):
        kmax = max(self.k_grid)
        s = szy.copy()
        R, Q, V = [], [], []
        scale = np.linalg.norm(szy)
        lam_max = np.linalg.eigvalsh(szz)[-1]
        for _ in range(kmax):
            norm = np.linalg.norm(s)
            if norm <= EIG_RTOL * scale:
                break  # nothing left to explain: the remaining K are not available
            r = s / norm
            tt = r @ szz @ r  # variance along the direction (Rayleigh quotient)
            if tt <= EIG_RTOL * lam_max:
                break
            r /= np.sqrt(tt)
            p_load = szz @ r
            v = p_load - sum((vk @ p_load) * vk for vk in V) if V else p_load
            v /= np.linalg.norm(v)
            s = s - v * (v @ s)
            R.append(r)
            Q.append(szy @ r)
            V.append(v)
        R, Q = np.array(R).T, np.array(Q)
        out = {0: np.zeros(len(szy))} if 0 in self.k_grid else {}
        for k in self.k_grid:
            if 0 < k <= len(Q):
                out[k] = R[:, :k] @ Q[:k]
        return out
