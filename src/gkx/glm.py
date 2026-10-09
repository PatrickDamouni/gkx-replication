"""GLM+H: order-2 splines on each characteristic with a group lasso and Huber loss (D035).

GKX eq. 13 prints a non-truncated basis; per D035 we use the evident intent, truncated
quadratic splines with 3 knots on each rank-normalized characteristic c ∈ [−1, 1]:
    p(c) = (c, (c+0.5)₊², c₊², (c−0.5)₊²)   → 4 columns × 94 characteristics + intercept.
No macro interactions and no industry dummies: GLM isolates nonlinearity without interactions.
Penalty (GKX eq. 14): λ Σ_j ‖θ_j‖₂ over each characteristic's 4 coefficients; intercept free.
The 377-column Gram is accumulated in row chunks; Huber via IRLS whose weighted Gram subtracts
only downweighted rows; inner problem by FISTA with group soft-thresholding, stopped on KKT.
ξ follows D037: the 99.9th percentile of |OLS-all residual| on the fold's training window.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from gkx.linear import fold_xi

KNOTS = (-0.5, 0.0, 0.5)
PER_CHAR = 1 + len(KNOTS)
CHUNK = 250_000


def spline_basis(C: np.ndarray) -> np.ndarray:
    """(n, p) characteristics → (n, 4p), grouped per characteristic: c, (c−k)₊² for each knot."""
    cols = [C] + [np.maximum(C - k, 0.0) ** 2 for k in KNOTS]
    return np.stack(cols, axis=2).reshape(len(C), -1)


class GLM:
    def __init__(
        self,
        char_cols: list[str],
        macro_cols: list[str],
        ind_col: str | None = "sic2",
        lambdas: Iterable[float] = tuple(np.logspace(-1, -4, 10)),
        loss: str = "huber",
        xi: float | None = None,
        kkt_tol: float = 1e-10,
        outer_tol: float = 1e-8,
        max_iter: int = 200_000,
    ):
        self.char_cols, self.macro_cols, self.ind_col = char_cols, macro_cols, ind_col
        self.lambdas = sorted((float(v) for v in lambdas), reverse=True)
        self.loss, self.xi = loss, xi
        self.kkt_tol, self.outer_tol, self.max_iter = kkt_tol, outer_tol, max_iter

    # --- design -----------------------------------------------------------------------------
    def _design(self, C: np.ndarray) -> np.ndarray:
        return np.column_stack([spline_basis(C), np.ones(len(C))])

    def _gram(self, C: np.ndarray, y: np.ndarray):
        k = PER_CHAR * C.shape[1] + 1
        A, b = np.zeros((k, k)), np.zeros(k)
        for s in range(0, len(C), CHUNK):
            B = self._design(C[s : s + CHUNK])
            A += B.T @ B
            b += B.T @ y[s : s + CHUNK]
        return A, b

    def _apply(self, C: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [self._design(C[s : s + CHUNK]) @ beta for s in range(0, len(C), CHUNK)]
        )

    # --- solver -----------------------------------------------------------------------------
    def _prox(self, v: np.ndarray, t: float) -> np.ndarray:
        out = v.copy()
        G = v[:-1].reshape(-1, PER_CHAR)
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        scale = np.maximum(1 - t / np.maximum(norms, 1e-300), 0.0)
        out[:-1] = (G * scale).ravel()
        return out

    def _kkt(self, g: np.ndarray, beta: np.ndarray, lam: float) -> float:
        G, B = g[:-1].reshape(-1, PER_CHAR), beta[:-1].reshape(-1, PER_CHAR)
        nb = np.linalg.norm(B, axis=1)
        active = nb > 0
        v_act = np.linalg.norm(G[active] + lam * B[active] / nb[active, None], axis=1).max(
            initial=0.0
        )
        v_zero = np.maximum(np.linalg.norm(G[~active], axis=1) - lam, 0.0).max(initial=0.0)
        return max(v_act, v_zero, abs(g[-1]))

    def _fista(self, Q, q, lam, beta, L):
        def grad(b):
            return 2 * (Q @ b - q)

        x, yk, t = beta.copy(), beta.copy(), 1.0
        for it in range(1, self.max_iter + 1):
            x_new = self._prox(yk - grad(yk) / L, lam / L)
            if (yk - x_new) @ (x_new - x) > 0:  # adaptive restart
                yk, t = x.copy(), 1.0
                continue
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            yk = x_new + ((t - 1) / t_new) * (x_new - x)
            x, t = x_new, t_new
            if it % 25 == 0 and self._kkt(grad(x), x, lam) <= self.kkt_tol:
                return x, True
        return x, self._kkt(grad(x), x, lam) <= self.kkt_tol

    # --- public API -------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        if self.loss == "huber" and self.xi is None:
            self.xi = fold_xi(X, y, self.char_cols, self.macro_cols, self.ind_col)
        C = X[self.char_cols].to_numpy(dtype=float)
        yy = y.to_numpy(dtype=float)
        n = len(yy)
        A0, b0 = self._gram(C, yy)
        L = 2 * np.linalg.eigvalsh(A0 / n)[-1]  # valid for every A_w ≤ A0
        Cv = X_val[self.char_cols].to_numpy(dtype=float)
        yv = y_val.to_numpy(dtype=float)
        Av, bv = self._gram(Cv, yv)
        yyv = float(yv @ yv)

        beta = np.zeros(A0.shape[0])
        beta[-1] = yy.mean()
        self.coefs_, self.val_scores_, self.converged_ = {}, {}, True
        for lam in self.lambdas:
            if self.loss == "l2":
                beta, ok = self._fista(A0 / n, b0 / n, lam, beta, L)
            else:
                ok = False
                for _ in range(200):  # IRLS outer loop
                    a = np.abs(yy - self._apply(C, beta))
                    w = np.where(a <= self.xi, 1.0, self.xi / np.maximum(a, 1e-300))
                    out = np.flatnonzero(w < 1.0)
                    A, bb = A0, b0
                    if out.size:
                        B = self._design(C[out])
                        shrink = 1.0 - w[out]
                        A = A0 - B.T @ (B * shrink[:, None])
                        bb = b0 - B.T @ (shrink * yy[out])
                    new, inner_ok = self._fista(A / n, bb / n, lam, beta, L)
                    rel = np.max(np.abs(new - beta)) / (1 + np.max(np.abs(new)))
                    beta = new
                    if rel <= self.outer_tol:
                        ok = inner_ok
                        break
            self.converged_ &= ok
            sse = yyv - 2 * beta @ bv + beta @ Av @ beta
            self.coefs_[lam], self.val_scores_[lam] = beta.copy(), 1 - sse / yyv
        self.chosen_ = max(self.val_scores_, key=self.val_scores_.get)
        self.beta_ = self.coefs_[self.chosen_]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._apply(X[self.char_cols].to_numpy(dtype=float), self.beta_)
