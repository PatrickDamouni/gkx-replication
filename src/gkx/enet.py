"""Elastic net (+Huber) on GKX sufficient statistics (D037).

Objective, GKX eqs. 4/7/8 with an unpenalized intercept (D037):
    (1/n) Σ ℓ(y_i − z̃_i'β) + λ(1−ρ) Σ_j |θ_j| + ½ λρ Σ_j θ_j²,
ℓ = squared loss or Huber(ξ). Squared loss is a quadratic in the sufficient statistics
A = Z̃'Z̃, b = Z̃'y, so the solver never touches rows. Huber is handled by IRLS
(majorize–minimize): each outer step is a weighted quadratic with A_w = A − Σ_out (1−w) z z'.
The inner problem is solved by FISTA (accelerated proximal gradient, as GKX App. B.1) with
adaptive restart, stopped on the KKT conditions. λ runs along a descending path with warm starts;
λ is chosen by validation R² computed from validation statistics.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from gkx.linear import LinearGKX, huber_threshold


def _soft(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


class ENet:
    def __init__(
        self,
        char_cols: list[str],
        macro_cols: list[str],
        ind_col: str | None = "sic2",
        lambdas: Iterable[float] = tuple(np.logspace(-1, -4, 10)),
        rho: float = 0.5,
        loss: str = "huber",
        xi: float | None = None,
        kkt_tol: float = 1e-10,
        max_iter: int = 200_000,
    ):
        self.lambdas = sorted((float(v) for v in lambdas), reverse=True)
        self.rho, self.loss, self.xi = rho, loss, xi
        self.kkt_tol, self.max_iter = kkt_tol, max_iter
        self.outer_tol = 1e-8  # above the inner KKT noise floor (~1e-10); D039
        self.base = LinearGKX(char_cols, macro_cols, ind_col, intercept=True)

    # --- inner solver ---------------------------------------------------------------------
    def _fista(self, Q, q, lam, beta, L):
        """min β'Qβ − 2q'β + λ(1−ρ)‖θ‖₁ + ½λρ‖θ‖²  (θ = β without the last, intercept, entry)."""
        l1, l2 = lam * (1 - self.rho), lam * self.rho
        pen = np.ones_like(beta)
        pen[-1] = 0.0

        def grad(b):
            return 2 * (Q @ b - q) + l2 * pen * b

        def prox(v):
            out = v.copy()
            out[:-1] = _soft(v[:-1], l1 / L)
            return out

        x, yk, t = beta.copy(), beta.copy(), 1.0
        for it in range(1, self.max_iter + 1):
            x_new = prox(yk - grad(yk) / L)
            if (yk - x_new) @ (x_new - x) > 0:  # adaptive restart (O'Donoghue & Candès)
                yk, t = x.copy(), 1.0
                continue
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            yk = x_new + ((t - 1) / t_new) * (x_new - x)
            x, t = x_new, t_new
            if it % 25 == 0:
                self.last_kkt_, self.last_iter_ = self._kkt(grad(x), x, l1), it
                if self.last_kkt_ <= self.kkt_tol:
                    return x, True
        self.last_kkt_, self.last_iter_ = self._kkt(grad(x), x, l1), self.max_iter
        return x, self.last_kkt_ <= self.kkt_tol

    @staticmethod
    def _kkt(g, b, l1):
        th, gt = b[:-1], g[:-1]
        nz = th != 0
        viol = np.where(nz, np.abs(gt + l1 * np.sign(th)), np.maximum(np.abs(gt) - l1, 0.0))
        return max(viol.max(initial=0.0), abs(g[-1]))

    # --- public API -----------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        b = self.base
        m = X[b.macro_cols].to_numpy(dtype=float) if b.macro_cols else np.zeros((len(X), 0))
        b.mu_, b.sd_ = m.mean(0), np.where(m.std(0) > 0, m.std(0), 1.0)
        b.industries_ = sorted(X[b.ind_col].dropna().unique()) if b.ind_col is not None else []

        yy = y.to_numpy(dtype=float)
        n = len(yy)
        A0, b0 = b._normal_equations(X, yy, np.ones(n))
        L = 2 * np.linalg.eigvalsh(A0 / n)[-1] + max(self.lambdas) * self.rho  # bound for A_w ≤ A0
        groups = list(b._groups(X))
        if self.loss == "huber" and self.xi is None:  # D037: ξ from the fold's OLS residuals
            b._identify(A0)
            b.coef_ = b._solve(A0, b0)
            self.xi = huber_threshold(b, X, y)

        yv = y_val.to_numpy(dtype=float)
        Av, bv = b._normal_equations(X_val, yv, np.ones(len(yv)))
        yyv = float(yv @ yv)

        beta = np.zeros(A0.shape[0])
        beta[-1] = yy.mean()
        self.coefs_, self.val_scores_, self.converged_, self.diagnostics_ = {}, {}, True, {}
        for lam in self.lambdas:
            deltas: list[float] = []
            if self.loss == "l2":
                beta, ok = self._fista(A0 / n, b0 / n, lam, beta, L)
            else:
                deltas = []
                for _ in range(200):  # IRLS outer loop
                    b.coef_ = beta
                    a = np.abs(yy - b.predict(X, groups))
                    w = np.where(a <= self.xi, 1.0, self.xi / np.maximum(a, 1e-300))
                    out = np.flatnonzero(w < 1.0)
                    A, bb = A0, b0
                    if out.size:
                        Z = b._design_rows(X.iloc[out])
                        shrink = 1.0 - w[out]
                        A = A0 - Z.T @ (Z * shrink[:, None])
                        bb = b0 - Z.T @ (shrink * yy[out])
                    new, ok = self._fista(A / n, bb / n, lam, beta, L)
                    rel = np.max(np.abs(new - beta)) / (1 + np.max(np.abs(new)))
                    deltas.append(float(rel))
                    done = rel <= self.outer_tol
                    beta = new
                    if done:
                        break
                else:
                    ok = False
            self.converged_ &= ok
            self.diagnostics_[lam] = {
                "kkt": self.last_kkt_,
                "iters": self.last_iter_,
                "ok": ok,
                "outer": len(deltas) if self.loss == "huber" else 0,
                "outer_deltas": deltas[-3:] if self.loss == "huber" else [],
            }
            sse = yyv - 2 * beta @ bv + beta @ Av @ beta
            self.coefs_[lam], self.val_scores_[lam] = beta.copy(), 1 - sse / yyv
        self.chosen_ = max(self.val_scores_, key=self.val_scores_.get)
        b.coef_ = self.coefs_[self.chosen_]
        self.theta_ = b.coef_[:-1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base.predict(X)
