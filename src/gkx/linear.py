"""GKX linear models via Kronecker sufficient statistics (D029, D030).

Design per row: z = [x ⊗ c, industry dummies, (intercept)], x = (1, scaled macro), c = chars.
Within any group of rows sharing the same macro vector x,
    Σ z z' over the group = (x x') ⊗ (C'WC)  (and analogous cross terms),
so X'WX and X'Wy are accumulated per group without ever materializing the 920 columns.
Grouping is by the macro values themselves, which is exact and needs no date column (D014).

Macro predictors are centred/scaled with training-window constants: with the constant in x,
span((1, x) ⊗ c) is unchanged, so OLS/Huber predictions are identical and only conditioning
improves (D030). Huber is fit by IRLS with ξ = 99.9th percentile of |OLS residual| (D030).
"""

import numpy as np
import pandas as pd

HUBER_QUANTILE = 0.999
EIG_RTOL = (
    1e-12  # eigenvalues of X'WX below this fraction of the largest are treated as zero (D031)
)


def huber_threshold(ols: "LinearGKX", X: pd.DataFrame, y: pd.Series) -> float:
    return float(np.quantile(np.abs(y.to_numpy() - ols.predict(X)), HUBER_QUANTILE))


class LinearGKX:
    def __init__(
        self,
        char_cols: list[str],
        macro_cols: list[str],
        ind_col: str | None = None,
        intercept: bool = False,
        loss: str = "l2",
        xi: float | None = None,
        max_iter: int = 500,
        tol: float = 1e-12,
    ):
        self.char_cols, self.macro_cols, self.ind_col = char_cols, macro_cols, ind_col
        self.intercept, self.loss, self.xi, self.max_iter, self.tol = (
            intercept,
            loss,
            xi,
            max_iter,
            tol,
        )
        self.fast_irls = True  # Huber: X'WX = X'X − Σ_out (1−w) z z'  (only downweighted rows)

    # --- layout ---------------------------------------------------------------------------
    def _groups(self, X: pd.DataFrame):
        """Yield (x̃ vector, row positions) for each group of rows sharing a macro vector."""
        if not self.macro_cols:
            yield np.ones(1), np.arange(len(X))
            return
        m = (X[self.macro_cols].to_numpy(dtype=float) - self.mu_) / self.sd_
        codes = pd.factorize(pd.MultiIndex.from_arrays(m.T))[0]
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        for rows in np.split(order, bounds):
            yield np.concatenate([[1.0], m[rows[0]]]), rows

    def _ind_codes(self, X: pd.DataFrame) -> np.ndarray:
        if self.ind_col is None:
            return np.full(len(X), -1)
        return pd.Categorical(X[self.ind_col], categories=self.industries_).codes  # unseen → -1

    # --- normal equations -----------------------------------------------------------------
    def _design_rows(self, X: pd.DataFrame) -> np.ndarray:
        """Materialize z for a small set of rows (the Huber-downweighted ones)."""
        C = X[self.char_cols].to_numpy(dtype=float)
        if self.macro_cols:
            m = (X[self.macro_cols].to_numpy(dtype=float) - self.mu_) / self.sd_
            x = np.column_stack([np.ones(len(X)), m])
        else:
            x = np.ones((len(X), 1))
        parts = [np.einsum("ij,ik->ijk", x, C).reshape(len(X), -1)]
        if self.industries_:
            ind = self._ind_codes(X)
            D = np.zeros((len(X), len(self.industries_)))
            D[np.flatnonzero(ind >= 0), ind[ind >= 0]] = 1.0
            parts.append(D)
        if self.intercept:
            parts.append(np.ones((len(X), 1)))
        return np.column_stack(parts)

    def _normal_equations(self, X, y, w):
        self.full_builds_ = getattr(self, "full_builds_", 0) + 1
        C = X[self.char_cols].to_numpy(dtype=float)
        ind = self._ind_codes(X)
        pc, pk, pd_ = len(self.char_cols), 1 + len(self.macro_cols), len(self.industries_)
        nz = pk * pc
        size = nz + pd_ + int(self.intercept)
        A, b = np.zeros((size, size)), np.zeros(size)
        for x, rows in self._groups(X):
            Cg, wg, yg, ig = C[rows], w[rows], y[rows], ind[rows]
            WC = Cg * wg[:, None]
            A[:nz, :nz] += np.kron(np.outer(x, x), Cg.T @ WC)
            b[:nz] += np.kron(x, WC.T @ yg)
            if pd_:
                has = ig >= 0
                DWC = np.zeros((pd_, pc))
                np.add.at(DWC, ig[has], WC[has])
                A[:nz, nz : nz + pd_] += np.kron(x[:, None], DWC.T)
                np.add.at(A[nz : nz + pd_, nz : nz + pd_], (ig[has], ig[has]), wg[has])
                np.add.at(b[nz : nz + pd_], ig[has], (wg * yg)[has])
            if self.intercept:
                j = size - 1
                A[:nz, j] += np.kron(x, WC.sum(0))
                if pd_:
                    np.add.at(A[nz : nz + pd_, j], ig[has], wg[has])
                A[j, j] += wg.sum()
                b[j] += (wg * yg).sum()
        A = np.triu(A) + np.triu(A, 1).T  # fill the lower triangle from the accumulated upper
        return A, b

    def _solve(self, A, b):
        """Minimum-norm solution of the (possibly singular) normal equations (D031).

        Directions with no variation in the training window (e.g. a characteristic not yet
        available, median-filled to 0) get exactly zero weight instead of arbitrary values.
        """
        vals, vecs = np.linalg.eigh(A)
        keep = vals > EIG_RTOL * vals.max()
        self.rank_ = int(keep.sum())
        return vecs[:, keep] @ ((vecs[:, keep].T @ b) / vals[keep])

    # --- public API -----------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None) -> "LinearGKX":
        m = X[self.macro_cols].to_numpy(dtype=float) if self.macro_cols else np.zeros((len(X), 0))
        self.mu_, self.sd_ = m.mean(0), np.where(m.std(0) > 0, m.std(0), 1.0)
        self.industries_ = (
            sorted(X[self.ind_col].dropna().unique()) if self.ind_col is not None else []
        )
        yy = y.to_numpy(dtype=float)
        self.full_builds_ = 0
        A0, b0 = self._normal_equations(X, yy, np.ones(len(yy)))
        self.coef_ = self._solve(A0, b0)
        self.cond_ = float(np.linalg.cond(A0))
        if self.loss == "l2":
            return self
        if self.xi is None:
            self.xi = huber_threshold(self, X, y)
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            self.n_iter_ = it
            a = np.abs(yy - self.predict(X))
            w = np.where(a <= self.xi, 1.0, self.xi / np.maximum(a, 1e-300))
            if self.fast_irls and not (w < 1.0).any():
                A, b = A0, b0
            elif self.fast_irls:
                out = np.flatnonzero(w < 1.0)
                Z = self._design_rows(X.iloc[out])
                shrink = 1.0 - w[out]
                A = A0 - Z.T @ (Z * shrink[:, None])
                b = b0 - Z.T @ (shrink * yy[out])
            else:
                A, b = self._normal_equations(X, yy, w)
            new = self._solve(A, b)
            done = np.max(np.abs(new - self.coef_)) <= self.tol * (1 + np.max(np.abs(new)))
            self.coef_ = new
            if done:
                break
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        C = X[self.char_cols].to_numpy(dtype=float)
        pc, pk = len(self.char_cols), 1 + len(self.macro_cols)
        nz = pk * pc
        theta = self.coef_[:nz].reshape(pk, pc)
        out = np.empty(len(X))
        for x, rows in self._groups(X):
            out[rows] = C[rows] @ (x @ theta)
        if self.industries_:
            ind = self._ind_codes(X)
            gamma = np.append(self.coef_[nz : nz + len(self.industries_)], 0.0)  # -1 → 0
            out += gamma[ind]
        if self.intercept:
            out += self.coef_[-1]
        return out
