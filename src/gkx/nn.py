"""NN1–NN3 (GKX §2.7, Table A.5; D029, D040, D043). PyTorch, CPU, deterministic.

Input is GKX's z = [x ⊗ c, sic2 dummies] with x = (1, raw macro) (D040), computed by a factored
first layer so the 920-column matrix is never built (3.3M × 920 float32 would be 12 GB):
    h = Σ_j x_j (c W_j) + W_ind[sic2] + b.
Hidden layers 32 / 32-16 / 32-16-8, ReLU, batch norm; Adam; L1 penalty on weights; batch 10,000;
≤100 epochs with early stopping (patience 5) on validation MSE, keeping the best epoch's weights.
Each seed selects its own (learning rate, L1) on validation; the harness averages seeds (D006).
If no configuration beats the training-mean forecast on validation, predict that constant (D043).

Never import this module in a process that has used LightGBM: the two OpenMP runtimes deadlock.
"""

import copy
from itertools import pairwise

import numpy as np
import pandas as pd
import torch
from torch import nn

NUM_THREADS = 4  # council 6: 8 threads no faster than 4 at this size
CHUNK = 200_000


class FactoredInput(nn.Module):
    def __init__(self, n_chars: int, n_macro: int, n_industries: int, width: int):
        super().__init__()
        self.w_z = nn.Parameter(torch.randn(1 + n_macro, n_chars, width) / np.sqrt(n_chars))
        self.w_ind = nn.Parameter(torch.zeros(n_industries, width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, C: torch.Tensor, M: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.ones(len(C), 1, dtype=C.dtype), M], dim=1)
        per_macro = torch.einsum("bk,jkh->bjh", C, self.w_z)  # (B, 1 + macro, width)
        h = torch.einsum("bj,bjh->bh", x, per_macro) + self.bias
        padded = torch.cat([self.w_ind, torch.zeros(1, self.w_ind.shape[1])], dim=0)
        return h + padded[torch.where(ind >= 0, ind, len(self.w_ind))]


class _Net(nn.Module):
    def __init__(self, n_chars, n_macro, n_industries, hidden):
        super().__init__()
        self.inp = FactoredInput(n_chars, n_macro, n_industries, hidden[0])
        layers: list[nn.Module] = [nn.BatchNorm1d(hidden[0]), nn.ReLU()]
        for a, b in pairwise(hidden):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        layers.append(nn.Linear(hidden[-1], 1))
        self.body = nn.Sequential(*layers)

    def forward(self, C, M, ind):
        return self.body(self.inp(C, M, ind)).squeeze(1)

    def l1(self) -> torch.Tensor:
        weights = [self.inp.w_z, self.inp.w_ind] + [
            m.weight for m in self.body if isinstance(m, nn.Linear)
        ]
        return sum(w.abs().sum() for w in weights)


class NN:
    stochastic = True  # the harness requires >= 5 seeds (D006)

    def __init__(
        self,
        char_cols: list[str],
        macro_cols: list[str],
        ind_col: str | None = "sic2",
        hidden: tuple[int, ...] = (32,),
        seed: int = 0,
        lrs: tuple[float, ...] = (0.001, 0.01),
        l1s: tuple[float, ...] = (1e-5, 1e-4, 1e-3),
        epochs: int = 100,
        patience: int = 5,
        batch: int = 10_000,
    ):
        self.char_cols, self.macro_cols, self.ind_col = char_cols, macro_cols, ind_col
        self.hidden, self.seed, self.lrs, self.l1s = hidden, seed, lrs, l1s
        self.epochs, self.patience, self.batch = epochs, patience, batch

    def _tensors(self, X: pd.DataFrame):
        C = torch.tensor(X[self.char_cols].to_numpy(dtype=np.float32))
        M = torch.tensor(X[self.macro_cols].to_numpy(dtype=np.float32))
        if self.ind_col is None:
            ind = torch.full((len(X),), -1)
        else:
            ind = torch.tensor(
                pd.Categorical(X[self.ind_col], categories=self.industries_).codes.astype(np.int64)
            )
        return C, M, ind

    @torch.no_grad()
    def _predict(self, net, C, M, ind) -> np.ndarray:
        net.eval()
        out = [
            net(C[s : s + CHUNK], M[s : s + CHUNK], ind[s : s + CHUNK])
            for s in range(0, len(C), CHUNK)
        ]
        return torch.cat(out).double().numpy()

    def _train(self, lr, l1, C, M, ind, y, Cv, Mv, indv, yv):
        torch.manual_seed(self.seed)
        net = _Net(len(self.char_cols), len(self.macro_cols), len(self.industries_), self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, best_epoch, waited = np.inf, None, 0, 0
        for epoch in range(1, self.epochs + 1):
            net.train()
            order = torch.randperm(len(C), generator=gen)
            for s in range(0, len(C), self.batch):
                idx = order[s : s + self.batch]
                if len(idx) < 2:  # batch norm needs at least two rows
                    continue
                loss = ((net(C[idx], M[idx], ind[idx]) - y[idx]) ** 2).mean() + l1 * net.l1()
                opt.zero_grad()
                loss.backward()
                opt.step()
            mse = float(((self._predict(net, Cv, Mv, indv) - yv) ** 2).mean())
            if mse < best:
                best, best_state, best_epoch, waited = (
                    mse,
                    copy.deepcopy(net.state_dict()),
                    epoch,
                    0,
                )
            else:
                waited += 1
                if waited >= self.patience:
                    break
        net.load_state_dict(best_state)
        return net, best_epoch

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        torch.set_num_threads(NUM_THREADS)
        torch.use_deterministic_algorithms(True)
        self.industries_ = (
            sorted(X[self.ind_col].dropna().unique()) if self.ind_col is not None else []
        )
        C, M, ind = self._tensors(X)
        Cv, Mv, indv = self._tensors(X_val)
        yt = torch.tensor(y.to_numpy(dtype=np.float32))
        yv = y_val.to_numpy(dtype=float)
        self.nets_, self.val_scores_, self.best_epochs_ = {}, {}, {}
        for lr in self.lrs:
            for l1 in self.l1s:
                net, epoch = self._train(lr, l1, C, M, ind, yt, Cv, Mv, indv, yv)
                pv = self._predict(net, Cv, Mv, indv)
                self.nets_[(lr, l1)], self.best_epochs_[(lr, l1)] = net, epoch
                self.val_scores_[(lr, l1)] = float(1 - ((yv - pv) ** 2).sum() / (yv @ yv))
        self.constant_ = float(y.mean())
        self.val_scores_["constant"] = float(1 - ((yv - self.constant_) ** 2).sum() / (yv @ yv))
        self.chosen_ = max(self.val_scores_, key=self.val_scores_.get)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.chosen_ == "constant":
            return np.full(len(X), self.constant_)
        return self._predict(self.nets_[self.chosen_], *self._tensors(X))
