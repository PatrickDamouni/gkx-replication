"""Power of candidate Phase 5 tests on a 48-, 60- or 96-month holdout (input to D050).

EXPLORATORY: built from the 1987-2016 monthly loss differentials, seen at result-v1. Each candidate
test's monthly differential series is resampled in 12-month circular blocks to the holdout length,
recentred so its mean is (decay x the 1987-2016 mean), and tested exactly as the frozen report does
(one-sided HAC t, 5 lags). Power = share of draws with p <= 0.05. The 1987-2016 means are the
winner-curse-inflated estimates, so "no decay" is an optimistic upper case.
Usage: uv run --locked python exploratory/phase5_power.py
"""

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/exploratory/phase5_power.md"
LENGTHS = (24, 36, 48, 60, 96)  # Arm A ≈ 2017–2021 (D025, exact end known only at unseal); Arm B 2017–2024
DECAY = (1.0, 0.5, 0.0)
BLOCK, DRAWS, SEED, LAGS = 12, 4000, 20261012, 5


def _phase4():
    spec = importlib.util.spec_from_file_location(
        "phase4", REPO / "exploratory/phase4_diagnostics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hac_t(x: np.ndarray, lags: int = LAGS) -> np.ndarray:
    """Newey–West (Bartlett) t of the mean, row-wise over (draws, T): the same estimator as
    gkx.metrics.hac_mean_test (statsmodels HAC, autocovariances divided by T, no correction)."""
    t_len = x.shape[1]
    e = x - x.mean(axis=1, keepdims=True)
    var = (e**2).mean(axis=1)
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (e[:, k:] * e[:, :-k]).sum(axis=1) / t_len  # /T, as statsmodels HAC
    return x.mean(axis=1) / np.sqrt(var / t_len)


def draws(d: np.ndarray, t_len: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = -(-t_len // BLOCK)
    starts = rng.integers(0, len(d), size=(DRAWS, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)) % len(d)
    return d[idx.reshape(DRAWS, -1)[:, :t_len]]


def candidates(p4) -> dict[str, np.ndarray]:
    _, base, _, _, _ = p4.load()
    mo = p4.Months(base["yyyymm"].to_numpy())
    y = base["target"].to_numpy()

    def raw(a: str, b: str) -> np.ndarray:  # positive = a has lower squared error
        return mo.sum((y - base[b].to_numpy()) ** 2 - (y - base[a].to_numpy()) ** 2)

    def demeaned(a: str, b: str) -> np.ndarray:  # month means removed from target and forecasts
        def dm(v: np.ndarray) -> np.ndarray:
            return v - (mo.sum(v) / mo.n)[mo.idx]

        yd = dm(y)
        return mo.sum((yd - dm(base[b].to_numpy())) ** 2 - (yd - dm(base[a].to_numpy())) ** 2)

    return {
        "test 1: NN3 vs constant (raw DM, frozen family)": raw("NN3", "train_mean"),
        "NN3 vs ENet+H (raw DM, frozen family)": raw("NN3", "ENet+H"),
        "NN3-dm vs constant (raw DM)": raw("NN3-dm", "train_mean"),
        "NN3-dm vs ENet+H (within-month DM, frozen secondary)": demeaned("NN3-dm", "ENet+H"),
        "NN3 vs ENet+H (within-month DM)": demeaned("NN3", "ENet+H"),
    }


def main() -> None:
    p4 = _phase4()
    rng = np.random.default_rng(SEED)
    rows = []
    for name, d in candidates(p4).items():
        t_full = float(hac_t(d[None, :])[0])
        row = {"test": name, "1987–2016 t (T=360)": t_full}
        for t_len in LENGTHS:
            base = draws(d, t_len, rng)
            for decay in DECAY:
                shifted = base - d.mean() + decay * d.mean()
                p = norm.sf(hac_t(shifted))
                row[f"T={t_len}, effect×{decay:g}"] = float(np.mean(p <= 0.05))
        rows.append(row)
    table = pd.DataFrame(rows)
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    OUT.write_text(
        p4.exploratory(
            "# Phase 5 power (input to D050)\n\n"
            f"Code `{head}`. {DRAWS} block-bootstrap draws per cell (block {BLOCK} months, seed {SEED}),"
            f" one-sided HAC t with {LAGS} lags, power = share p ≤ 0.05. effect×1 assumes the"
            " 1987–2016 mean differential persists unchanged (optimistic: winner's curse);"
            " effect×0 is the false-positive rate under no effect.\n\n"
            + table.round(3).to_markdown(index=False)
            + "\n"
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
