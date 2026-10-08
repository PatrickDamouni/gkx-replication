"""Synthetic panels with a planted signal, for known-answer tests (D008).

target = beta · x + noise, keyed on return month like `align` output (D014).
With mean-zero target, the population GKX R² is beta² / (beta² + sigma²).
"""

import numpy as np
import pandas as pd


def months(start: int = 195703, end: int = 201612) -> np.ndarray:
    """Every yyyymm from start to end inclusive."""
    first, last = start // 100 * 12 + start % 100 - 1, end // 100 * 12 + end % 100 - 1
    m = np.arange(first, last + 1)
    return (m // 12) * 100 + m % 12 + 1


def make_panel(seed: int, firms: int, beta: float, sigma: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yyyymm = months()
    n = len(yyyymm) * firms
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "permno": np.tile(np.arange(firms), len(yyyymm)),
            "yyyymm": np.repeat(yyyymm, firms),
            "x": x,
            "target": beta * x + sigma * rng.normal(size=n),
        }
    )
