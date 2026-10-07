"""Future-invariance: the one leakage check reused across the pipeline (D008).

Scramble every row after `cutoff`, rerun the step, and require that outputs up
to `cutoff` are bit-identical. Any step that peeks at later months fails, no
matter how small the leak is, with no R² threshold involved.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

KEYS = ["permno", "yyyymm"]


class FutureLeak(AssertionError):
    """A pipeline step's output for early months changed when only later months changed."""


def assert_future_invariant(
    step: Callable[[pd.DataFrame], pd.DataFrame],
    panel: pd.DataFrame,
    cutoff: int,
    keep_rows: Callable[[pd.DataFrame], pd.Series] | None = None,
    seed: int = 0,
) -> None:
    keep_rows = keep_rows or (lambda o: o["yyyymm"] <= cutoff)
    rng = np.random.default_rng(seed)

    scrambled = panel.copy()
    future = scrambled["yyyymm"] > cutoff
    for col in scrambled.columns.difference(KEYS):
        scrambled.loc[future, col] = rng.normal(size=future.sum()) * 1e3

    def early(out: pd.DataFrame) -> pd.DataFrame:
        return out[keep_rows(out)].sort_values(KEYS).reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(early(step(panel)), early(step(scrambled)), check_exact=True)
    except AssertionError as err:
        raise FutureLeak(f"output for months <= {cutoff} depends on later months:\n{err}") from err
