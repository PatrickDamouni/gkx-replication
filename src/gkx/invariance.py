"""Future-invariance: the one leakage check reused across the pipeline (D008).

Change the future two ways and require that outputs up to `cutoff` are
bit-identical:
  1. scramble every value in rows after `cutoff` (catches steps that read later values);
  2. delete every row after `cutoff` (catches steps that depend on which later rows
     exist, e.g. survivorship filters).
No R² threshold is involved, so even a tiny leak fails.

It cannot catch a model reading its own row's label; that is prevented
structurally by passing models only feature columns (D014).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

KEYS = ["permno", "yyyymm"]


class FutureLeak(AssertionError):
    """A pipeline step's output for early months changed when only later months changed."""


def _scramble(panel: pd.DataFrame, future: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Replace future values with noise of the same dtype, so dtypes never change."""
    out = panel.copy()
    n = int(future.sum())
    for col in out.columns.difference(KEYS):
        dtype = out[col].dtype
        if dtype.kind == "f":
            noise = (rng.normal(size=n) * 1e3).astype(dtype)
        elif dtype.kind in "iu":
            noise = rng.integers(0, 10_000, size=n).astype(dtype)
        else:
            noise = rng.permutation(out.loc[future, col].to_numpy())
        out.loc[future, col] = noise
    return out


def _first_difference(a: pd.DataFrame, b: pd.DataFrame) -> str | None:
    a_keys, b_keys = a.set_index(KEYS).index, b.set_index(KEYS).index
    if not a_keys.equals(b_keys):
        only = a_keys.symmetric_difference(b_keys)[0]
        return f"row set differs: permno {only[0]}, yyyymm {only[1]} is present in only one run"
    for col in a.columns.difference(KEYS):
        if a[col].dtype != b[col].dtype:
            return f"column '{col}' dtype changed: {a[col].dtype} -> {b[col].dtype}"
        same = (a[col] == b[col]) | (a[col].isna() & b[col].isna())
        if not same.all():
            i = int(np.argmin(same.to_numpy()))
            return (
                f"column '{col}' differs first at permno {a['permno'].iloc[i]}, yyyymm {a['yyyymm'].iloc[i]}: "
                f"{a[col].iloc[i]!r} -> {b[col].iloc[i]!r}"
            )
    return None


def assert_future_invariant(
    step: Callable[[pd.DataFrame], pd.DataFrame],
    panel: pd.DataFrame,
    cutoff: int,
    seed: int = 0,
) -> None:
    future = panel["yyyymm"] > cutoff

    def early(out: pd.DataFrame) -> pd.DataFrame:
        return out[out["yyyymm"] <= cutoff].sort_values(KEYS).reset_index(drop=True)

    baseline = early(step(panel))
    for how, changed in (
        ("scrambled", _scramble(panel, future, np.random.default_rng(seed))),
        ("deleted", panel[~future]),
    ):
        diff = _first_difference(baseline, early(step(changed)))
        if diff:
            raise FutureLeak(
                f"output for months <= {cutoff} changed when later rows were {how}: {diff}"
            )
