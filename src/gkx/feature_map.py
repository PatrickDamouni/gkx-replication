"""The reviewed GKX → OSAP/CRSP characteristic map (D024) and its application.

`config/feature_map.csv` is the source of truth: hand-reviewed, version-controlled, and
checked by tests. Nothing here selects matches from data; measurement only validates them
(scripts/measure_feature_map.py).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

MAP_PATH = Path("config/feature_map.csv")
DOC_PATH = Path("data/raw/osap_signal_doc_202510.csv")
TRANSFORMS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "identity": lambda x: x,
    "abs": np.abs,
    "square": np.square,
    "negate": lambda x: -x,  # datashare stores some characteristics negated (D026)
}


def load_map(path: Path = MAP_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"signal": "string"}).astype({"shift": int})


def osap_signs(doc: Path = DOC_PATH) -> dict[str, int]:
    """OSAP's `Sign` column: the published file stores sign × raw so higher = higher return."""
    d = pd.read_csv(doc)
    return d.set_index("Acronym")["Sign"].dropna().astype(int).to_dict()


def _add_months(yyyymm: pd.Series, k: int) -> pd.Series:
    m = yyyymm // 100 * 12 + yyyymm % 100 - 1 + k
    return m // 12 * 100 + m % 12 + 1


def apply_osap(osap: pd.DataFrame, fmap: pd.DataFrame, signs: dict[str, int]) -> pd.DataFrame:
    """OSAP panel (permno, yyyymm, signals...) → GKX-named characteristics at the signal month.

    Undoes OSAP's sign, applies the declared transform, and for shift k takes the OSAP
    value dated k months earlier (datashare lags quarterly data 4 months; D024).
    """
    rows = fmap[fmap["source"] == "osap"]
    keys = osap[["permno", "yyyymm"]].drop_duplicates()
    out = keys.copy()
    for _, r in rows.iterrows():
        col = osap[["permno", "yyyymm", r["signal"]]].copy()
        col["yyyymm"] = _add_months(col["yyyymm"], int(r["shift"]))
        col[r["gkx"]] = TRANSFORMS[r["transform"]](signs[r["signal"]] * col.pop(r["signal"]))
        out = out.merge(col, on=["permno", "yyyymm"], how="left")
    return out
