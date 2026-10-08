"""D024: the reviewed characteristic map is structurally valid, and applying it is correct."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from gkx.feature_map import TRANSFORMS, apply_osap, load_map

DATASHARE = "data/raw/datashare_pre2017.parquet"
DOC = "data/raw/osap_signal_doc_202510.csv"


@pytest.fixture(scope="module")
def fmap() -> pd.DataFrame:
    return load_map()


def test_every_gkx_characteristic_appears_exactly_once(fmap):
    gkx = [c for c in pq.read_schema(DATASHARE).names if c not in ("permno", "yyyymm", "sic2")]
    assert len(gkx) == 94
    assert sorted(fmap["gkx"]) == sorted(gkx)


def test_osap_signals_are_published_predictors(fmap):
    doc = pd.read_csv(DOC)
    predictors = set(doc.loc[doc["Cat.Signal"] == "Predictor", "Acronym"])
    used = set(fmap.loc[fmap["source"] == "osap", "signal"])
    assert used <= predictors, used - predictors


def test_each_osap_column_is_used_once_except_declared_transforms_of_a_mapped_base(fmap):
    osap = fmap[fmap["source"] == "osap"]
    identity = osap[osap["transform"].isin(["identity", "negate"])]
    assert identity["signal"].is_unique
    derived = osap[~osap["transform"].isin(["identity", "negate"])]
    assert set(derived["signal"]) <= set(identity["signal"])  # abs/square only of a mapped base


def test_fields_are_from_closed_sets(fmap):
    assert set(fmap["source"]) <= {"crsp", "osap", "none"}
    assert set(fmap.loc[fmap["source"] != "none", "transform"]) <= set(TRANSFORMS)
    assert set(fmap["shift"]) <= {0, 4}
    assert (fmap.loc[fmap["shift"] == 4, "source"] == "osap").all()
    assert fmap.loc[fmap["source"] == "none", "signal"].isna().all()
    assert fmap["concept_note"].str.len().gt(10).all()


def test_apply_restores_gkx_sign_and_applies_transforms():
    """OSAP's file stores sign × raw; mapping multiplies by the documented sign again."""
    osap = pd.DataFrame({"permno": [1, 2], "yyyymm": [200001, 200001], "X": [-2.0, 3.0]})
    fmap = pd.DataFrame(
        {
            "gkx": ["x", "x_abs", "x_sq"],
            "source": "osap",
            "signal": "X",
            "transform": ["identity", "abs", "square"],
            "shift": 0,
        }
    )
    out = apply_osap(osap, fmap, signs={"X": -1}).set_index("permno")
    assert out.loc[1, "x"] == 2.0 and out.loc[2, "x"] == -3.0
    assert out.loc[2, "x_abs"] == 3.0 and out.loc[2, "x_sq"] == 9.0


def test_quarterly_shift_uses_the_osap_value_from_four_months_earlier():
    osap = pd.DataFrame(
        {"permno": 1, "yyyymm": [200001, 200002, 200003, 200004, 200005], "Q": [1.0, 2, 3, 4, 5]}
    )
    fmap = pd.DataFrame(
        {"gkx": ["q"], "source": "osap", "signal": "Q", "transform": "identity", "shift": 4}
    )
    out = apply_osap(osap, fmap, signs={"Q": 1}).set_index("yyyymm")
    assert out.loc[200005, "q"] == 1.0  # value dated 200001
    assert np.isnan(out.loc[200004, "q"])
