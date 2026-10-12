"""Phase 5 (D050): seal-keeping defaults and the opt-in holdout paths."""

import pandas as pd
import pytest

from gkx.align import align
from gkx.panel import SealedHoldoutError
from gkx.walk_forward import walk_forward


def test_holdout_folds_only_when_explicitly_allowed():
    """D050: the Phase 5 runner opts in; the default still seals."""
    with pytest.raises(SealedHoldoutError):
        walk_forward(first_test_year=2016, last_test_year=2018)
    folds = walk_forward(first_test_year=2016, last_test_year=2018, allow_holdout=True)
    assert [f.refit for f in folds] == [2016, 2017, 2018]
    assert folds[1].train == (195703, 200412) and folds[1].val == (200501, 201612)
    assert folds[0] == walk_forward()[-1]  # the 2016 canary fold is the frozen 2016 fold


def test_align_keeps_the_seal_by_default_and_opens_only_through_a_given_month():
    sig = pd.DataFrame(
        {"permno": [1, 1, 1], "yyyymm": [201611, 201612, 201701], "x": [1.0, 2.0, 3.0]}
    )
    ret = pd.DataFrame(
        {"permno": [1, 1, 1], "yyyymm": [201612, 201701, 201702], "ret_excess": [0.1, 0.2, 0.3]}
    )
    assert align(sig, ret)["yyyymm"].tolist() == [201612]
    opened = align(sig, ret, through=201702)
    assert opened["yyyymm"].tolist() == [201612, 201701, 201702]
    assert opened["target"].tolist() == [0.1, 0.2, 0.3]
