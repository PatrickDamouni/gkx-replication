"""Seam 3: the GKX (2020) §2 sample-splitting schedule (PREREGISTRATION.md §Splits).

GKX: training starts at 18 years (1957-1974) and grows by one year per refit;
validation is 12 years and rolls forward; each refit is tested on the next year.
"""

from itertools import pairwise
from pathlib import Path

import pytest

from gkx.panel import SealedHoldoutError
from gkx.walk_forward import Fold, walk_forward

SNAPSHOT = Path(__file__).parent / "snapshots" / "walk_forward.txt"


def test_first_and_last_folds_match_the_paper():
    folds = walk_forward()
    assert len(folds) == 30
    assert folds[0] == Fold(
        refit=1987, train=(195703, 197412), val=(197501, 198612), test=(198701, 198712)
    )
    assert folds[-1] == Fold(
        refit=2016, train=(195703, 200312), val=(200401, 201512), test=(201601, 201612)
    )


def test_train_expands_validation_rolls_and_windows_never_overlap():
    folds = walk_forward()
    for prev, cur in pairwise(folds):
        assert cur.train[0] == prev.train[0]  # expanding
        assert cur.val[0] == prev.val[0] + 100  # rolls one year
        assert cur.test[0] == prev.test[1] - 11 + 100  # tests are back-to-back years
    for f in folds:
        assert f.train[1] < f.val[0] and f.val[1] < f.test[0]


def test_schedule_reaching_into_the_holdout_raises():
    with pytest.raises(SealedHoldoutError):
        walk_forward(last_test_year=2017)


def test_schedule_matches_committed_snapshot():
    """Eyeball-checkable in a diff; changing the schedule must show up in review."""
    assert "\n".join(str(f) for f in walk_forward()) + "\n" == SNAPSHOT.read_text()
