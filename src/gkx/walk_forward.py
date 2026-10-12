"""GKX (2020) §2 sample splitting: expanding train, rolling 12-year validation, annual refit."""

from dataclasses import dataclass

from gkx.panel import HOLDOUT_START, SealedHoldoutError

# Inclusive (first, last) RETURN month as yyyymm; matches align() keying (D014).
Window = tuple[int, int]


@dataclass(frozen=True)
class Fold:
    refit: int
    train: Window
    val: Window
    test: Window

    def __str__(self) -> str:
        def fmt(w: Window) -> str:
            return f"{w[0] // 100}-{w[0] % 100:02d}..{w[1] // 100}-{w[1] % 100:02d}"

        return f"refit {self.refit} | train {fmt(self.train)} | val {fmt(self.val)} | test {fmt(self.test)}"


def walk_forward(
    train_start: int = 195703,
    val_years: int = 12,
    first_test_year: int = 1987,
    last_test_year: int = 2016,
    allow_holdout: bool = False,
) -> list[Fold]:
    """One fold per test year; train ends where validation begins.

    `allow_holdout=True` is passed only by the Phase 5 runner (D050)."""
    if not allow_holdout and last_test_year * 100 + 12 >= HOLDOUT_START:
        raise SealedHoldoutError(
            f"test year {last_test_year} reaches months >= {HOLDOUT_START}, sealed until Phase 5"
        )
    folds = []
    for year in range(first_test_year, last_test_year + 1):
        val_start = year - val_years
        folds.append(
            Fold(
                refit=year,
                train=(train_start, (val_start - 1) * 100 + 12),
                val=(val_start * 100 + 1, (year - 1) * 100 + 12),
                test=(year * 100 + 1, year * 100 + 12),
            )
        )
    return folds
