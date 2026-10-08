"""Panel loading with the post-publication holdout sealed at read time.

PREREGISTRATION.md §Splits: months >= 2017-01 are not loaded into any model or
plot until Phase 5. The seal lives here, in the one loader, so a notebook can't
walk past it by calling a lower-level function (D008).
"""

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

HOLDOUT_START = 201701  # yyyymm; first sealed month


class SealedHoldoutError(Exception):
    """Raised when code asks for a month inside the sealed holdout."""


def load_panel(
    path: Path, end: int = HOLDOUT_START - 1, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read a (permno, yyyymm, ...) Parquet panel up to and including `end`.

    Rows from the holdout are filtered out by the Parquet reader, so they never
    reach memory.
    """
    if end >= HOLDOUT_START:
        raise SealedHoldoutError(
            f"requested months through {end}; months >= {HOLDOUT_START} are sealed "
            "until Phase 5 (PREREGISTRATION.md §Splits)"
        )
    return pd.read_parquet(path, columns=columns, filters=[("yyyymm", "<=", end)])


def split_holdout(src: Path, pre_out: Path, holdout_out: Path) -> str:
    """Split a full panel at ingest into pre-holdout and holdout files.

    Streams record batches, so neither half is ever held in memory as a frame.
    Returns the holdout file's sha256, to be recorded in DECISIONS.md: a
    reviewer can check a hash, not a promise.
    """
    dataset = ds.dataset(src, format="parquet")
    month = ds.field("yyyymm")
    written = 0
    for out, cond in (
        (pre_out, month < HOLDOUT_START),
        (holdout_out, month >= HOLDOUT_START),
    ):
        with pq.ParquetWriter(out, dataset.schema, compression="zstd") as writer:
            for batch in dataset.to_batches(filter=cond):
                writer.write_batch(batch)
                written += batch.num_rows
    total = dataset.count_rows()
    if written != total:
        raise ValueError(
            f"split wrote {written} of {total} rows; {total - written} have no usable yyyymm"
        )
    return hashlib.sha256(holdout_out.read_bytes()).hexdigest()
