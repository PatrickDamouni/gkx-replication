"""GKX's own characteristic data ("datashare", D022), converted to the signal-month convention (D023).

datashare's DATE is the month whose return the row predicts (its readme: DATE=19570329 ->
response is RET at 195703). We store `yyyymm` = DATE month − 1, the month by whose end
the characteristics are known, so every characteristic source shares one convention and
goes through the same `align`.

Usage: uv run python -m gkx.datashare   (ingests data/raw/datashare_download/datashare.zip)
"""

import hashlib
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gkx.panel import split_holdout

ZIP = Path("data/raw/datashare_download/datashare.zip")
FULL = Path("data/raw/datashare_full.parquet")  # transient: deleted after split
PRE = Path("data/raw/datashare_pre2017.parquet")
HOLDOUT = Path("data/sealed/datashare_holdout.parquet")
CHUNK_ROWS = 500_000


def prev_month(yyyymm: pd.Series) -> pd.Series:
    """201601 -> 201512."""
    year, month = yyyymm // 100, yyyymm % 100
    return (year - (month == 1)) * 100 + (month - 2) % 12 + 1


def to_signal_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns="DATE")
    out.insert(1, "yyyymm", prev_month(df["DATE"] // 100))
    return out


def _convert(zip_path: Path, out: Path) -> int:
    rows, writer = 0, None
    with zipfile.ZipFile(zip_path) as zf, zf.open("datashare.csv") as csv:
        for chunk in pd.read_csv(csv, chunksize=CHUNK_ROWS):
            chunk = to_signal_month(chunk)
            ids = chunk[["permno", "yyyymm"]].astype("int32")
            chars = chunk.drop(columns=["permno", "yyyymm", "sic2"]).astype("float32")
            sic2 = chunk["sic2"].astype("Int8")
            table = pa.Table.from_pandas(
                pd.concat([ids, chars, sic2], axis=1), preserve_index=False
            )
            writer = writer or pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
    if writer is not None:
        writer.close()
    return rows


def main() -> int:
    zip_sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    rows = _convert(ZIP, FULL)
    HOLDOUT.parent.chmod(0o755)
    digest = split_holdout(FULL, PRE, HOLDOUT)
    HOLDOUT.chmod(0o444)
    HOLDOUT.parent.chmod(0o555)
    FULL.unlink()
    ZIP.unlink()  # an unsealed copy (D009)
    ZIP.parent.rmdir()
    print(f"zip sha256 {zip_sha}\n{rows:,} rows\npre-2017 -> {PRE}\nholdout sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
