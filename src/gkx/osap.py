"""Download Open Source Asset Pricing firm characteristics to Parquet.

Streams the release zip to disk and converts it chunk by chunk, so the
uncompressed CSV (10 GB+) never touches disk or memory in full. The three
CRSP-derived signals (Price, Size, STreversal) need WRDS and are added later
from our own CRSP pull.

Usage: uv run python -m gkx.osap
"""

import hashlib
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from openassetpricing import OpenAP

RELEASE = 202510
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / f"osap_firm_char_{RELEASE}.zip"
PARQUET_PATH = RAW_DIR / f"osap_firm_char_{RELEASE}.parquet"
DOC_PATH = RAW_DIR / f"osap_signal_doc_{RELEASE}.csv"
CHUNK_ROWS = 500_000


def download_zip(url: str, dest: Path) -> str:
    """Stream url to dest; return its sha256."""
    sha = hashlib.sha256()
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
                f.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)
    return sha.hexdigest()


def zip_to_parquet(zip_path: Path, out: Path) -> int:
    """Convert the single CSV inside zip_path to Parquet; return row count."""
    rows = 0
    writer: pq.ParquetWriter | None = None
    with zipfile.ZipFile(zip_path) as zf, zf.open(zf.filelist[0]) as csv:
        for chunk in pd.read_csv(csv, chunksize=CHUNK_ROWS, low_memory=False):
            ids = chunk[["permno", "yyyymm"]].astype("int32")
            signals = chunk.drop(columns=["permno", "yyyymm"]).astype("float32")
            table = pa.Table.from_pandas(pd.concat([ids, signals], axis=1), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            print(f"\r{rows:,} rows", end="", flush=True)
    print()
    if writer is not None:
        writer.close()
    return rows


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ap = OpenAP(RELEASE)

    ap.dl_signal_doc("pandas").to_csv(DOC_PATH, index=False)
    print(f"signal doc -> {DOC_PATH}")

    if not ZIP_PATH.exists():
        sha = download_zip(ap._get_url("firm_char"), ZIP_PATH)
        print(f"sha256 {sha}")

    rows = zip_to_parquet(ZIP_PATH, PARQUET_PATH)
    print(f"{rows:,} rows -> {PARQUET_PATH} ({PARQUET_PATH.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
