"""CRSP monthly returns with delisting adjustment, universe filter and lagged ME (D019).

I/O (SQL) is kept separate from the pure transform `build_returns`, which is tested
on hand-built rows. Every query that touches returns filters dates server-side, so
no holdout return ever reaches this machine.

Usage: uv run python -m gkx.crsp   (needs ~/.pgpass from a one-time wrds.Connection())
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.align import next_month
from gkx.panel import HOLDOUT_START

_CUTOFF = f"{HOLDOUT_START // 100}-01-01"
QUERY_MSF = f"select permno, date, ret, prc, shrout from crsp.msf where date < '{_CUTOFF}'"
QUERY_DELIST = (
    f"select permno, dlstdt, dlstcd, dlret from crsp.msedelist where dlstdt < '{_CUTOFF}'"
)
QUERY_NAMES = "select permno, namedt, nameendt, shrcd, exchcd from crsp.msenames"
QUERY_RF = f"select date, rf from ff.factors_monthly where date < '{_CUTOFF}'"

OUT_PATH = Path("data/raw/crsp_returns_pre2017.parquet")
EXCHANGES = (1, 2, 3)  # GKX universe: every security on NYSE/AMEX/NASDAQ (D020)
COMMON_SHARES = (10, 11)  # robustness run only: filter on the shrcd column
SHUMWAY = {1: -0.30, 2: -0.30, 3: -0.55}  # missing performance DLRET by exchange (D019)


def _yyyymm(dates: pd.Series) -> pd.Series:
    return dates.dt.year * 100 + dates.dt.month


def _is_performance_delisting(code: pd.Series) -> pd.Series:
    return (code == 500) | code.between(520, 584)


def build_returns(
    msf: pd.DataFrame, delist: pd.DataFrame, names: pd.DataFrame, rf: pd.DataFrame
) -> pd.DataFrame:
    """Return-month panel: permno, yyyymm, ret, ret_excess, me, me_lag, shrcd, exchcd."""
    df = msf.assign(yyyymm=_yyyymm(msf["date"])).sort_values("date")
    df = pd.merge_asof(
        df, names.sort_values("namedt"), left_on="date", right_on="namedt", by="permno"
    )
    df.loc[df["date"] > df["nameendt"], ["shrcd", "exchcd"]] = np.nan  # no name record covers it

    dl = delist.assign(yyyymm=_yyyymm(delist["dlstdt"]))[["permno", "yyyymm", "dlstcd", "dlret"]]
    df = df.merge(dl, on=["permno", "yyyymm"], how="left", validate="one_to_one")
    impute = df["dlret"].isna() & _is_performance_delisting(df["dlstcd"])
    df.loc[impute, "dlret"] = df.loc[impute, "exchcd"].map(SHUMWAY)
    has_dl = df["dlret"].notna()
    df["ret"] = df["ret"].where(~has_dl, (1 + df["ret"].fillna(0)) * (1 + df["dlret"]) - 1)

    df["me"] = df["prc"].abs() * df["shrout"] / 1000
    lag = df[["permno", "yyyymm", "me"]].assign(yyyymm=lambda d: next_month(d["yyyymm"]))
    df = df.merge(lag.rename(columns={"me": "me_lag"}), on=["permno", "yyyymm"], how="left")

    rf_m = rf.assign(yyyymm=_yyyymm(pd.to_datetime(rf["date"])))[["yyyymm", "rf"]]
    df = df.merge(rf_m, on="yyyymm", how="left", validate="many_to_one")
    df["ret_excess"] = df["ret"] - df["rf"]

    universe = df["exchcd"].isin(EXCHANGES)
    cols = ["permno", "yyyymm", "ret", "ret_excess", "me", "me_lag", "shrcd", "exchcd"]
    return df.loc[universe, cols].sort_values(["yyyymm", "permno"]).reset_index(drop=True)


def fetch(db, start: str, end: str) -> pd.DataFrame:
    """Build returns for dates in [start, end) with the same queries and `build_returns` (D050).

    Used by the Phase 5 scripts only: the 2015-2016 consistency re-pull and the sealed 2017+ pull.
    """
    msf = db.raw_sql(
        f"select permno, date, ret, prc, shrout from crsp.msf "
        f"where date >= '{start}' and date < '{end}'",
        date_cols=["date"],
    )
    delist = db.raw_sql(
        f"select permno, dlstdt, dlstcd, dlret from crsp.msedelist "
        f"where dlstdt >= '{start}' and dlstdt < '{end}'",
        date_cols=["dlstdt"],
    )
    names = db.raw_sql(QUERY_NAMES, date_cols=["namedt", "nameendt"])
    rf = db.raw_sql(
        f"select date, rf from ff.factors_monthly where date >= '{start}' and date < '{end}'",
        date_cols=["date"],
    )
    return build_returns(msf, delist, names, rf)


def main() -> int:
    import wrds

    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME", "patrickdamouni"))
    msf = db.raw_sql(QUERY_MSF, date_cols=["date"])
    delist = db.raw_sql(QUERY_DELIST, date_cols=["dlstdt"])
    names = db.raw_sql(QUERY_NAMES, date_cols=["namedt", "nameendt"])
    rf = db.raw_sql(QUERY_RF, date_cols=["date"])
    db.close()

    out = build_returns(msf, delist, names, rf)
    if out["yyyymm"].max() >= HOLDOUT_START:
        raise RuntimeError("holdout month in CRSP output; the server-side filter failed")
    out.astype({"shrcd": "int8", "exchcd": "int8"}).to_parquet(OUT_PATH, compression="zstd")
    print(
        f"{len(msf):,} msf rows -> {len(out):,} universe rows, {out['permno'].nunique():,} permnos, "
        f"{out['yyyymm'].min()}..{out['yyyymm'].max()}, ret NaN {out['ret'].isna().mean():.2%} -> {OUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
