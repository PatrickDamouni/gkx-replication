"""Map GKX's 94 characteristics (datashare) to OSAP signals by measured agreement (D013, D022).

For 24 months spread over 1960-2016, rank every characteristic within the month and compute
the Pearson correlation of ranks (= Spearman) between each GKX characteristic and each OSAP
signal on the same (permno, signal month). Average over months. The best OSAP signal by |ρ|
is the candidate match; the second best is shown so near-ties are visible.

Verdict thresholds (fixed before looking at results): |ρ| >= 0.90 same; 0.60-0.90 close
variant; < 0.60 no OSAP counterpart. Sign is reported because OSAP sometimes flips a signal.

Usage: uv run python scripts/feature_map.py   (writes results/feature_map.csv, docs/feature_map.md)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DS = "data/raw/datashare_pre2017.parquet"
OSAP = "data/raw/osap_firm_char_202510_pre2017.parquet"
DOC = "data/raw/osap_signal_doc_202510.csv"
MONTHS = [int(f"{y}06") for y in range(1960, 2017, 2)][:24] + [201612]
SAME, CLOSE = 0.90, 0.60


def monthly_rank_corr(ds: pd.DataFrame, osap: pd.DataFrame, gkx: list[str], sig: list[str]):
    both = ds.merge(osap, on=["permno", "yyyymm"], suffixes=("", "__osap"))
    mats = []
    for _, m in both.groupby("yyyymm"):
        a = m[gkx].rank().to_numpy()
        b = m[[s + "__osap" if s in gkx else s for s in sig]].rank().to_numpy()
        mats.append(_nan_corr(a, b))
    return np.nanmean(np.stack(mats), axis=0)  # (len(gkx), len(sig))


def _nan_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise-complete Pearson correlation between columns of a and columns of b."""
    out = np.full((a.shape[1], b.shape[1]), np.nan)
    va, vb = ~np.isnan(a), ~np.isnan(b)
    for j in range(b.shape[1]):
        ok = va & vb[:, [j]]
        n = ok.sum(0)
        x = np.where(ok, a, 0.0)
        y = np.where(ok, b[:, [j]], 0.0)
        mx, my = x.sum(0) / np.maximum(n, 1), y.sum(0) / np.maximum(n, 1)
        cov = (x * y).sum(0) / np.maximum(n, 1) - mx * my
        vx = (x * x).sum(0) / np.maximum(n, 1) - mx**2
        vy = (y * y).sum(0) / np.maximum(n, 1) - my**2
        # Near-constant columns (e.g. 0/1 event flags) make the one-pass variance ~0 or slightly
        # negative; treat those as undefined rather than letting cov/0 become ±inf.
        valid = (n >= 100) & (vx > 1e-8 * np.maximum(mx**2, 1)) & (vy > 1e-8 * np.maximum(my**2, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = np.where(valid, np.clip(cov / np.sqrt(vx * vy), -1, 1), np.nan)
    return out


def main() -> None:
    gkx = [c for c in pq.read_schema(DS).names if c not in ("permno", "yyyymm", "sic2")]
    sig = [c for c in pq.read_schema(OSAP).names if c not in ("permno", "yyyymm")]
    month_filter = [("yyyymm", "in", MONTHS)]
    ds = pd.read_parquet(DS, columns=["permno", "yyyymm", *gkx], filters=month_filter)
    osap = pd.read_parquet(OSAP, columns=["permno", "yyyymm", *sig], filters=month_filter)
    rho = monthly_rank_corr(ds, osap, gkx, sig)

    doc = pd.read_csv(DOC).set_index("Acronym")
    rows = []
    for i, g in enumerate(gkx):
        order = np.argsort(-np.nan_to_num(np.abs(rho[i]), nan=-1))
        best, second = order[0], order[1]
        r1, r2 = rho[i, best], rho[i, second]
        verdict = "same" if abs(r1) >= SAME else "close variant" if abs(r1) >= CLOSE else "none"
        rows.append(
            {
                "gkx": g,
                "osap": sig[best] if verdict != "none" else "",
                "rho": round(float(r1), 3),
                "verdict": verdict,
                "osap_description": doc["LongDescription"].get(sig[best], ""),
                "runner_up": f"{sig[second]} ({r2:+.2f})",
            }
        )
    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/feature_map.csv", index=False)
    counts = table["verdict"].value_counts().to_dict()
    Path("docs").mkdir(exist_ok=True)
    Path("docs/feature_map.md").write_text(
        "# GKX characteristic → OSAP signal map (measured)\n\n"
        f"Method: `scripts/feature_map.py`. Mean within-month Spearman ρ over {len(MONTHS)} months "
        f"({MONTHS[0]}..{MONTHS[-1]}), same (permno, signal month). Verdict: |ρ| ≥ {SAME} same, "
        f"≥ {CLOSE} close variant, else none (thresholds fixed before running).\n\n"
        f"**Summary:** {counts}\n\n" + table.to_markdown(index=False) + "\n"
    )
    print(table.to_string(index=False))
    print(counts)


if __name__ == "__main__":
    main()
