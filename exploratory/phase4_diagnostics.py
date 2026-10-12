"""Phase 4 step 1: the zero-refit diagnostics listed in D049, from the frozen test predictions.

EXPLORATORY: the 1987-2016 window was seen at result-v1. Nothing here changes the D048 verdict.
Usage: uv run --locked python exploratory/phase4_diagnostics.py
Lives outside scripts/ and src/ on purpose: those paths are frozen since freeze-v1 (final_report.py).
Output: results/exploratory/phase4_diagnostics.md
"""

import hashlib
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.evaluate import LEDGER
from gkx.metrics import hac_mean_test, r2_oos

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/exploratory/phase4_diagnostics.md"
CRSP = REPO / "data/raw/crsp_returns_pre2017.parquet"
LAST_MONTH = 201612  # never load anything later (D009 seal)
BANNER = (
    "> **EXPLORATORY (D049).** The 1987–2016 test window was seen at `result-v1`. Nothing below is"
    " a confirmatory test or changes the D048 verdict.\n\n"
)
NN = ["NN1", "NN1-dm", "NN2", "NN2-dm", "NN3", "NN3-dm"]
WHERE = ["train_mean", "ENet+H", "GBRT+H", "RF", "NN3", "NN3-dm"]


def exploratory(markdown: str) -> str:
    """Every Phase 4 output starts with the banner (tested)."""
    return BANNER + markdown


def _final_report():
    """Deliberate: reuse the frozen report's own loaders (_latest_runs, _training_mean), not a copy."""
    spec = importlib.util.spec_from_file_location("final_report", REPO / "scripts/final_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _dm_p(dsse_month: np.ndarray) -> float:
    """One-sided HAC p for E[monthly loss differential] > 0 (D010 machinery)."""
    return hac_mean_test(pd.Series(dsse_month))[1]


def load():
    rep = _final_report()
    runs = rep._latest_runs("test")
    keys = ["yyyymm", "permno"]
    base, seeds, shas = None, {}, {}
    for m in rep.MODELS:
        _, fin = runs[m]
        path = LEDGER.parent / fin["preds_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fin["preds_sha256"], m
        shas[m] = fin["preds_sha256"][:12]
        p = pd.read_parquet(path)
        assert p["yyyymm"].max() <= LAST_MONTH
        if base is None:
            base = p[[*keys, "target", "me_lag"]].copy()
        assert (p[keys].to_numpy() == base[keys].to_numpy()).all(), m
        base[m] = p["pred"].to_numpy()
        cols = sorted(c for c in p.columns if c.startswith("pred_seed"))
        if m in NN:
            seeds[m] = p[cols].to_numpy()
            assert np.allclose(seeds[m].mean(axis=1), p["pred"]), m  # ensemble = frozen forecast
    panel = pd.read_parquet(rep.PANEL, columns=["yyyymm", "target"])
    panel = panel[panel["yyyymm"] <= LAST_MONTH]
    base["train_mean"] = rep._training_mean(base, panel, rep.walk_forward()).to_numpy()
    fallback = {
        m: sorted(
            {s["refit"] for s in runs[m][1].get("selection", []) if s["chosen"] == "constant"}
        )
        for m in rep.MODELS
    }
    return rep, base, seeds, fallback, shas


class Months:
    """Fast per-month sums over a fixed row order."""

    def __init__(self, yyyymm: np.ndarray):
        self.labels, self.idx = np.unique(yyyymm, return_inverse=True)
        self.n = np.bincount(self.idx)

    def sum(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.idx, weights=x, minlength=len(self.labels))


def seed_fragility(base, seeds, mo, y, sst_m):
    sse_tm = mo.sum((y - base["train_mean"].to_numpy()) ** 2)
    rows, nn3 = [], []
    for m, mat in seeds.items():
        k_all = mat.shape[1]
        for k in range(1, k_all + 1):
            r2s, ps = [], []
            for sub in itertools.combinations(range(k_all), k):
                pred = mat[:, list(sub)].mean(axis=1)
                sse = mo.sum((y - pred) ** 2)
                r2s.append(1 - sse.sum() / sst_m.sum())
                ps.append(_dm_p(sse_tm - sse))
                if m == "NN3" and k in (1, k_all - 1):
                    nn3.append(
                        {
                            "seed columns": "+".join(map(str, sub)),
                            "R2 %": 100 * r2s[-1],
                            "DM p vs train_mean": ps[-1],
                        }
                    )
            rows.append(
                {
                    "model": m,
                    "seeds in ensemble": k,
                    "subsets": len(r2s),
                    "R2 % min": 100 * min(r2s),
                    "R2 % median": 100 * float(np.median(r2s)),
                    "R2 % max": 100 * max(r2s),
                    "DM p vs train_mean, median": float(np.median(ps)),
                    "DM p vs train_mean, max": max(ps),
                    "share p ≤ 0.05": float(np.mean(np.array(ps) <= 0.05)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(nn3)


def level_vs_ranking(base, models, mo, y, sst):
    rows = []
    for m in models:
        e = y - base[m].to_numpy()
        between = (mo.sum(e) ** 2 / mo.n).sum()
        within = (e**2).sum() - between
        p = base[m].to_numpy()
        slope = np.cov(y, p)[0, 1] / p.var(ddof=1) if p.var() > 0 else np.nan
        rows.append(
            {
                "model": m,
                "R2 %": 100 * (1 - (between + within) / sst),
                "between-month SSE % of Σr²": 100 * between / sst,
                "within-month (cross-sectional) SSE % of Σr²": 100 * within / sst,
                "Mincer–Zarnowitz slope": slope,
            }
        )
    df = pd.DataFrame(rows).set_index("model")
    gain = df.loc["train_mean"] - df.loc["NN3"]
    note = (
        f"NN3's gain over train_mean ({gain['R2 %'] * -1:+.3f} pp of R²) splits into "
        f"{gain['between-month SSE % of Σr²']:+.3f} pp from month levels and "
        f"{gain['within-month (cross-sectional) SSE % of Σr²']:+.3f} pp from within-month "
        "cross-sectional fit (ordering and forecast scale together; not a pure ranking measure)."
    )
    return df.reset_index(), note


def nyse_quintiles(df: pd.DataFrame) -> np.ndarray:
    q = np.full(len(df), -1)
    for _, g in df.groupby("yyyymm"):
        nyse = g.loc[g["exchcd"] == 1, "me_lag"].dropna()
        if len(nyse) < 5:
            continue
        cuts = np.quantile(nyse, [0.2, 0.4, 0.6, 0.8])
        ok = g["me_lag"].notna().to_numpy()
        q[g.index[ok]] = np.searchsorted(cuts, g.loc[ok, "me_lag"].to_numpy(), side="right")
    return q


def where_r2_lives(base):
    crsp = pd.read_parquet(CRSP, columns=["permno", "yyyymm", "ret_excess", "shrcd", "exchcd"])
    crsp = crsp[crsp["yyyymm"] <= LAST_MONTH]
    last = crsp.groupby("permno")["yyyymm"].max().rename("last_month")
    df = base.merge(crsp, on=["permno", "yyyymm"], how="left", validate="one_to_one")
    df = df.merge(last, left_on="permno", right_index=True, how="left")
    matched = np.isclose(df["target"], df["ret_excess"], atol=1e-9).mean()
    df["size_q"] = nyse_quintiles(df.reset_index(drop=True))
    masks = {"all": np.ones(len(df), bool)}
    for q, name in enumerate(["NYSE size Q1 (small)", "Q2", "Q3", "Q4", "Q5 (large)"]):
        masks[name] = (df["size_q"] == q).to_numpy()
    masks["common shares (shrcd 10/11)"] = df["shrcd"].isin([10, 11]).to_numpy()
    delist = (df["yyyymm"] == df["last_month"]) & (df["last_month"] < LAST_MONTH)
    masks[
        "excluding each permno's last CRSP month (look-ahead; delisting/merger proxy)"
    ] = ~delist.to_numpy()
    masks["each permno's last CRSP month only (delisting/merger proxy)"] = delist.to_numpy()
    rows = []
    for name, mask in masks.items():
        sub = df[mask]
        rows.append(
            {
                "subset": name,
                "rows": int(mask.sum()),
                "share of Σr² %": 100 * (sub["target"] ** 2).sum() / (df["target"] ** 2).sum(),
            }
            | {f"{m} R2 %": 100 * r2_oos(sub, m) for m in WHERE}
        )
    unassigned = int((df["size_q"] < 0).sum())
    return pd.DataFrame(rows), matched, unassigned


def influential_months(base, mo, y, sst_m):
    sse_tm = mo.sum((y - base["train_mean"].to_numpy()) ** 2)
    sse_nn = mo.sum((y - base["NN3"].to_numpy()) ** 2)
    d = sse_tm - sse_nn
    rows = []
    for side, order in (("NN3's best", np.argsort(-d)), ("train_mean's best", np.argsort(d))):
        for k in (0, 1, 3, 5) if side == "NN3's best" else (1, 3, 5):
            keep = np.ones(len(d), bool)
            keep[order[:k]] = False
            rows.append(
                {
                    "months dropped (post hoc)": f"{k} × {side}" if k else "none",
                    "dropped": ", ".join(str(mo.labels[i]) for i in order[:k]),
                    "ΔR² % (NN3 − train_mean)": 100 * d[keep].sum() / sst_m[keep].sum(),
                    "DM p after dropping": _dm_p(d[keep]),
                }
            )
    years = mo.labels // 100
    per_year = pd.DataFrame({"year": np.unique(years)}).assign(
        **{
            "ΔR² % (NN3 − train_mean)": [
                100 * d[years == yr].sum() / sst_m[years == yr].sum() for yr in np.unique(years)
            ]
        }
    )
    return pd.DataFrame(rows), per_year


def gbrt_fallbacks(base, fallback, mo, y, sst_m):
    years = mo.labels // 100
    fb = set(fallback["GBRT+H"])
    keep = np.array([yr not in fb for yr in years])
    sse_g = mo.sum((y - base["GBRT+H"].to_numpy()) ** 2)
    sse_e = mo.sum((y - base["ENet+H"].to_numpy()) ** 2)
    d = sse_e - sse_g
    return (
        f"GBRT+H constant-fallback test years: {sorted(fb)}. On the other {int(keep.sum())} months: "
        f"ΔR² (GBRT+H − ENet+H) = {100 * d[keep].sum() / sst_m[keep].sum():+.3f}%, one-sided DM p = "
        f"{_dm_p(d[keep]):.4f}; on the fallback years: {100 * d[~keep].sum() / sst_m[~keep].sum():+.3f}%."
    )


def ols_concentration(base, mo, y):
    e2 = (y - base["OLS+H"].to_numpy()) ** 2
    month = np.sort(mo.sum(e2))[::-1]
    rows = np.sort(e2)[::-1]
    n = max(1, int(len(rows) * 1e-4))
    return (
        f"OLS+H squared error: worst 1 month {100 * month[:1].sum() / month.sum():.1f}%, worst 5 "
        f"{100 * month[:5].sum() / month.sum():.1f}%, worst 10 {100 * month[:10].sum() / month.sum():.1f}% "
        f"of the total; worst 0.01% of rows ({n}) {100 * rows[:n].sum() / rows.sum():.1f}%."
    )


def dm_weighting(rep, base, mo, y):
    rows = []
    for a, b, kind in rep.CONFIRMATORY:
        if kind != "DM":
            continue
        d = (y - base[b].to_numpy()) ** 2 - (y - base[a].to_numpy()) ** 2
        rows.append(
            {
                "candidate": a,
                "vs": b,
                "p, months summed (D010, frozen)": _dm_p(mo.sum(d)),
                "p, months equal-weighted": _dm_p(mo.sum(d) / mo.n),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    rep, base, seeds, fallback, shas = load()
    mo = Months(base["yyyymm"].to_numpy())
    y = base["target"].to_numpy()
    sst_m = mo.sum(y**2)
    sst = sst_m.sum()
    md = lambda df: df.round(4).to_markdown(index=False)

    seed_table, nn3_seeds = seed_fragility(base, seeds, mo, y, sst_m)
    lvr, lvr_note = level_vs_ranking(base, [*rep.MODELS, "train_mean"], mo, y, sst)
    where, matched, unassigned = where_r2_lives(base)
    infl, per_year = influential_months(base, mo, y, sst_m)
    dmw = dm_weighting(rep, base, mo, y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        exploratory(
            "# Phase 4 step 1 — zero-refit diagnostics (D049)\n\n"
            f"Inputs: the 18 frozen test prediction files (sha256-checked), months "
            f"{mo.labels.min()}–{mo.labels.max()}. Panel target matches CRSP `ret_excess` on "
            f"{100 * matched:.2f}% of rows. Code: `{_head()}`. Prediction sha256 (first 12): "
            + ", ".join(f"{m} {h}" for m, h in shas.items())
            + ".\n\n"
            "## 1. Seed fragility (test 1 = DM vs train_mean)\n\n"
            + md(seed_table)
            + "\n\nNN3 single seeds and leave-one-seed-out ensembles:\n\n"
            + md(nn3_seeds)
            + "\n\n## 2. Level vs ranking\n\n"
            + md(lvr)
            + f"\n\n{lvr_note}\n"
            + "\n## 3. Where the R² lives\n\n"
            + md(where)
            + f"\n\nNYSE breakpoints use every NYSE row in the panel (all share codes; Fama–French use NYSE"
            f" common stocks only); `exchcd` is from the return month, `me_lag` from the formation month;"
            f" {unassigned} rows have no quintile (missing `me_lag`). The last-CRSP-month rows include"
            " mergers and the 2016 sample end is excluded; conditioning on them uses look-ahead, so"
            " those rows are a diagnostic, never an R² to quote. Price ≥ $5 is deferred: the return"
            " build keeps no price (needs a new WRDS pull). Seed columns are positions in sorted"
            " `pred_seed*` order.\n"
            + "\n## 4. Influential months (NN3 vs train_mean)\n\n"
            + md(infl)
            + "\n\nTrimming one side's best months always moves ΔR² against it; read the rows as how"
            " many months the significance rests on, not as robustness. DM p's here treat the kept"
            " months as contiguous (HAC lag 5).\n"
            + "\n\nPer year:\n\n"
            + md(per_year)
            + "\n\n## 5. GBRT+H\n\n"
            + gbrt_fallbacks(base, fallback, mo, y, sst_m)
            + "\n\n## 6. OLS+H\n\n"
            + ols_concentration(base, mo, y)
            + "\n\n## 7. DM weighting sensitivity\n\n"
            + md(dmw)
            + "\n\nTest 7 (ENet+H vs OLS-3+H) is Clark–West, not DM, so it is not re-weighted here."
            " D008 says DM uses the monthly *mean* loss differential; D010 and `gkx.metrics` use"
            " per-month *sums* (months weighted by stock count). The frozen report used sums.\n"
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
