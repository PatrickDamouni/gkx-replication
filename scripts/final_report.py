"""Every pre-registered number, computed once after all test-window predictions exist (D027).

Usage: uv run python scripts/final_report.py [--rehearsal]
--rehearsal runs the same analysis on the 1975-1986 rehearsal predictions (dry run of this code).
Test mode refuses unless every model in the frozen list finished on the test window, from one
clean commit. Output: results/final_report{,_rehearsal}.md and .csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gkx.evaluate import LEDGER
from gkx.metrics import (
    clark_west,
    decile_long_short,
    deflated_sharpe,
    diebold_mariano,
    paired_r2_ci,
    r2_oos,
)
from gkx.panel import load_panel
from gkx.report import CRASH_MONTHS, demeaned_r2, r2_excluding, sharpe_ci, size_masks
from gkx.walk_forward import walk_forward

PANEL = "data/processed/panel_datashare.parquet"
MODELS = [
    "OLS-3", "OLS-3+H", "OLS", "OLS+H", "PCR", "PLS", "ENet+H", "GLM+H",
    "GBRT+H", "GBRT+H-dm", "RF", "RF-dm",
    "NN1", "NN1-dm", "NN2", "NN2-dm", "NN3", "NN3-dm",
]  # fmt: skip
GKX_COLUMN = {
    "OLS+H": "OLS+H",
    "OLS-3+H": "OLS-3+H",
    "PLS": "PLS",
    "PCR": "PCR",
    "ENet+H": "ENet+H",
    "GLM+H": "GLM+H",
    "RF": "RF",
    "GBRT+H": "GBRT+H",
    "NN1": "NN1",
    "NN2": "NN2",
    "NN3": "NN3",
}
# PROPOSED pre-specified ranking tests (pending council 8 + user approval before the freeze tag):
# the prereg ranking NN/trees > penalized linear > OLS-all, on the faithful (raw-target) arm,
# repeated for the demeaned arm. All non-nested → Diebold–Mariano (D010).
BASELINES = {"OLS-3", "OLS-3+H", "OLS", "OLS+H"}  # reported, never trials (D028/D034)
HEADLINE = "NN3"
PAIRS = [
    ("NN3", "ENet+H"), ("GBRT+H", "ENet+H"), ("ENet+H", "OLS+H"),
    ("NN3-dm", "ENet+H"), ("GBRT+H-dm", "ENet+H"),
]  # fmt: skip


def _latest_runs(tag: str) -> dict[str, tuple[dict, dict]]:
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    started = {r["run_id"]: r for r in rows if r["event"] == "started"}
    out = {}
    for r in rows:
        if r["event"] == "finished" and r["run_id"] in started:
            st = started[r["run_id"]]
            if st["name"].endswith(f"-{tag}"):
                out[st["name"].removesuffix(f"-{tag}")] = (st, r)
    return out


def _training_mean(preds: pd.DataFrame, panel: pd.DataFrame, folds) -> pd.Series:
    """D036 benchmark: each fold's training-window mean of the raw target."""
    bench = pd.Series(np.nan, index=preds.index)
    for f in folds:
        tr = panel[panel["yyyymm"].between(*f.train) & panel["target"].notna()]
        rows = preds["yyyymm"].between(*f.test)
        bench[rows] = tr["target"].mean()
    return bench


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true")
    test = not ap.parse_args().rehearsal
    tag = "test" if test else "rehearsal"
    runs = _latest_runs(tag)
    missing = [m for m in MODELS if m not in runs]
    if missing:
        raise SystemExit(f"missing finished {tag} runs: {missing}")
    if test:
        shas = {st["git_sha"] for st, _ in runs.values()}
        dirty = [m for m, (st, _) in runs.items() if st["git_dirty"] or not st["test_window"]]
        if len(shas) != 1 or dirty:
            raise SystemExit(f"test runs must share one clean commit; shas={shas} dirty={dirty}")

    folds = walk_forward() if test else walk_forward(first_test_year=1975, last_test_year=1986)
    panel = load_panel(PANEL) if test else load_panel(PANEL, end=198612)
    keys = ["yyyymm", "permno"]
    preds = {m: pd.read_parquet(LEDGER.parent / runs[m][1]["preds_path"]) for m in MODELS}
    base = preds[MODELS[0]][[*keys, "target", "me_lag"]].copy()
    for m in MODELS:
        p = preds[m]
        assert len(p) == len(base) and (p[keys].to_numpy() == base[keys].to_numpy()).all(), m
        base[m] = p["pred"].to_numpy()
    base["zero"] = 0.0
    base["train_mean"] = _training_mean(base, panel, folds).to_numpy()
    top, bottom = size_masks(base)

    rows, sr_monthly = [], {}
    for m in [*MODELS, "train_mean"]:
        ls = decile_long_short(
            base.rename(columns={m: "_p"})[[*keys, "target", "me_lag", "_p"]], "_p"
        )
        ew, vw = (sharpe_ci(ls[c]) if len(ls) > 1 else (np.nan,) * 3 for c in ("ew", "vw"))
        cw_t, cw_p = clark_west(base, small="zero", big=m)
        sr_monthly[m] = ls["vw"].mean() / ls["vw"].std(ddof=1) if len(ls) > 1 else np.nan
        rows.append(
            {
                "model": m,
                "R2 %": 100 * r2_oos(base, m),
                "R2 top1000 %": 100 * r2_oos(base[top], m),
                "R2 bottom1000 %": 100 * r2_oos(base[bottom], m),
                "demeaned R2 %": 100 * demeaned_r2(base, m),
                "R2 ex-crash %": 100 * r2_excluding(base, m, CRASH_MONTHS),
                "CW vs zero t": cw_t,
                "CW p": cw_p,
                "Sharpe EW": ew[0],
                "EW 95% CI": f"[{ew[1]:.2f}, {ew[2]:.2f}]",
                "Sharpe VW": vw[0],
                "VW 95% CI": f"[{vw[1]:.2f}, {vw[2]:.2f}]",
            }
        )
    table = pd.DataFrame(rows)

    candidates = [m for m in MODELS if m not in BASELINES]  # D034, from the frozen list
    if test:  # the ledger's flags must agree with the frozen list
        flagged = [m for m in MODELS if runs[m][0]["counts_as_trial"]]
        if flagged != candidates:
            raise SystemExit(f"counts_as_trial flags {flagged} != frozen candidates {candidates}")
    srs = np.array([sr_monthly[m] for m in candidates])
    sr_var = float(np.nanvar(srs, ddof=1))
    dsr = {}
    for m in candidates:
        ls = decile_long_short(
            base.rename(columns={m: "_p"})[[*keys, "target", "me_lag", "_p"]], "_p"
        )
        v = ls["vw"]
        dsr[m] = deflated_sharpe(
            sr=float(sr_monthly[m]),
            n_trials=len(candidates),
            sr_var=sr_var,
            n_obs=len(v),
            skew=float(v.skew()),
            kurt=float(v.kurt() + 3),
        )
    table["DSR (VW)"] = table["model"].map(dsr)

    pair_rows = []
    for a, b in PAIRS:
        t, p = diebold_mariano(base, base=b, candidate=a)
        d, lo, hi = paired_r2_ci(base, b, a)
        pair_rows.append(
            {
                "candidate": a,
                "vs": b,
                "DM t": t,
                "DM p (one-sided)": p,
                "ΔR2 %": 100 * d,
                "95% CI": f"[{100 * lo:.3f}, {100 * hi:.3f}]",
            }
        )
    pairs = pd.DataFrame(pair_rows)

    gkx = pd.read_csv("paper/gkx_table1.csv").set_index("sample")
    comp = pd.DataFrame(
        [
            {
                "model": m,
                "ours All": table.set_index("model").loc[m, "R2 %"],
                "GKX All": gkx.loc["All", col],
                "ours Top1000": table.set_index("model").loc[m, "R2 top1000 %"],
                "GKX Top1000": gkx.loc["Top 1000", col],
                "ours Bottom1000": table.set_index("model").loc[m, "R2 bottom1000 %"],
                "GKX Bottom1000": gkx.loc["Bottom 1000", col],
            }
            for m, col in GKX_COLUMN.items()
        ]
    )

    suffix = "" if test else "_rehearsal"
    out = Path(f"results/final_report{suffix}.md")
    sha = next(iter(runs.values()))[0]["git_sha"]
    out.write_text(
        f"# Final report ({tag})\n\nCommit `{sha}`. All numbers computed by `scripts/final_report.py`"
        f" after every prediction existed. Headline model (fixed before the look): **{HEADLINE}**.\n\n"
        "## Per model\n\n"
        + table.round(4).to_markdown(index=False)
        + "\n\n## Pre-specified ranking tests (Diebold–Mariano, paired ΔR² block bootstrap)\n\n"
        + pairs.round(4).to_markdown(index=False)
        + "\n\n## Against GKX Table 1 (monthly R²_oos, %)\n\n"
        + comp.round(3).to_markdown(index=False)
        + f"\n\nDSR: N = {len(candidates)} reported candidate variants (D034), VW decile spreads.\n"
    )
    table.to_csv(f"results/final_report{suffix}.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
