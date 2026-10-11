"""Every pre-registered number, computed once after all test-window predictions exist (D027).

Usage: uv run python scripts/final_report.py [--rehearsal]
--rehearsal runs the same analysis on the 1975-1986 rehearsal predictions (dry run of this code).
Refuses unless every model in the frozen list finished from one commit with intact prediction
files; test mode also needs a clean tree whose src/ and scripts/ match that commit. The verdict is
printed mechanically from D045. Output: results/final_report{,_rehearsal}.md and .csv.
"""

import argparse
import hashlib
import json
import subprocess
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
from gkx.report import (
    CRASH_MONTHS,
    demeaned_frame,
    demeaned_r2,
    holm,
    r2_excluding,
    sharpe_ci,
    size_masks,
    verdict,
)
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
BASELINES = {"OLS-3", "OLS-3+H", "OLS", "OLS+H"}  # reported, never trials (D028/D034)
HEADLINE = "NN3"  # D045, fixed ex ante
# D045/D047 confirmatory family, raw arm: (candidate, vs, test). "replicates" = every p <= 5%
# (intersection-union); Holm at 5% says which legs hold; test 1 must hold for "partially".
CONFIRMATORY = [
    ("NN3", "train_mean", "DM"),
    ("NN3", "PCR", "DM"), ("NN3", "PLS", "DM"), ("NN3", "ENet+H", "DM"), ("NN3", "GLM+H", "DM"),
    ("GBRT+H", "ENet+H", "DM"),
    ("ENet+H", "OLS-3+H", "CW"),
]  # fmt: skip
# D045 secondary, never change the verdict: ranking-only DM for the -dm arms; the can't-lose pair
SECONDARY = [
    ("NN3-dm", "ENet+H", "DM-demeaned"), ("GBRT+H-dm", "ENet+H", "DM-demeaned"),
    ("ENet+H", "OLS+H", "DM"),
]  # fmt: skip
PAIRS = CONFIRMATORY + SECONDARY
ALPHA = 0.05


FROZEN_PATHS = ("src", "scripts", "pyproject.toml", "uv.lock")


def provenance_errors(
    runs: dict, run_sha: str, freeze_sha: str, code_changed: str, tree_dirty: bool
) -> list[str]:
    """Test mode (D047): every run from the freeze-v1 commit, clean and on the test window."""
    errors = []
    if not (run_sha and freeze_sha.startswith(run_sha)):  # ledger keeps short shas
        errors.append(f"runs come from {run_sha[:7]}, not freeze-v1 {freeze_sha[:7]}")
    bad = [m for m, (st, _) in runs.items() if st["git_dirty"] or not st["test_window"]]
    if bad:
        errors.append(f"dirty or non-test runs: {bad}")
    if code_changed:
        errors.append(f"frozen code changed since the run: {code_changed!r}")
    if tree_dirty:
        errors.append("uncommitted changes")
    return errors


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


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
    ap.add_argument("--deviation", default="", help="D-entry allowing a post-look analysis fix")
    args = ap.parse_args()
    test = not args.rehearsal
    deviation_note = ""
    tag = "test" if test else "rehearsal"
    runs = _latest_runs(tag)
    missing = [m for m in MODELS if m not in runs]
    if missing:
        raise SystemExit(f"missing finished {tag} runs: {missing}")
    runs = {m: runs[m] for m in MODELS}
    shas = {st["git_sha"] for st, _ in runs.values()}
    if len(shas) != 1:
        raise SystemExit(f"{tag} runs must share one commit; shas={shas}")
    run_sha = shas.pop()
    head = _git("rev-parse", "HEAD")
    if test:
        changed = _git("diff", "--name-only", run_sha, "HEAD", "--", *FROZEN_PATHS)
        errors = provenance_errors(
            runs,
            run_sha,
            freeze_sha=_git("rev-parse", "freeze-v1^{commit}"),
            code_changed="" if args.deviation else changed,
            tree_dirty=bool(_git("status", "--porcelain", "--", ".", ":!results")),
        )
        if errors:
            raise SystemExit("refused: " + "; ".join(errors))
        if args.deviation and changed:  # D047: a logged post-look analysis fix, shown in the header
            deviation_note = f"\n\n**Deviation {args.deviation}:** analysis code changed since the run: {changed.split()}."
    for m, (_, fin) in runs.items():
        path = LEDGER.parent / fin["preds_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != fin["preds_sha256"]:
            raise SystemExit(f"{m}: {path} does not match its logged sha256")

    folds = walk_forward() if test else walk_forward(first_test_year=1975, last_test_year=1986)
    panel = load_panel(PANEL) if test else load_panel(PANEL, end=198612)
    keys = ["yyyymm", "permno"]
    preds = {m: pd.read_parquet(LEDGER.parent / runs[m][1]["preds_path"]) for m in MODELS}
    base = preds[MODELS[0]][[*keys, "target", "me_lag"]].copy()
    seed_rows = []
    for m in MODELS:
        p = preds[m]
        assert len(p) == len(base) and (p[keys].to_numpy() == base[keys].to_numpy()).all(), m
        base[m] = p["pred"].to_numpy()
        seed_cols = [c for c in p.columns if c.startswith("pred_seed")]
        if len(seed_cols) > 1:  # D006b: mean ± spread across seeds, never the best seed
            r2s = [r2_oos(p, c) for c in seed_cols]
            seed_rows.append(
                {
                    "model": m,
                    "seeds": len(r2s),
                    "seed R2 mean %": 100 * np.mean(r2s),
                    "seed R2 sd %": 100 * np.std(r2s, ddof=1),
                    "seed R2 min %": 100 * min(r2s),
                    "seed R2 max %": 100 * max(r2s),
                    "ensemble R2 %": 100 * r2_oos(p, "pred"),
                }
            )
    del preds
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
        _, r2_lo, r2_hi = paired_r2_ci(base, "zero", m)
        rows.append(
            {
                "model": m,
                "R2 %": 100 * r2_oos(base, m),
                "R2 95% CI": f"[{100 * r2_lo:.3f}, {100 * r2_hi:.3f}]",
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

    fallback = []
    for m in MODELS:
        sel = runs[m][1].get("selection", [])
        if sel:
            n = sum(s["chosen"] == "constant" for s in sel)
            fallback.append({"model": m, "constant fallback": f"{n} / {len(sel)} fold-seeds"})

    pair_rows = []
    for a, b, kind in PAIRS:
        if kind == "CW":
            t, p = clark_west(base, small=b, big=a)
        elif kind == "DM-demeaned":
            t, p = diebold_mariano(demeaned_frame(base, [a, b]), base=b, candidate=a)
        else:
            t, p = diebold_mariano(base, base=b, candidate=a)
        d, lo, hi = paired_r2_ci(base, b, a)
        pair_rows.append(
            {
                "candidate": a,
                "vs": b,
                "test": kind,
                "role": "confirmatory" if (a, b, kind) in CONFIRMATORY else "secondary",
                "t": t,
                "p (one-sided)": p,
                "ΔR2 %": 100 * d,
                "95% CI": f"[{100 * lo:.3f}, {100 * hi:.3f}]",
            }
        )
    pairs = pd.DataFrame(pair_rows)
    conf = pairs["role"] == "confirmatory"
    conf_p = pairs.loc[conf, "p (one-sided)"].tolist()
    rejected = holm(conf_p, ALPHA)
    pairs["Holm reject"] = ""
    pairs.loc[conf, "Holm reject"] = ["yes" if r else "no" for r in rejected]
    primary, primary_lo, primary_hi = paired_r2_ci(base, "zero", HEADLINE)
    assert CONFIRMATORY[0] == (HEADLINE, "train_mean", "DM")  # test 1 gates "partially" (D048)
    result = verdict(primary_lo, conf_p, alpha=ALPHA)
    verdict_text = (
        f"**Verdict (D045 as amended by D047, computed mechanically): {result}.** Primary: {HEADLINE} R²_oos "
        f"{100 * primary:.3f}%, 95% CI [{100 * primary_lo:.3f}, {100 * primary_hi:.3f}] "
        f"({'excludes' if primary_lo > 0 else 'does not exclude'} zero). Confirmatory family: "
        f"{sum(p <= ALPHA for p in conf_p)} of {len(conf_p)} with p ≤ {ALPHA} (all needed for "
        f"'replicates'); {sum(rejected)} rejected under Holm (descriptive); test 1 ({HEADLINE} vs train_mean) "
        f"{'rejected' if rejected[0] else 'not rejected'}."
    )

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
    out.write_text(
        f"# Final report ({tag})\n\nPredictions from commit `{run_sha}`; analysis run at `{head}`."
        " All numbers computed by `scripts/final_report.py` after every prediction existed."
        f" Headline model (fixed before the look, D045): **{HEADLINE}**.{deviation_note}\n\n"
        + verdict_text
        + "\n\n## Pre-specified tests (D045/D047; one-sided p; ΔR² with month-block bootstrap CI)\n\n"
        + pairs.round(4).to_markdown(index=False)
        + "\n\n## Per model\n\n"
        + table.round(4).to_markdown(index=False)
        + "\n\n## Seed spread (D006b; the reported model is the 5-seed ensemble)\n\n"
        + pd.DataFrame(seed_rows).round(4).to_markdown(index=False)
        + "\n\n## Constant-fallback selections (D041)\n\n"
        + pd.DataFrame(fallback).to_markdown(index=False)
        + "\n\n## Against GKX Table 1 (monthly R²_oos, %)\n\n"
        + comp.round(3).to_markdown(index=False)
        + f"\n\nDSR: N = {len(candidates)} reported candidate variants (D034), VW decile spreads;"
        " it ignores variants abandoned in rehearsal and treats correlated variants as independent"
        " (D045).\n"
    )
    table.to_csv(f"results/final_report{suffix}.csv", index=False)
    pairs.to_csv(f"results/final_report_tests{suffix}.csv", index=False)
    print(f"wrote {out}\n{verdict_text}")


if __name__ == "__main__":
    main()
