"""Write-up table and figures, generated (never hand-copied) from committed results.

Table and figure 1: frozen pre-specified numbers (results/final_report*.csv, single look).
Figure 2: EXPLORATORY (D049), from the hash-checked frozen predictions.
Usage: uv run --locked --with matplotlib==3.11.2 python exploratory/writeup_figures.py
matplotlib is not in uv.lock on purpose: pyproject.toml and uv.lock are frozen since freeze-v1.
"""

import importlib.util
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/writeup"
INK, ACCENT, MUTED, DARK = "#1f2933", "#c2410c", "#9aa5b1", "#52606d"
NAMES = {"train_mean": "constant"}
TABLE_MODELS = [
    "train_mean", "OLS-3+H", "PCR", "PLS", "ENet+H", "GLM+H", "RF", "GBRT+H", "NN1", "NN2", "NN3",
]  # fmt: skip
LABELS = {
    "train_mean": "Constant (training-window mean)",
    "OLS-3+H": "OLS-3+H (size, B/M, momentum)",
    "NN3": "**NN3 (headline, fixed ex ante)**",
}


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _phase4():
    spec = importlib.util.spec_from_file_location(
        "phase4", REPO / "exploratory/phase4_diagnostics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table() -> str:
    """Markdown table from results/final_report.csv and paper/gkx_table1.csv (3 decimals, as stored)."""
    rep = pd.read_csv(REPO / "results/final_report.csv").set_index("model")
    gkx = pd.read_csv(REPO / "paper/gkx_table1.csv").set_index("sample")
    rows = [
        (
            "| Model | R²_oos % [95% CI] | GKX | Top-1000 % (GKX) | Bottom-1000 % (GKX)"
            " | VW Sharpe, annualized [95% CI] |"
        ),
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for m in TABLE_MODELS:
        r = rep.loc[m]

        def g(sample: str, m: str = m) -> str:
            return f"{gkx.loc[sample, m]:.2f}" if m in gkx.columns else "–"

        sharpe = "–" if pd.isna(r["Sharpe VW"]) else f"{r['Sharpe VW']:.2f} {r['VW 95% CI']}"
        rows.append(
            f"| {LABELS.get(m, m)} | {r['R2 %']:.3f} {r['R2 95% CI']} | {g('All')} "
            f"| {r['R2 top1000 %']:.3f} ({g('Top 1000')}) "
            f"| {r['R2 bottom1000 %']:.3f} ({g('Bottom 1000')}) | {sharpe} |"
        )
    return "\n".join(rows) + "\n"


def confirmatory() -> Path:
    """Figure 1: the seven pre-specified tests, from the frozen report's CSV."""
    t = pd.read_csv(REPO / "results/final_report_tests.csv")
    t = t[t["role"] == "confirmatory"].reset_index(drop=True)
    ci = t["95% CI"].str.strip("[]").str.split(",", expand=True).astype(float)
    labels = [
        f"{i + 1}. {a} vs {NAMES.get(b, b)}" + (" ★" if h == "yes" else "")
        for i, (a, b, h) in enumerate(zip(t["candidate"], t["vs"], t["Holm reject"], strict=True))
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    n = len(t)
    for i in range(n):
        y = n - 1 - i
        p = t.loc[i, "p (one-sided)"]
        color = ACCENT if p <= 0.05 else MUTED
        ax.plot([ci.loc[i, 0], ci.loc[i, 1]], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(t.loc[i, "ΔR2 %"], y, "o", color=color, ms=6)
        label = f"p = {p:.3f}" if p >= 1e-3 else "p < 0.001"
        x = max(ci.loc[i, 1], t.loc[i, "ΔR2 %"]) + 0.02
        ax.text(x, y, label, va="center", fontsize=8, color=INK)
    ax.set_yticks(range(n), labels[::-1])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("ΔR² = R²(candidate) − R²(comparator), percentage points", fontsize=9, color=INK)
    ax.set_title(
        "Pre-specified confirmatory tests, 1987–2016 (single look from freeze-v1)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    fig.text(
        0.01,
        0.015,
        "Bars: two-sided 95% month-block bootstrap CI of ΔR². Orange: one-sided p ≤ 0.05"
        " (Diebold–Mariano; test 7 Clark–West, which tests a nested\nadjustment, not ΔR²)."
        " ★ survives Holm across the 7. Source: results/final_report_tests.csv."
        f" Generated at {_head()}.",
        fontsize=6.5,
        color=DARK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = OUT / "fig1_confirmatory.png"
    fig.savefig(path, dpi=200)
    return path


def per_year() -> Path:
    """Figure 2 (exploratory): NN3 minus the constant, R² by test year."""
    p4 = _phase4()
    _, base, _, _, _ = p4.load()
    mo = p4.Months(base["yyyymm"].to_numpy())
    y = base["target"].to_numpy()
    infl, years = p4.influential_months(base, mo, y, mo.sum(y**2))
    d = years["ΔR² % (NN3 − train_mean)"].to_numpy()
    full = float(infl.loc[0, "ΔR² % (NN3 − train_mean)"])
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.bar(years["year"], d, color=[DARK if v > 0 else MUTED for v in d], width=0.75)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(full, color=ACCENT, lw=1.2, ls="--")
    ax.text(
        years["year"].max() + 0.6,
        full,
        f"1987–2016\npooled {full:+.2f}",
        color=ACCENT,
        fontsize=7.5,
        va="center",
    )
    ax.set_ylabel("ΔR² (pp)", fontsize=9, color=INK)
    ax.set_title(
        "EXPLORATORY — NN3 ensemble minus constant forecast, R² by year",
        fontsize=10,
        color=INK,
        loc="left",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    fig.text(
        0.01,
        0.015,
        f"Source: frozen test predictions (sha256-checked); D049. Generated at {_head()}.",
        fontsize=6.5,
        color=DARK,
    )
    fig.tight_layout(rect=(0, 0.05, 0.94, 1))
    path = OUT / "fig2_per_year.png"
    fig.savefig(path, dpi=200)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.md").write_text(table())
    print(OUT / "table.md", confirmatory(), per_year(), f"matplotlib {matplotlib.__version__}")
