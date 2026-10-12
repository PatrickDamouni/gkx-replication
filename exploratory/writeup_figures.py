"""The two write-up figures. Figure 1 plots frozen numbers only; figure 2 is EXPLORATORY (D049).

Usage: uv run --locked --with matplotlib python exploratory/writeup_figures.py
matplotlib is not in uv.lock on purpose: pyproject.toml and uv.lock are frozen since freeze-v1.
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/exploratory"
INK, ACCENT, MUTED = "#1f2933", "#c2410c", "#9aa5b1"


def _phase4():
    spec = importlib.util.spec_from_file_location(
        "phase4", REPO / "exploratory/phase4_diagnostics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def confirmatory() -> Path:
    """Figure 1: the pre-specified tests, straight from the frozen report's CSV."""
    t = pd.read_csv(REPO / "results/final_report_tests.csv")
    t = t[t["role"] == "confirmatory"].iloc[::-1]
    ci = t["95% CI"].str.strip("[]").str.split(",", expand=True).astype(float)
    labels = [f"{a} vs {b}" for a, b in zip(t["candidate"], t["vs"], strict=True)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(t))
    for i, (d, lo, hi, p) in enumerate(
        zip(t["ΔR2 %"], ci[0], ci[1], t["p (one-sided)"], strict=True)
    ):
        color = ACCENT if p <= 0.05 else MUTED
        ax.plot([lo, hi], [i, i], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(d, i, "o", color=color, ms=6)
        ax.text(max(hi, d) + 0.02, i, f"p = {p:.3f}" if p >= 1e-3 else "p < 0.001",
                va="center", fontsize=8, color=INK)  # fmt: skip
    ax.set_yticks(y, labels)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("ΔR² (percentage points), 95% month-block bootstrap CI", fontsize=9, color=INK)
    ax.set_title(
        "Pre-specified confirmatory tests, 1987–2016 (single look, freeze-v1)",
        fontsize=10, color=INK, loc="left",
    )  # fmt: skip
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    fig.text(0.01, 0.01, "Orange: one-sided p ≤ 0.05. Test 7 is Clark–West (nested); its ΔR² is the raw R² gap.",
             fontsize=7, color=MUTED)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = OUT / "fig1_confirmatory.png"
    fig.savefig(path, dpi=200)
    return path


def per_year() -> Path:
    """Figure 2 (exploratory): NN3 minus the training-mean forecast, R² by test year."""
    p4 = _phase4()
    _, base, _, _, _ = p4.load()
    mo = p4.Months(base["yyyymm"].to_numpy())
    y = base["target"].to_numpy()
    _, table = p4.influential_months(base, mo, y, mo.sum(y**2))
    d = table["ΔR² % (NN3 − train_mean)"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.bar(table["year"], d, color=[ACCENT if v > 0 else MUTED for v in d], width=0.75)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("ΔR² (pp)", fontsize=9, color=INK)
    ax.set_title(
        "EXPLORATORY — NN3 ensemble minus training-mean forecast, by year",
        fontsize=10, color=INK, loc="left",
    )  # fmt: skip
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    fig.tight_layout()
    path = OUT / "fig2_per_year.png"
    fig.savefig(path, dpi=200)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(confirmatory(), per_year(), f"matplotlib {matplotlib.__version__}", sep="\n")
