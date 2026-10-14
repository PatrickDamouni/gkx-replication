"""README figure: the four holdout tests' t-statistics, 1987–2016 vs 2017–2021.

Numbers are parsed from results/phase5_report.md (written once by the frozen code at result-v2),
never typed by hand. t-statistics, not R² levels: D054 rules out reading holdout R² levels against
1987–2016 levels.
Usage: uv run --locked --with matplotlib==3.11.2 python exploratory/readme_figure.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "results/phase5_report.md"
OUT = REPO / "docs/figures/holdout_tests.png"
INK, ACCENT, MUTED = "#1f2933", "#c2410c", "#9aa5b1"
ROWS = {  # report row label → figure label
    "H1: NN3-dm vs constant (confirmatory)": "H1 (confirmatory): NN3-dm vs constant",
    "NN3 vs train_mean": "NN3 vs constant",
    "NN3-dm vs ENet+H, within-month errors": "NN3-dm vs ENet+H, within-month",
    "ENet+H vs constant": "ENet+H vs constant",
}


def _table_rows(text: str) -> list[list[str]]:
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("|") and not re.match(r"^\|[:\- |]+\|$", line.strip())
    ]


def stats() -> dict[str, tuple[float, float]]:
    """{report label: (1987–2016 t, 2017–2021 t)} from the frozen holdout report."""
    out: dict[str, tuple[float, float]] = {}
    header: list[str] = []
    for row in _table_rows(REPORT.read_text()):
        if "t" in row and "1987–2016 t" in row:
            header = row
            continue
        if not header or len(row) != len(header):
            continue
        cell = dict(zip(header, row, strict=True))
        if "candidate" in cell:  # the seven-test table; its "test" column is DM/CW
            label = f"{cell['candidate']} vs {cell['vs']}"
        else:
            label = cell["test"]
        if label in ROWS:
            out[label] = (float(cell["1987–2016 t"]), float(cell["t"]))
    missing = set(ROWS) - set(out)
    if missing:
        raise SystemExit(f"rows not found in {REPORT}: {sorted(missing)}")
    return out


def main() -> None:
    data = stats()
    labels = list(ROWS)
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    for i, key in enumerate(labels):
        y = len(labels) - 1 - i
        before, after = data[key]
        ax.barh(y + 0.18, before, height=0.34, color=MUTED)
        ax.barh(y - 0.18, after, height=0.34, color=ACCENT)
        ax.text(
            max(before, 0) + 0.08, y + 0.18, f"{before:.2f}", va="center", fontsize=8, color=INK
        )
        ax.text(max(after, 0) + 0.08, y - 0.18, f"{after:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels)), [ROWS[k] for k in labels][::-1])
    ax.axvline(0, color=INK, lw=0.8)
    ax.axvline(1.645, color=INK, lw=0.8, ls="--")
    ax.text(1.68, len(labels) - 0.42, "one-sided 5% critical value", fontsize=7.5, color=INK)
    ax.set_ylim(-0.6, len(labels) - 0.2)
    ax.set_xlabel(
        "Diebold–Mariano t-statistic (positive = first model has lower error)", fontsize=9
    )
    ax.set_title(
        "The same four tests: 1987–2016 (grey) vs sealed 2017–2021 (orange)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8.5)
    fig.text(
        0.01,
        0.015,
        "H1 was chosen after seeing 1987–2016, so its grey bar is inflated by selection. "
        "Source: results/phase5_report.md (tag result-v2).",
        fontsize=6.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(OUT, {k: data[k] for k in labels})


if __name__ == "__main__":
    main()
