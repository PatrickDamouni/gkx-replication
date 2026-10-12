"""The write-up's table is the generated one, never a hand-edited copy (council 13)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_writeup_table_matches_generated_table():
    text = (REPO / "WRITEUP.md").read_text()
    block = text.split("<!-- table:start -->\n", 1)[1].split("<!-- table:end -->", 1)[0]
    assert block == (REPO / "results/writeup/table.md").read_text()


def test_writeup_figures_exist():
    for name in ("fig1_confirmatory.png", "fig2_per_year.png"):
        assert (REPO / "results/writeup" / name).exists(), name
