"""D050 §6: the diff from freeze-v1 is a whitelist, enforced here (council 14)."""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# May change, only by additive keyword arguments whose defaults keep the seal.
ADDITIVE = {"src/gkx/panel.py", "src/gkx/walk_forward.py", "src/gkx/align.py", "src/gkx/crsp.py"}
# New files Phase 5 may add under the frozen paths.
NEW = {
    "src/gkx/holdout.py",
    "scripts/pull_crsp_holdout.py",
    "scripts/build_panel_holdout.py",
    "scripts/run_holdout.py",
    "scripts/run_holdout.sh",
    "scripts/phase5_report.py",
}
BYTE_IDENTICAL = [
    *(
        f"src/gkx/{m}.py"
        for m in (
            "linear", "reduction", "enet", "glm", "trees", "nn", "evaluate", "metrics",
            "report", "preprocess", "build", "macro", "datashare",
        )
    ),
    "scripts/run_test.py", "scripts/final_report.py", "scripts/run_test.sh",
    "pyproject.toml", "uv.lock",
]  # fmt: skip


def _diff() -> list[tuple[str, str]]:
    tag = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", "freeze-v1"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if tag.returncode != 0:
        pytest.skip("no freeze-v1 tag")
    out = subprocess.run(
        ["git", "diff", "--name-status", "freeze-v1", "--", "src", "scripts", "pyproject.toml", "uv.lock"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [tuple(line.split("\t", 1)) for line in out.splitlines()]


def test_frozen_files_are_byte_identical_to_freeze_v1():
    changed = {path for _, path in _diff()}
    assert not changed & set(BYTE_IDENTICAL), changed & set(BYTE_IDENTICAL)


def test_every_other_change_is_whitelisted():
    for status, path in _diff():
        if status == "M":
            assert path in ADDITIVE, f"{path} modified but not whitelisted (D050 §6)"
        elif status == "A":
            assert path in NEW, f"{path} added but not whitelisted (D050 §6)"
        else:
            raise AssertionError(f"{status} {path}: deletions and renames are not allowed")
