import os

import pytest

# LightGBM and torch deadlock in one process (council 6 Performance, reproduced 4/4). NN tests run
# in their own pytest process: `GKX_TORCH=1 uv run pytest tests/nn`; the default run skips them.
collect_ignore = [] if os.environ.get("GKX_TORCH") else ["nn"]


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    """Tests must never write to the real trial ledger or predictions store (D028)."""
    monkeypatch.setattr("gkx.evaluate.LEDGER", tmp_path / "trials.jsonl")
