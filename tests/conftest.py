import pytest


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    """Tests must never write to the real trial ledger or predictions store (D028)."""
    monkeypatch.setattr("gkx.evaluate.LEDGER", tmp_path / "trials.jsonl")
