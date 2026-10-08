import pytest


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    """Tests must never write to the real trial ledger: it is the deflated-Sharpe trial count."""
    monkeypatch.setattr("gkx.evaluate.LEDGER", tmp_path / "trials.jsonl")
