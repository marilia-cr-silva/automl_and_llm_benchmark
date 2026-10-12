import pytest


@pytest.fixture(autouse=True)
def _unset_collection(monkeypatch):
    """Tests start from the canonical collection whatever the caller's environment says."""
    monkeypatch.delenv("COLLECTION", raising=False)
