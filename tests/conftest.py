import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mpa.memory_store import MemoryStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    """A store backed by a temporary working copy of the seed data."""
    return MemoryStore(path=tmp_path / "memories.json")
