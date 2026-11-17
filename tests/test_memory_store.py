import json

import pytest

from mpa.memory_store import MemoryStore


def test_seed_loads_four_memories(store):
    assert [m["id"] for m in store.all()] == ["m1", "m2", "m3", "m4"]


def test_source_flags_the_turn_that_created_the_memory(store):
    src = store.source_for("m1")
    assert src["title"].startswith("Essay revision")
    flagged = [t for t in src["turns"] if t["creates_this_memory"]]
    assert len(flagged) == 1
    assert "short bullet points" in flagged[0]["text"]


def test_edit_persists_across_reload(store, tmp_path):
    store.edit("m1", "You prefer bullet points for essay feedback.")
    reloaded = MemoryStore(path=tmp_path / "memories.json")
    m = reloaded.get("m1")
    assert m["text"] == "You prefer bullet points for essay feedback."
    assert m["edited"] is not None


def test_edit_rejects_empty_text(store):
    with pytest.raises(ValueError):
        store.edit("m1", "   ")


def test_exclude_and_undo_for_task(store):
    store.exclude_for_task("m1", "explain")
    store.exclude_for_task("m1", "explain")  # idempotent
    assert store.get("m1")["excluded_tasks"] == ["explain"]
    store.include_for_task("m1", "explain")
    assert store.get("m1")["excluded_tasks"] == []


def test_forget_removes_memory_from_disk(store, tmp_path):
    store.forget("m2")
    on_disk = json.loads((tmp_path / "memories.json").read_text())
    assert "m2" not in [m["id"] for m in on_disk]
    assert store.get("m2") is None


def test_reset_restores_seed_without_touching_it(store):
    store.forget("m1")
    store.reset()
    assert store.get("m1") is not None
