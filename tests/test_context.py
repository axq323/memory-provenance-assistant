import pytest
from mpa import context
from mpa.context import assemble


def _mem(i, excluded=()):
    return {"id": f"m{i}", "text": f"memory {i}", "excluded_tasks": list(excluded)}


def test_excluded_memory_is_withheld_with_reason():
    rec = assemble("explain recursion", [_mem(1, ["explain"]), _mem(2)], "explain", "explain")
    assert rec.sent_ids == {"m2"}
    assert rec.memories_withheld == [{"id": "m1", "text": "memory 1",
                                      "reason": "you chose not to use it for explanations"}]
    assert "[m1]" not in " ".join(m["content"] for m in rec.messages)


def test_exclusion_only_applies_to_its_task():
    rec = assemble("summarise this", [_mem(1, ["explain"])], "summarise", "summarise")
    assert rec.sent_ids == {"m1"}


def test_order_puts_task_last_after_memories_and_history():
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    rec = assemble("explain recursion", [_mem(1)], "explain", "explain", history=hist)
    roles = [m["role"] for m in rec.messages]
    assert roles == ["system", "system", "user", "assistant", "user"]
    assert rec.messages[-1]["content"] == "explain recursion"
    assert rec.messages[1]["content"].startswith("STORED MEMORIES")


def test_budgets_cap_memories_and_history():
    mems = [_mem(i) for i in range(context.MAX_MEMORIES + 2)]
    hist = [{"role": "user", "content": str(i)} for i in range(20)]
    rec = assemble("x", mems, "other", None, history=hist)
    assert len(rec.memories_sent) == context.MAX_MEMORIES
    assert sum(w["reason"] == "memory budget reached" for w in rec.memories_withheld) == 2
    assert rec.history_turns_sent == context.MAX_HISTORY_TURNS


def test_no_memory_block_when_none_selected():
    rec = assemble("x", [], "other", None)
    assert len(rec.messages) == 2


def test_redact_also_covers_withheld_memories():
    rec = assemble("explain", [_mem(1, ["explain"])], "explain", "explain")
    rec.redact_memory("m1")
    assert "memory 1" not in str(rec.to_dict())


def test_redact_removes_memory_text_from_record():
    rec = assemble("x", [_mem(1), _mem(2)], "other", None)
    rec.redact_memory("m1")
    everything = str(rec.to_dict())
    assert "memory 1" not in everything
    assert "memory 2" in everything



@pytest.mark.parametrize("version", ["v2", "v4"])
def test_redact_removes_every_line_of_a_multiline_memory(version):
    """Edit uses a text area, so a memory can span lines; Forget must blank all of them."""
    mems = [{"id": "m1", "text": "Bullet points.\nKeep them short.\nPRIVATE DETAIL", "source_title": "Essay chat"},
            {"id": "m2", "text": "Formal tone.\nSecond line kept.", "source_title": "Work email"}]
    rec = assemble("hi", mems, "other", None, prompt_version=version)
    before = rec.messages[1]["content"]
    rec.redact_memory("m1")
    block = rec.messages[1]["content"]
    assert "Keep them short" not in block and "PRIVATE DETAIL" not in block and "Essay chat" not in block
    assert "- [m1] [forgotten]" in block
    assert "Formal tone.\nSecond line kept." in block            # other memories untouched
    assert ("Work email" in block) == (version == "v4")
    assert "PRIVATE DETAIL" not in str(rec.to_dict())
    assert before.splitlines()[0] == block.splitlines()[0] == "STORED MEMORIES:"


def test_redacted_single_line_block_matches_the_old_format(store):
    rec = assemble("hi", store.all(), "other", None, prompt_version="v4")
    original = rec.messages[1]["content"]
    rec.redact_memory("m2")
    expected = "\n".join("- [m2] [forgotten]" if line.startswith("- [m2]") else line
                          for line in original.splitlines())
    assert rec.messages[1]["content"] == expected


# ------------------------------------------------------------ prompt versions ---
def test_v2_is_unchanged_and_shows_no_sources(store):
    from mpa.context import SYSTEM_PROMPTS
    assert "Use a memory only if it fits the current request." in SYSTEM_PROMPTS["v2"]
    rec = assemble("hi", store.all(), "other", None, prompt_version="v2")
    assert rec.messages[0]["content"] == SYSTEM_PROMPTS["v2"]
    assert "saved from" not in rec.messages[1]["content"]


def test_v3_adds_the_rules_without_sources(store):
    rec = assemble("hi", store.all(), "other", None, prompt_version="v3")
    assert "The current request comes first" in rec.messages[0]["content"]
    assert "saved from:" not in rec.messages[1]["content"]
    assert rec.prompt_version == "v3"


def test_v4_shows_each_memory_with_its_source(store):
    rec = assemble("hi", store.all(), "other", None, prompt_version="v4")
    block = rec.messages[1]["content"]
    assert "- [m2] You want a formal, professional tone in the messages you write. " \
           "(saved from: Work email: asking my manager for an extension)" in block
    assert "Each memory shows where it was" in rec.messages[0]["content"]


def test_record_keeps_the_source_and_forgetting_removes_it(store):
    rec = assemble("hi", store.all(), "other", None, prompt_version="v4")
    assert next(m for m in rec.memories_sent if m["id"] == "m2")["source"].startswith("Work email")
    rec.redact_memory("m2")
    m2 = next(m for m in rec.memories_sent if m["id"] == "m2")
    assert m2["text"] == "[forgotten]" and "source" not in m2
    assert "Work email" not in rec.messages[1]["content"]


def test_store_lists_memories_with_source_titles_but_does_not_save_them(store, tmp_path):
    import json
    assert store.all()[0]["source_title"].startswith("Essay revision")
    store.edit("m1", "You prefer bullet points for essay feedback.")
    assert "source_title" not in json.loads((tmp_path / "memories.json").read_text())[0]