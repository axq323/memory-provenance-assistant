from mpa import context
from mpa.context import assemble


def _mem(i, excluded=()):
    return {"id": f"m{i}", "text": f"memory {i}", "excluded_tasks": list(excluded)}


def test_excluded_memory_is_withheld_with_reason():
    rec = assemble("explain recursion", [_mem(1, ["explain"]), _mem(2)], "explain", "explain")
    assert rec.sent_ids == {"m2"}
    assert rec.memories_withheld == [{"id": "m1", "reason": "you chose not to use it for explanations"}]
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


def test_redact_removes_memory_text_from_record():
    rec = assemble("x", [_mem(1), _mem(2)], "other", None)
    rec.redact_memory("m1")
    everything = str(rec.to_dict())
    assert "memory 1" not in everything
    assert "memory 2" in everything
