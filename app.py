"""Memory Provenance Assistant — Streamlit interface.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from mpa.context import assemble  # noqa: E402
from mpa.interaction_log import log_event  # noqa: E402
from mpa.llm import ModelError, ReplayModel, get_model  # noqa: E402
from mpa.memory_store import MemoryStore  # noqa: E402
from mpa.task_types import classify, label  # noqa: E402
from mpa.validate import ValidatedReply, validate  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.getenv("MPA_RUNTIME_DIR", ROOT / "runtime"))

st.set_page_config(page_title="Memory Provenance Assistant", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- state ---
@st.cache_resource
def load_model():
    return get_model()


model = load_model()
MODE = "replay" if isinstance(model, ReplayModel) else "live"
store = MemoryStore(path=RUNTIME / "memories.json")

ss = st.session_state
ss.setdefault("messages", [])          # user / assistant turns
ss.setdefault("session_id", uuid.uuid4().hex[:6])
ss.setdefault("open_sources", set())   # keys of source panels currently open
ss.setdefault("pending_prompt", None)


def log(event: str, **kw) -> None:
    log_event(event, session_id=ss.session_id, condition="A", mode=MODE,
              path=RUNTIME / "interaction_log.csv", **kw)


# -------------------------------------------------------------- actions ---
def history_before(index: int) -> list[dict]:
    hist = []
    for m in ss.messages[:index]:
        if m["role"] == "user":
            hist.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            hist.append({"role": "assistant", "content": m["reply"].answer})
    return hist


def respond(user_text: str, history: list[dict]) -> dict:
    task, trigger = classify(user_text)
    record = assemble(user_text, store.all(), task, trigger, history)
    try:
        reply = validate(model.generate(record, user_text), record.sent_ids)
    except ModelError as exc:
        reply = ValidatedReply(answer="Sorry, I couldn't reach the model.", parse_ok=False, note=str(exc))
    log("reply_shown", turn_id=record.turn_id, task=task,
        detail="used=" + ",".join(u["id"] for u in reply.used))
    return {"role": "assistant", "turn_id": record.turn_id, "user_text": user_text,
            "task": task, "reply": reply, "record": record}


def toggle_source(key: str, memory_id: str, turn_id: str = "") -> None:
    if key in ss.open_sources:
        ss.open_sources.discard(key)
        log("source_closed", turn_id=turn_id, memory_id=memory_id)
    else:
        ss.open_sources.add(key)
        log("source_opened", turn_id=turn_id, memory_id=memory_id)


def queue_prompt(text: str) -> None:
    ss.pending_prompt = text


# ------------------------------------------------------------ rendering ---
def render_source(memory_id: str) -> None:
    src = store.source_for(memory_id)
    if src is None:
        st.caption("Source conversation unavailable.")
        return
    st.caption(f"Saved from **{src['title']}** · {src['date']}")
    for t in src["turns"]:
        who = "You" if t["speaker"] == "user" else "Assistant"
        if t["creates_this_memory"]:
            st.info(f"**{who}:**\n\n{t['text']}\n\n↑ This memory was saved from this message.")
        else:
            st.caption(f"**{who}:**\n\n{t['text']}")


def render_memory_chip(msg: dict, used: dict) -> None:
    mem = store.get(used["id"])
    key = f"{msg['turn_id']}:{used['id']}"
    with st.container(border=True):
        if mem is None:
            st.caption("A memory used in this reply has since been forgotten.")
            return
        st.markdown(f"🧠 **{mem['text']}**" + ("  _(edited by you)_" if mem.get("edited") else ""))
        if used["how"]:
            st.caption(f"How it was applied: {used['how']}")
        is_open = key in ss.open_sources
        st.button("Hide source" if is_open else "Show where this came from",
                  key=f"src_{key}", on_click=toggle_source, args=(key, used["id"], msg["turn_id"]))
        if is_open:
            render_source(used["id"])


def render_reply(msg: dict) -> None:
    reply: ValidatedReply = msg["reply"]
    st.markdown(reply.answer)
    if reply.note:
        st.warning(reply.note)
    if reply.used:
        st.markdown("**Memories used in this reply**")
        for used in reply.used:
            render_memory_chip(msg, used)
    elif reply.parse_ok:
        st.caption("No stored memories were used in this reply.")
    if ss.get("researcher_view"):
        render_record(msg)


def render_record(msg: dict) -> None:
    rec = msg["record"]
    reply: ValidatedReply = msg["reply"]
    with st.expander("Researcher view: what was sent to the model"):
        st.markdown(f"- **Task type:** {label(rec.task)} (trigger: `{rec.trigger or 'none'}`)\n"
                    f"- **Prompt version:** {rec.prompt_version} · **Model:** {model.name}\n"
                    f"- **History turns sent:** {rec.history_turns_sent}")
        st.markdown("**Memories sent**")
        st.json(rec.memories_sent, expanded=True)
        if rec.memories_withheld:
            st.markdown("**Memories withheld**")
            st.json(rec.memories_withheld, expanded=True)
        if reply.dropped:
            st.markdown("**Model claims rejected by the validator**")
            st.json(reply.dropped, expanded=True)


def render_memory_panel() -> None:
    st.subheader("What I remember about you")
    memories = store.all()
    if not memories:
        st.caption("Nothing is stored.")
    for mem in memories:
        key = f"panel:{mem['id']}"
        with st.container(border=True):
            st.markdown(mem["text"] + ("  _(edited)_" if mem.get("edited") else ""))
            src = store.source_for(mem["id"])
            if src:
                st.caption(f"From {src['title']} · {src['date']}")
            st.button("Hide source" if key in ss.open_sources else "Show source",
                      key=f"src_{key}", on_click=toggle_source, args=(key, mem["id"]))
            if key in ss.open_sources:
                render_source(mem["id"])


# ---------------------------------------------------------------- page ---
with st.sidebar:
    st.title("🧠 Memory Provenance Assistant")
    st.caption(f"Mode: **{MODE}** · Session `{ss.session_id}`")
    st.toggle("Researcher view", key="researcher_view")
    st.divider()
    render_memory_panel()

st.header("Your assistant")
if MODE == "replay":
    st.info("**Replay mode:** replies are scripted (simulated) for the three demo prompts below. "
            "Which memories are sent still depends on your corrections, so the reply changes as it would live.")
    cols = st.columns(len(model.scenarios))
    for col, scenario in zip(cols, model.scenarios):
        col.button(scenario["prompt"], key=f"demo_{scenario['id']}", on_click=queue_prompt,
                   args=(scenario["prompt"],), use_container_width=True)

for msg in ss.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_reply(msg)

typed = st.chat_input("Ask your assistant anything…")
prompt = typed or ss.pending_prompt
ss.pending_prompt = None
if prompt:
    history = history_before(len(ss.messages))
    ss.messages.append({"role": "user", "text": prompt})
    with st.spinner("Thinking…"):
        ss.messages.append(respond(prompt, history))
    st.rerun()
