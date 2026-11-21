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
from mpa.counterfactual import check_influence, summarise  # noqa: E402
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
ss.setdefault("flash", [])             # short confirmations shown as toasts
ss.setdefault("influence_check", True)  # show the counterfactual check (turn off for study sessions)
# Study condition: A = memories with source context, B = memories only.
# A researcher can fix it per participant with ?condition=B in the URL.
ss.setdefault("condition", "B" if st.query_params.get("condition", "A").upper() == "B" else "A")


def log(event: str, **kw) -> None:
    log_event(event, session_id=ss.session_id, condition=ss.condition, mode=MODE,
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


def not_for_task(msg_index: int, memory_id: str) -> None:
    """Scoped correction: keep the memory, stop using it for this task type, redo the reply."""
    msg = ss.messages[msg_index]
    task = msg["task"]
    mem = store.get(memory_id)
    if mem is None:
        return
    store.exclude_for_task(memory_id, task)
    log("not_for_task", turn_id=msg["turn_id"], task=task, memory_id=memory_id)
    ss.messages.append({"role": "notice", "kind": "excluded", "memory_id": memory_id, "task": task,
                        "undone": False,
                        "text": f"Got it. I won't use “{mem['text']}” for {label(task)}. "
                                "Here's that reply again without it."})
    redo = respond(msg["user_text"], history_before(msg_index - 1))
    redo["regenerated_from"] = msg["turn_id"]
    ss.messages.append(redo)


def undo_exclusion(notice_index: int) -> None:
    notice = ss.messages[notice_index]
    if store.get(notice["memory_id"]) is None:
        return
    store.include_for_task(notice["memory_id"], notice["task"])
    notice["undone"] = True
    log("undo_not_for_task", task=notice["task"], memory_id=notice["memory_id"])
    ss.flash.append(f"Undone: I'll use that memory for {label(notice['task'])} again.")


def use_again(memory_id: str, task: str) -> None:
    store.include_for_task(memory_id, task)
    log("undo_not_for_task", task=task, memory_id=memory_id, detail="from memory panel")
    ss.flash.append(f"I'll use that memory for {label(task)} again.")


def save_edit(memory_id: str, text_key: str, turn_id: str = "") -> None:
    try:
        store.edit(memory_id, ss[text_key])
    except ValueError as exc:
        ss.flash.append(str(exc))
        return
    log("memory_edited", turn_id=turn_id, memory_id=memory_id)
    ss.flash.append("Memory updated. Future replies will use the new wording.")


def forget(memory_id: str, turn_id: str = "") -> None:
    store.forget(memory_id)
    for m in ss.messages:                      # remove derived copies too (Lecture 10)
        if m["role"] == "assistant":
            m["record"].redact_memory(memory_id)
            for r in m.get("influence") or []:
                if r.id == memory_id:
                    r.text, r.evidence = "[forgotten]", "[forgotten]"
    ss.open_sources = {k for k in ss.open_sources if not k.endswith(memory_id)}
    log("memory_forgotten", turn_id=turn_id, memory_id=memory_id)
    ss.flash.append("Forgotten. It's gone from memory and from the records of past replies.")


def set_verdict(msg_index: int, verdict: str) -> None:
    msg = ss.messages[msg_index]
    msg["verdict"] = verdict
    log("reply_accepted" if verdict == "accepted" else "reply_flagged",
        turn_id=msg["turn_id"], task=msg["task"],
        detail="used=" + ",".join(u["id"] for u in msg["reply"].used))


def reset_demo() -> None:
    store.reset()
    ss.messages = []
    ss.open_sources = set()
    log("demo_reset")
    ss.flash.append("Demo reset: memories restored and chat cleared.")


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


def render_edit_forget(cols, mem: dict, key: str, turn_id: str = "") -> None:
    with cols[0].popover("Edit"):
        st.text_area("Memory wording", value=mem["text"], key=f"edit_text_{key}")
        if MODE == "replay":
            st.caption("Replay mode: replies are scripted, so new wording won't change their text.")
        st.button("Save", key=f"edit_save_{key}", on_click=save_edit,
                  args=(mem["id"], f"edit_text_{key}", turn_id))
    with cols[1].popover("Forget"):
        st.caption("Deletes this memory and removes it from the records of past replies. "
                   "This can't be undone.")
        st.button("Forget this memory", key=f"forget_{key}", type="primary",
                  on_click=forget, args=(mem["id"], turn_id))


def render_memory_chip(msg_index: int, msg: dict, used: dict) -> None:
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
        cols = st.columns([2.2, 2.2, 1, 1]) if SHOW_SOURCES else st.columns([2.2, 1, 1])
        if SHOW_SOURCES:
            cols[0].button("Hide source" if is_open else "Show where this came from",
                           key=f"src_{key}", on_click=toggle_source,
                           args=(key, used["id"], msg["turn_id"]))
            cols = cols[1:]
        if msg["task"] in mem["excluded_tasks"]:
            cols[0].caption(f"Not used for {label(msg['task'])} any more")
        else:
            cols[0].button(f"Not for {label(msg['task'])}", key=f"nft_{key}",
                           on_click=not_for_task, args=(msg_index, used["id"]),
                           help="Keep this memory, but stop using it for this kind of request")
        render_edit_forget(cols[1:], mem, key, msg["turn_id"])
        if SHOW_SOURCES and is_open:
            render_source(used["id"])


VERDICT_LABELS = {
    "unreported influence": "⚠️ Changed the reply, but the assistant didn't list it",
    "claimed, no measurable influence": "⚠️ Listed by the assistant, but made no measurable difference",
    "unclear": "❔ Couldn't tell",
}


def run_influence_check(msg_index: int) -> dict:
    """Counterfactual check: regenerate without each memory and compare (mpa/counterfactual.py)."""
    msg = ss.messages[msg_index]
    rec = msg["record"]
    reported = {u["id"] for u in msg["reply"].used}
    results = check_influence(model, msg["user_text"], store.all(), rec.task, rec.trigger,
                              reported, history_before(msg_index - 1))
    msg["influence"] = results
    counts = summarise(results)
    log("influence_checked", turn_id=msg["turn_id"], task=rec.task,
        detail=";".join(f"{r.id}={r.verdict}" for r in results))
    return counts


def render_influence(msg_index: int, msg: dict) -> None:
    results = msg.get("influence")
    if results is None:
        calls = 1 + 2 * len(msg["record"].memories_sent) if MODE == "live" else 0
        help_text = ("Removes each memory in turn, regenerates the reply and compares. "
                     + (f"Makes about {calls} model calls, so it can take a few minutes."
                        if MODE == "live" else "Uses the scripted replies in replay mode."))
        if st.button("🔍 Check which memories mattered", key=f"infl_{msg['turn_id']}", help=help_text):
            with st.spinner("Regenerating the reply without each memory…"):
                try:
                    run_influence_check(msg_index)
                except ModelError as exc:
                    st.error(f"The check couldn't run: {exc}")
                    return
            st.rerun()
        return
    with st.container(border=True):
        st.markdown("**Which memories mattered (measured, not self-reported)**")
        if not results:
            st.caption("No memories were sent with this request, so there was nothing to test.")
            return
        for r in results:
            said = "listed it" if r.reported else "didn't list it"
            changed = {True: "changed the reply", False: "made no difference", None: "unclear"}[r.influenced]
            if r.verdict == "agrees":
                label_text = ("✅ Used, as the assistant said" if r.influenced
                              else "✅ Not used, as the assistant said")
            else:
                label_text = VERDICT_LABELS[r.verdict]
            st.markdown(f"🧠 {r.text}  \n{label_text}")
            st.caption(f"Assistant {said} · removing it {changed} · {r.evidence}")
        mismatches = [r for r in results if r.verdict in ("unreported influence",
                                                          "claimed, no measurable influence")]
        if mismatches:
            st.warning(f"{len(mismatches)} memor{'y' if len(mismatches) == 1 else 'ies'} "
                       "behaved differently from what the assistant reported.")


def render_reply(msg_index: int, msg: dict) -> None:
    reply: ValidatedReply = msg["reply"]
    st.markdown(reply.answer)
    if reply.note:
        st.warning(reply.note)
    if reply.used:
        st.markdown("**Memories used in this reply**")
        for used in reply.used:
            render_memory_chip(msg_index, msg, used)
    elif reply.parse_ok:
        st.caption("No stored memories were used in this reply.")
    verdict = msg.get("verdict")
    if verdict:
        st.caption("You marked this reply as working for you." if verdict == "accepted"
                   else "You flagged this reply.")
    else:
        c1, c2, _ = st.columns([1.3, 1.3, 3])
        c1.button("👍 This works for me", key=f"ok_{msg['turn_id']}",
                  on_click=set_verdict, args=(msg_index, "accepted"))
        c2.button("👎 Something's off", key=f"flag_{msg['turn_id']}",
                  on_click=set_verdict, args=(msg_index, "flagged"))
    if ss.get("influence_check"):
        render_influence(msg_index, msg)
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
            if SHOW_SOURCES and src:
                st.caption(f"From {src['title']} · {src['date']}")
            else:
                st.caption(f"Saved {mem['created']}")
            for task in mem["excluded_tasks"]:
                st.caption(f"⛔ Not used for {label(task)}")
                st.button(f"Use for {label(task)} again", key=f"again_{mem['id']}_{task}",
                          on_click=use_again, args=(mem["id"], task), use_container_width=True)
            if SHOW_SOURCES:
                st.button("Hide source" if key in ss.open_sources else "Show source",
                          key=f"src_{key}", on_click=toggle_source, args=(key, mem["id"]),
                          use_container_width=True)
            render_edit_forget(st.columns(2), mem, key)
            if SHOW_SOURCES and key in ss.open_sources:
                render_source(mem["id"])


# ---------------------------------------------------------------- page ---
SHOW_SOURCES = ss.condition == "A"
for note in ss.flash:
    st.toast(note)
ss.flash = []

with st.sidebar:
    st.title("🧠 Memory Provenance Assistant")
    st.caption(f"Mode: **{MODE}** · Session `{ss.session_id}`")
    st.divider()
    render_memory_panel()
    st.divider()
    with st.expander("Researcher controls"):
        st.radio("Study condition", ["A", "B"], key="condition",
                 format_func=lambda c: {"A": "A: memories + source context",
                                        "B": "B: memories only"}[c])
        st.toggle("Researcher view", key="researcher_view")
        st.toggle("Memory influence check", key="influence_check",
                  help="Turn off during study sessions so both conditions see the same interface.")
        st.button("Reset demo", key="reset_demo", on_click=reset_demo)
        log_file = RUNTIME / "interaction_log.csv"
        if log_file.exists():
            st.download_button("Download interaction log", log_file.read_bytes(),
                               file_name=f"interaction_log_{ss.session_id}.csv", mime="text/csv")

st.header("Your assistant")
if MODE == "replay":
    st.info("**Replay mode:** replies are scripted (simulated) for the three demo prompts below. "
            "Which memories are sent still depends on your corrections, so the reply changes as it would live.")
    cols = st.columns(len(model.scenarios))
    for col, scenario in zip(cols, model.scenarios):
        col.button(scenario["prompt"], key=f"demo_{scenario['id']}", on_click=queue_prompt,
                   args=(scenario["prompt"],), use_container_width=True)

for i, msg in enumerate(ss.messages):
    if msg["role"] == "notice":
        with st.chat_message("assistant", avatar="🛠️"):
            st.success(msg["text"])
            if msg["kind"] == "excluded" and not msg["undone"]:
                st.button("Undo", key=f"undo_{i}", on_click=undo_exclusion, args=(i,))
            elif msg.get("undone"):
                st.caption("Undone.")
        continue
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_reply(i, msg)

typed = st.chat_input("Ask your assistant anything…")
prompt = typed or ss.pending_prompt
ss.pending_prompt = None
if prompt:
    history = history_before(len(ss.messages))
    ss.messages.append({"role": "user", "text": prompt})
    with st.spinner("Thinking…"):
        ss.messages.append(respond(prompt, history))
    st.rerun()
