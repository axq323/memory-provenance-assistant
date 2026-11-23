"""Memory Provenance Assistant — Streamlit interface.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from mpa.context import assemble  # noqa: E402
from mpa.counterfactual import JUDGE_PROMPT_VERSION, calls_needed, check_influence, summarise  # noqa: E402
from mpa.display import tame_headings  # noqa: E402
from mpa.interaction_log import log_event, read_log  # noqa: E402
from mpa.llm import ModelError, ReplayModel, get_judge, get_model  # noqa: E402
from mpa.memory_store import MemoryStore  # noqa: E402
from mpa.evaluation import (FIT_ANSWERS, LIKERT, QUESTIONNAIRE, clean_comment, ground_truth,  # noqa: E402
                       reliance, score_fit, score_prediction, summarise_task)
from mpa.task_types import LABELS, classify, label  # noqa: E402
from mpa.validate import ValidatedReply, validate  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.getenv("MPA_RUNTIME_DIR", ROOT / "runtime"))

st.set_page_config(page_title="Memory Provenance Assistant", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- state ---
@st.cache_resource
def load_model():
    return get_model()


@st.cache_resource
def load_judge():
    return get_judge()


model = load_model()
judge = load_judge()
MODE = "replay" if isinstance(model, ReplayModel) else "live"

# A locked evaluation session (?evaluate=1 in the URL) is what a participant sees: evaluation mode
# is on, the memory-sources setting comes from the URL (?condition=B) and can't be changed, and
# the developer, reset and download controls are hidden.
SESSION_LOCKED = st.query_params.get("evaluate", "0") == "1"
URL_CONDITION = "B" if st.query_params.get("condition", "A").upper() == "B" else "A"

ss = st.session_state
ss.setdefault("messages", [])          # user / assistant turns
ss.setdefault("session_id", uuid.uuid4().hex[:6])


def session_dir() -> Path:
    """Each browser session has its own working copy of the memories and its own log,
    so one person's corrections can't change another person's replies."""
    return RUNTIME / "sessions" / ss.session_id


def log_path() -> Path:
    return session_dir() / "interaction_log.csv"


store = MemoryStore(path=session_dir() / "memories.json")
ss.setdefault("open_sources", set())   # keys of source panels currently open
ss.setdefault("pending_prompt", None)
ss.setdefault("pending_redo", None)     # index of a reply to rewrite after "Not for ..."
ss.setdefault("flash", [])             # short confirmations shown as toasts
ss.setdefault("influence_check", True)  # show the counterfactual check (hidden in evaluation mode)
# Evaluation mode turns each request into a guided task (predict -> reply -> react -> fit question,
# then a short questionnaire), so the design can be tested as Assignment 1 proposes. It hides the
# influence check. Switch it on in the sidebar or with ?evaluate=1 in the URL.
ss.setdefault("evaluation_mode", st.query_params.get("evaluate", "0") == "1")
ss.setdefault("awaiting_prediction", None)  # prompt waiting for the user's prediction
# A task runs from the prediction to the fit question: {"msg_index", "turn_id", "started"}.
# While it is open the user reacts freely; new prompts are locked until it is finished.
ss.setdefault("eval_task", None)
ss.setdefault("awaiting_fit", False)
ss.setdefault("questionnaire", None)   # None, "open" or "done" (end of an evaluation session)
# Memory-sources setting: A = memories shown with where they came from (the design), B = memories
# only, so the two can be compared. Logged as "condition". Fix it with ?condition=B in the URL.
ss.setdefault("condition", URL_CONDITION)
if SESSION_LOCKED:                     # enforced on every run, not just as a default
    ss.evaluation_mode = True
    ss.condition = URL_CONDITION
    ss.developer_view = False
    ss.influence_check = False


def log(event: str, **kw) -> None:
    log_event(event, session_id=ss.session_id, condition=ss.condition, mode=MODE,
              path=log_path(), **kw)


# -------------------------------------------------------------- actions ---
def history_before(index: int) -> list[dict]:
    """Chat history for a new prompt. A reply that was later regenerated (after
    'Not for ...') is left out, so a corrected reply can't leak into new ones."""
    superseded = {m.get("regenerated_from") for m in ss.messages if m.get("regenerated_from")}
    hist = []
    for m in ss.messages[:index]:
        if m["role"] == "user":
            hist.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant" and m["turn_id"] not in superseded:
            hist.append({"role": "assistant", "content": m["reply"].answer})
    return hist


def history_of(msg: dict) -> list[dict]:
    """The exact history that was sent with this reply (reused for regenerations and checks)."""
    return [dict(h) for h in msg.get("history", [])]


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
            "task": task, "reply": reply, "record": record,
            "history": [dict(h) for h in history]}


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
    ss.pending_redo = msg_index   # rewritten in the page script, so a progress message can show


def run_pending_redo() -> None:
    """Rewrite a corrected reply, with the same history it was sent with."""
    msg = ss.messages[ss.pending_redo]
    ss.pending_redo = None
    with st.spinner("Rewriting the reply without that memory…"):
        redo = respond(msg["user_text"], history_of(msg))
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


def exclude_from_panel(memory_id: str) -> None:
    """Scoped correction from the memory panel, without waiting for a reply that uses it."""
    task = ss.get(f"nft_pick_{memory_id}")
    if not task or store.get(memory_id) is None:
        return
    store.exclude_for_task(memory_id, task)
    log("not_for_task", task=task, memory_id=memory_id, detail="from memory panel")
    ss.flash.append(f"I won't use that memory for {label(task)} any more.")


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
                    r.text, r.evidence, r.reply_without = "[forgotten]", "[forgotten]", "[forgotten]"
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
    clear_session_state()
    log("demo_reset")
    ss.flash.append("Demo reset: memories restored and chat cleared.")


def clear_chat() -> None:
    """Start a fresh conversation but keep the memories and the user's corrections."""
    clear_session_state()
    log("chat_cleared")
    ss.flash.append("Chat cleared. Your memories and corrections are kept.")


def clear_session_state() -> None:
    ss.messages = []
    ss.open_sources = set()
    ss.awaiting_prediction = None
    ss.pending_redo = None
    ss.eval_task = None
    ss.awaiting_fit = False
    ss.questionnaire = None


def new_session() -> None:
    """A new session id with its own fresh copy of the memories and its own log, so the
    next person's actions are kept apart. The previous session's files are left as they are."""
    clear_session_state()
    ss.session_id = uuid.uuid4().hex[:6]
    shown = "shown" if ss.condition == "A" else "hidden"
    ss.flash = [f"New session {ss.session_id}: memory sources {shown}."]
    log("session_started", detail=f"condition={ss.condition}")


def open_questionnaire() -> None:
    ss.questionnaire = "open"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _task_rows(turn_id: str) -> list[dict]:
    """This session's log rows after the task's prediction was made."""
    rows = [r for r in read_log(log_path()) if r["session_id"] == ss.session_id]
    start = max((i for i, r in enumerate(rows)
                 if r["event"] == "prediction_made" and r["turn_id"] == turn_id), default=-1)
    return rows[start + 1:]


def finish_task() -> None:
    """Evaluation mode: the user has finished reacting to the reply. Log what they
    did (behavioural reliance), then ask the fit question about the memories used."""
    task = ss.eval_task
    msg = ss.messages[task["msg_index"]]
    used = {u["id"] for u in msg["reply"].used}
    truth = ground_truth(msg["user_text"])
    behaviour = summarise_task(_task_rows(task["turn_id"]), task["started"], _now())
    misapplied = used & set(truth["misapplied"]) if truth else set()
    fitting = used & set(truth["fits"]) if truth else set()
    outcome = reliance(misapplied, fitting, behaviour)
    msg["task_summary"] = {**behaviour, "reliance": outcome}
    log("task_finished", turn_id=task["turn_id"], task=msg["task"],
        detail=f"scenario={truth['scenario'] if truth else 'unknown'};"
               f"accepted={behaviour['accepted']};flagged={behaviour['flagged']};"
               f"corrected={','.join(behaviour['corrected']) or 'none'};"
               f"sources_opened={behaviour['sources_opened']};seconds={behaviour['seconds']};"
               f"misapplied_shown={','.join(sorted(misapplied)) or 'none'};reliance={outcome}")
    if fit_questions(msg):
        ss.awaiting_fit = True
    else:                                   # no memory was used, so there is nothing to judge
        ss.eval_task = None


def fit_questions(msg: dict) -> list[dict]:
    """The memories the fit question asks about: those the assistant used in the
    reply the user reacted to, worded as they were sent (forgotten ones skipped)."""
    used = {u["id"] for u in msg["reply"].used}
    return [m for m in msg["record"].memories_sent if m["id"] in used and m["text"] != "[forgotten]"]


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


def render_memory_chip(msg_index: int, msg: dict, used: dict, note: str | None = None) -> None:
    """One memory under a reply, with its controls. Used for memories the assistant
    claimed and for memories that were sent but not claimed."""
    mem = store.get(used["id"])
    key = f"{msg['turn_id']}:{used['id']}"
    with st.container(border=True):
        if mem is None:
            st.caption("A memory sent with this reply has since been forgotten.")
            return
        st.markdown(f"🧠 **{mem['text']}**" + ("  _(edited by you)_" if mem.get("edited") else ""))
        if used["how"]:
            st.caption(f"How it was applied: {used['how']}")
        if note:
            st.caption(note)
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
    "fits, no difference": "➖ Fits the reply, but it would be the same without it",
    "unreported influence": "⚠️ Changed the reply, but the assistant didn't list it",
    "claimed, no measurable influence": "⚠️ Listed by the assistant, but made no measurable difference",
    "unclear": "❔ Couldn't tell",
}


def clear_influence(msg_index: int) -> None:
    ss.messages[msg_index].pop("influence", None)


def influence_label(r) -> str:
    if r.verdict == "agrees":
        return "✅ Used, as the assistant said" if r.influenced else "✅ Not used, as the assistant said"
    return VERDICT_LABELS[r.verdict]


def check_note(msg: dict, memory_id: str) -> str | None:
    """The influence check's verdict for one memory, shown on its chip."""
    for r in msg.get("influence") or []:
        if r.id == memory_id:
            return f"Measured by the check: {influence_label(r)}"
    return None


def memories_as_sent(msg: dict) -> list[dict]:
    """The memories exactly as they were sent with this reply. Later edits or
    exclusions don't change what is tested; forgotten memories are skipped."""
    return [{"id": m["id"], "text": m["text"], "excluded_tasks": [],
             **({"source_title": m["source"]} if m.get("source") else {})}
            for m in msg["record"].memories_sent if m["text"] != "[forgotten]"]


def run_influence_check(msg_index: int) -> dict:
    """Counterfactual check: regenerate without each memory and compare (mpa/counterfactual.py).
    Uses the memories and history that were actually sent with this reply."""
    msg = ss.messages[msg_index]
    rec = msg["record"]
    reported = {u["id"] for u in msg["reply"].used}
    results = check_influence(model, msg["user_text"], memories_as_sent(msg), rec.task, rec.trigger,
                              reported, history_of(msg), shown_answer=msg["reply"].answer, judge=judge)
    msg["influence"] = results
    counts = summarise(results)
    log("influence_checked", turn_id=msg["turn_id"], task=rec.task,
        detail=";".join(f"{r.id}={r.verdict}" for r in results)
        + (f";judge={judge.model}/{JUDGE_PROMPT_VERSION}" if judge is not None else ""))
    return counts


def render_influence(msg_index: int, msg: dict) -> None:
    results = msg.get("influence")
    if results is None:
        calls = calls_needed(len(msg["record"].memories_sent), MODE == "live")
        help_text = ("Removes each memory in turn and regenerates the reply. A judge model then checks, "
                     "for each reply separately, whether it follows the memory. "
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
        if judge is not None:
            st.caption(f"Judged by {judge.model} (judge prompt {JUDGE_PROMPT_VERSION}), one reply at a time.")
        if not results:
            st.caption("No memories were sent with this request, so there was nothing to test.")
            return
        st.button("↻ Run the check again", key=f"infl_again_{msg['turn_id']}",
                  on_click=clear_influence, args=(msg_index,),
                  help="Clears this result so the check can run again on the same reply, "
                       "e.g. after the judge has changed. Each run is logged with its judge version.")
        for r in results:
            said = "listed it" if r.reported else "didn't list it"
            changed = {True: "changed the reply", False: "made no difference", None: "unclear"}[r.influenced]
            st.markdown(f"🧠 {r.text}  \n{influence_label(r)}")
            st.caption(f"Assistant {said} · removing it {changed} · {r.evidence}")
            if r.reply_without and r.reply_without != "[forgotten]":
                with st.popover("See the reply without this memory"):
                    st.markdown(tame_headings(r.reply_without))
        mismatches = [r for r in results if r.verdict in ("unreported influence",
                                                          "claimed, no measurable influence")]
        if mismatches:
            st.warning(f"{len(mismatches)} memor{'y' if len(mismatches) == 1 else 'ies'} "
                       "behaved differently from what the assistant reported.")


def render_reply(msg_index: int, msg: dict) -> None:
    reply: ValidatedReply = msg["reply"]
    rec = msg["record"]
    st.markdown(tame_headings(reply.answer))
    if reply.note:
        st.warning(reply.note)
    # 1. memories the assistant says it used
    if reply.used:
        st.markdown("**Memories used in this reply** _(as reported by the assistant)_")
        for used in reply.used:
            render_memory_chip(msg_index, msg, used, note=check_note(msg, used["id"]))
    elif reply.parse_ok:
        st.caption("The assistant didn't report using any memories.")
    # 2. memories that were sent but not claimed: the assistant could still have used them
    claimed = {u["id"] for u in reply.used}
    unclaimed = [m for m in rec.memories_sent if m["id"] not in claimed]
    if unclaimed:
        flagged = {r.id for r in msg.get("influence") or [] if r.verdict == "unreported influence"}
        title = f"Also sent with this request, not listed by the assistant ({len(unclaimed)})"
        with st.expander(("⚠️ " if flagged else "") + title, expanded=bool(flagged)):
            st.caption("The assistant could see these memories while writing the reply, "
                       "so it may have used them without saying so. You can correct them here too.")
            for m in sorted(unclaimed, key=lambda m: m["id"] not in flagged):   # flagged first
                render_memory_chip(msg_index, msg, {"id": m["id"], "how": ""},
                                   note=check_note(msg, m["id"]))
    # 3. memories that were not sent, e.g. because the user restricted them
    for w in rec.memories_withheld:
        mem = store.get(w["id"])
        st.caption(f"⛔ Not sent: {mem['text'] if mem else '[forgotten]'} ({w['reason']})")
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
    if ss.get("influence_check") and not ss.get("evaluation_mode"):
        render_influence(msg_index, msg)
    if ss.get("developer_view"):
        render_record(msg)


def render_record(msg: dict) -> None:
    rec = msg["record"]
    reply: ValidatedReply = msg["reply"]
    with st.expander("Developer view: what was sent to the model"):
        st.markdown(f"- **Task type:** {label(rec.task)} (trigger: `{rec.trigger or 'none'}`)\n"
                    f"- **Prompt version:** {rec.prompt_version} · **Model:** {model.name}\n"
                    f"- **History turns sent:** {rec.history_turns_sent}")
        if msg.get("prediction"):
            p = msg["prediction"]
            st.markdown(f"- **You predicted:** {', '.join(p['predicted']) or 'none'} "
                        f"(exact match: {p['exact']}, overlap: {p['jaccard']})")
        if msg.get("task_summary"):
            t = msg["task_summary"]
            st.markdown(f"- **Reaction:** accepted {t['accepted']}, flagged {t['flagged']}, "
                        f"corrected {', '.join(t['corrected']) or 'nothing'}, "
                        f"sources opened {t['sources_opened']}, {t['seconds']} s → **{t['reliance']}**")
        if msg.get("fit"):
            st.markdown("- **Fit answers:** " + "; ".join(
                f"{mid}: {f['answer']} (truth: {f['truth']}, correct: {f['correct']})"
                for mid, f in msg["fit"].items()))
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
            available = [t for t in LABELS if t not in mem["excluded_tasks"]]
            if available:
                with st.popover("Don't use for…", use_container_width=True):
                    st.selectbox("Kind of request", available, format_func=label,
                                 key=f"nft_pick_{mem['id']}")
                    st.button("Apply", key=f"nft_apply_{mem['id']}", on_click=exclude_from_panel,
                              args=(mem["id"],))
            if SHOW_SOURCES:
                st.button("Hide source" if key in ss.open_sources else "Show source",
                          key=f"src_{key}", on_click=toggle_source, args=(key, mem["id"]),
                          use_container_width=True)
            render_edit_forget(st.columns(2), mem, key)
            if SHOW_SOURCES and key in ss.open_sources:
                render_source(mem["id"])


def render_controls() -> None:
    """Developer controls. Hidden in a locked evaluation session (?evaluate=1)."""
    with st.expander("Demo and evaluation controls"):
        st.radio("Memory sources", ["A", "B"], key="condition",
                 format_func=lambda c: {"A": "Shown: memories + where they came from",
                                        "B": "Hidden: memories only (for comparison)"}[c])
        st.toggle("Developer view", key="developer_view",
                  help="Shows what was sent to the model with each reply.")
        st.toggle("Evaluation mode", key="evaluation_mode",
                  help="Guided tasks: predict which memories will be used, react to the reply, say "
                       "whether each memory suited the request, then a short questionnaire. "
                       "Hides the influence check.")
        st.toggle("Memory influence check", key="influence_check",
                  help="Developer tool. Hidden automatically in evaluation mode.")
        st.button("Clear chat (keep memories)", key="clear_chat", on_click=clear_chat,
                  help="Starts a fresh conversation, so earlier replies can't influence the next one. "
                       "Memories and your corrections are kept.")
        st.button("Reset demo", key="reset_demo", on_click=reset_demo,
                  help="Restores the original memories and clears the chat.")
        st.button("Start a new session", key="new_session", on_click=new_session,
                  help="Starts a new session ID with fresh memories and its own log, "
                       "keeping the memory-sources setting.")
        render_log_download()
        st.caption("For a participant, open the app with `?evaluate=1` (sources shown) or "
                   "`?evaluate=1&condition=B` (sources hidden). That locks the setting and hides these controls.")


def render_log_download() -> None:
    """This session's log only."""
    if log_path().exists():
        st.download_button("Download this session's log", log_path().read_bytes(),
                           file_name=f"interaction_log_{ss.session_id}.csv", mime="text/csv",
                           key="download_log")


# ---------------------------------------------------------------- page ---
SHOW_SOURCES = ss.condition == "A"
# In evaluation mode a new prompt waits until the current task (prediction -> reply -> fit) is finished.
LOCKED = bool(ss.awaiting_prediction) or (ss.evaluation_mode and (ss.eval_task is not None
                                                             or ss.questionnaire is not None))
for note in ss.flash:
    st.toast(note)
ss.flash = []

with st.sidebar:
    st.title("🧠 Memory Provenance Assistant")
    st.caption(f"Mode: **{MODE}** · Session `{ss.session_id}`")
    st.divider()
    render_memory_panel()
    if not SESSION_LOCKED:
        st.divider()
        render_controls()

st.header("Your assistant")
if MODE == "replay":
    st.info("**Replay mode:** replies are scripted (simulated) for the three demo prompts below. "
            "Which memories are sent still depends on your corrections, so the reply changes as it would live.")
    cols = st.columns(len(model.scenarios))
    for col, scenario in zip(cols, model.scenarios):
        col.button(scenario["prompt"], key=f"demo_{scenario['id']}", on_click=queue_prompt,
                   args=(scenario["prompt"],), use_container_width=True,
                   disabled=LOCKED)

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

def render_prediction_step() -> None:
    """Evaluation mode: ask which memories the user expects, before showing the reply."""
    memories = store.all()
    with st.chat_message("assistant", avatar="🔮"):
        with st.form("prediction_form"):
            st.markdown("**Before you see the reply:** which of your memories do you expect "
                        "the assistant to use?")
            for mem in memories:
                st.checkbox(mem["text"], key=f"pred_{mem['id']}")
            st.checkbox("None of them", key="pred_none")
            submitted = st.form_submit_button("Show me the reply")
    if not submitted:
        return
    predicted = {m["id"] for m in memories if ss.get(f"pred_{m['id']}")}
    if not predicted and not ss.get("pred_none"):
        st.warning("Tick at least one memory, or “None of them”.")
        return
    if predicted and ss.get("pred_none"):
        st.warning("You ticked “None of them” and a memory. Please choose one or the other.")
        return
    prompt = ss.awaiting_prediction
    with st.spinner("Thinking…"):
        reply_msg = respond(prompt, history_before(len(ss.messages) - 1))
    used = {u["id"] for u in reply_msg["reply"].used}
    score = score_prediction(predicted, used)
    truth = ground_truth(prompt)
    reply_msg["prediction"] = {"predicted": sorted(predicted), **score}
    ss.messages.append(reply_msg)
    log("prediction_made", turn_id=reply_msg["turn_id"], task=reply_msg["task"],
        detail=f"predicted={','.join(sorted(predicted)) or 'none'};used={','.join(sorted(used)) or 'none'};"
               f"exact={score['exact']};jaccard={score['jaccard']};"
               f"scenario={truth['scenario'] if truth else 'unknown'}")
    for key in [f"pred_{m['id']}" for m in memories] + ["pred_none"]:
        ss.pop(key, None)
    ss.awaiting_prediction = None
    ss.eval_task = {"msg_index": len(ss.messages) - 1, "turn_id": reply_msg["turn_id"], "started": _now()}
    st.rerun()


def render_task_controls() -> None:
    """Evaluation mode, after the reply: free reaction first, then the fit question."""
    task = ss.eval_task
    msg = ss.messages[task["msg_index"]]
    if not ss.awaiting_fit:
        with st.chat_message("assistant", avatar="🔮"):
            st.markdown("Use the reply however you would normally: accept it, flag it, open sources "
                        "or correct a memory. When you've finished with it, continue.")
            st.button("I'm done with this reply →", key="task_done", on_click=finish_task, type="primary")
        return
    questions = fit_questions(msg)
    with st.chat_message("assistant", avatar="🔮"):
        with st.form("fit_form"):
            st.markdown(f"**One last question about your request:** “{msg['user_text']}”")
            for m in questions:
                st.radio(f"The assistant used this memory: “{m['text'].rstrip('.')}”. Did it suit your request?",
                         list(FIT_ANSWERS), format_func=FIT_ANSWERS.get, index=None, key=f"fit_{m['id']}")
            submitted = st.form_submit_button("Next task →")
    if not submitted:
        return
    answers = {m["id"]: ss.get(f"fit_{m['id']}") for m in questions}
    if any(a is None for a in answers.values()):
        st.warning("Please answer every question.")
        return
    truth = ground_truth(msg["user_text"])
    msg["fit"] = {}
    for mid, answer in answers.items():
        score = score_fit(answer, mid, truth)
        msg["fit"][mid] = {"answer": answer, **score}
        log("fit_judged", turn_id=task["turn_id"], task=msg["task"], memory_id=mid,
            detail=f"answer={answer};truth={score['truth']};correct={score['correct']};"
                   f"scenario={truth['scenario'] if truth else 'unknown'}")
        ss.pop(f"fit_{mid}", None)
    ss.eval_task = None
    ss.awaiting_fit = False
    st.rerun()


def tasks_finished() -> int:
    return sum(1 for m in ss.messages if m.get("task_summary"))


def render_session_end() -> None:
    """Evaluation mode: 'Finish the session' when no task is open, then the questionnaire."""
    if ss.questionnaire == "done":
        with st.chat_message("assistant", avatar="🔮"):
            st.success("Thank you, that's the end of the session. Your answers are saved in the interaction log.")
            if SESSION_LOCKED:
                st.caption("For the person running the session:")
                render_log_download()
        return
    if ss.questionnaire is None:
        with st.chat_message("assistant", avatar="🔮"):
            st.caption(f"Tasks finished: {tasks_finished()}. Ask the next question, or finish when "
                       "you've done all the tasks.")
            st.button("Finish the session →", key="finish_session", on_click=open_questionnaire)
        return
    with st.chat_message("assistant", avatar="🔮"):
        with st.form("questionnaire_form"):
            st.markdown("**Last step: how did you find the assistant?** "
                        "How much do you agree with each statement? (1 = strongly disagree, 7 = strongly agree)")
            for item, statement, _ in QUESTIONNAIRE:
                st.radio(statement, list(LIKERT),
                         format_func=lambda v: f"{v} {LIKERT[v]}" if v in (1, 7) else str(v),
                         index=None, horizontal=True, key=f"q_{item}")
            st.text_area("Anything else you'd like to tell us? (optional)", key="q_comment")
            submitted = st.form_submit_button("Submit")
    if not submitted:
        return
    answers = {item: ss.get(f"q_{item}") for item, _, _ in QUESTIONNAIRE}
    if any(v is None for v in answers.values()):
        st.warning("Please answer every statement.")
        return
    for item, _, reverse in QUESTIONNAIRE:
        log("questionnaire_answered", detail=f"item={item};score={answers[item]};reverse={reverse}")
    comment = clean_comment(ss.get("q_comment", ""))
    if comment:
        log("questionnaire_answered", detail=f"item=comment;text={comment}")
    log("session_finished", detail=f"tasks={tasks_finished()}")
    ss.questionnaire = "done"
    st.rerun()


if ss.pending_redo is not None:
    run_pending_redo()
    st.rerun()

if ss.awaiting_prediction:
    render_prediction_step()
elif ss.evaluation_mode and ss.eval_task is not None:
    render_task_controls()
elif ss.evaluation_mode:
    render_session_end()

typed = st.chat_input("Ask your assistant anything…", disabled=LOCKED)
prompt = typed or ss.pending_prompt
ss.pending_prompt = None
if prompt and not LOCKED:
    history = history_before(len(ss.messages))
    ss.messages.append({"role": "user", "text": prompt})
    if ss.evaluation_mode:
        ss.awaiting_prediction = prompt            # reply is generated after the prediction
    else:
        with st.spinner("Thinking…"):
            ss.messages.append(respond(prompt, history))
    st.rerun()
