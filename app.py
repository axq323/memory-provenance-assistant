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
from mpa.evaluation import (FIT_ANSWERS, LIKERT, QUESTIONNAIRE, carries_from, clean_comment,  # noqa: E402
                            correction_actions, correction_outcome, evaluation_schedule, final_state,
                            scenario, truth_for,
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
ss.setdefault("awaiting_reason", False)  # after "I'm done": "What made you choose that?"
ss.setdefault("awaiting_fit", False)
ss.setdefault("trial_count", 0)        # evaluation tasks started in this session
# Evaluation mode serves a fixed list of scenarios (data/replay.json), one task each, in order.
ss.setdefault("eval_schedule", evaluation_schedule())
ss.setdefault("trials_done", [])       # scenario ids of the tasks finished so far
ss.setdefault("awaiting_scenario", None)
ss.setdefault("questionnaire", None)   # None, "open" or "done" (end of an evaluation session)
# Memory-sources setting: A = memories shown with where they came from (the design), B = memories
# only, so the two can be compared. Logged as "condition". Fix it with ?condition=B in the URL.
ss.setdefault("condition", URL_CONDITION)
if SESSION_LOCKED:                     # enforced on every run, not just as a default
    ss.evaluation_mode = True
    ss.condition = URL_CONDITION
    ss.developer_view = False
    ss.influence_check = False


# Actions that can happen in the memory panel, without a reply. During an evaluation task they
# belong to the open task; between tasks the panel is read-only in evaluation mode.
PANEL_EVENTS = {"source_opened", "source_closed", "not_for_task", "undo_not_for_task",
                "memory_edited", "memory_forgotten"}


def trial_of(turn_id: str) -> str:
    """The evaluation task (trial) a reply belongs to. Rewrites inherit the original's trial."""
    for m in ss.messages:
        if m["role"] == "assistant" and m["turn_id"] == turn_id:
            return m.get("trial_id", "")
    return ""


def log(event: str, trial_id: str | None = None, **kw) -> None:
    """Log one event. An action on a reply is filed under that reply's trial, so acting on an
    earlier reply can never be scored as part of a later task."""
    if trial_id is None:
        if kw.get("turn_id"):
            trial_id = trial_of(kw["turn_id"])
        elif event in PANEL_EVENTS and ss.get("eval_task"):
            trial_id = ss.eval_task["trial_id"]
        else:
            trial_id = ""
    log_event(event, session_id=ss.session_id, condition=ss.condition, mode=MODE,
              path=log_path(), trial_id=trial_id, **kw)


# -------------------------------------------------------------- actions ---
def history_before(index: int) -> list[dict]:
    """Chat history for a new prompt. A reply that was later regenerated (after
    'Not for ...') is replaced by its latest rewrite IN ITS OWN PLACE, right after the
    request it answers, so a corrected reply can't leak into new ones and a rewrite of an
    older reply isn't sent after newer turns. (The chat shows the rewrite at the bottom,
    where it was made; the model sees the conversation in order.)"""
    shown = ss.messages[:index]
    rewrite_of = {m["regenerated_from"]: m for m in shown
                  if m["role"] == "assistant" and m.get("regenerated_from")}   # a later rewrite wins

    def latest(m: dict) -> dict:
        while m["turn_id"] in rewrite_of:                 # follows rewrites of rewrites
            m = rewrite_of[m["turn_id"]]
        return m

    hist = []
    for m in shown:
        if m["role"] == "user":
            hist.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant" and not m.get("regenerated_from"):
            hist.append({"role": "assistant", "content": latest(m)["reply"].answer})
    return hist


def history_of(msg: dict) -> list[dict]:
    """The exact history that was sent with this reply (reused for regenerations and checks)."""
    return [dict(h) for h in msg.get("history", [])]


def respond(user_text: str, history: list[dict], trial_id: str = "") -> dict:
    task, trigger = classify(user_text)
    record = assemble(user_text, store.all(), task, trigger, history)
    try:
        reply = validate(model.generate(record, user_text), record.sent_ids)
    except ModelError as exc:
        reply = ValidatedReply(answer="Sorry, I couldn't reach the model.", parse_ok=False, note=str(exc))
    log("reply_shown", turn_id=record.turn_id, task=task, trial_id=trial_id,
        detail="used=" + ",".join(u["id"] for u in reply.used))
    return {"role": "assistant", "turn_id": record.turn_id, "user_text": user_text,
            "task": task, "reply": reply, "record": record, "trial_id": trial_id,
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
                        "undone": False, "turn_id": msg["turn_id"], "memory_text": mem["text"]})
    ss.pending_redo = msg_index   # rewritten in the page script, so a progress message can show


def run_pending_redo() -> None:
    """Rewrite a corrected reply, with the same history it was sent with."""
    msg = ss.messages[ss.pending_redo]
    ss.pending_redo = None
    with st.spinner("Rewriting the reply without that memory…"):
        redo = respond(msg["user_text"], history_of(msg), trial_id=msg.get("trial_id", ""))
    redo["regenerated_from"] = msg["turn_id"]
    ss.messages.append(redo)


def undo_exclusion(notice_index: int) -> None:
    notice = ss.messages[notice_index]
    if store.get(notice["memory_id"]) is None:
        return
    store.include_for_task(notice["memory_id"], notice["task"])
    notice["undone"] = True
    log("undo_not_for_task", turn_id=notice.get("turn_id", ""), task=notice["task"],
        memory_id=notice["memory_id"])
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
        elif m["role"] == "notice" and m.get("memory_id") == memory_id:
            m["memory_text"] = None
    for key in [k for k in ss if isinstance(k, str) and k.startswith("edit_text_") and k.endswith(memory_id)]:
        del ss[key]                            # unsaved text in the Edit boxes
    ss.open_sources = {k for k in ss.open_sources if not k.endswith(memory_id)}
    log("memory_forgotten", turn_id=turn_id, memory_id=memory_id)
    ss.flash.append("Forgotten. It's deleted, and its wording is blanked in this chat's records. "
                    "Replies it already shaped are unchanged.")


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
    ss.awaiting_reason = False
    ss.awaiting_fit = False
    ss.questionnaire = None
    ss.awaiting_scenario = None
    ss.trials_done = []


def new_session() -> None:
    """A new session id with its own fresh copy of the memories and its own log, so the
    next person's actions are kept apart. The previous session's files are left as they are."""
    clear_session_state()
    ss.trial_count = 0
    ss.session_id = uuid.uuid4().hex[:6]
    shown = "shown" if ss.condition == "A" else "hidden"
    ss.flash = [f"New session {ss.session_id}: memory sources {shown}."]
    log("session_started", detail=f"condition={ss.condition}")


def next_scenario() -> str | None:
    """The next scheduled task that hasn't been finished, or None when all are done."""
    return next((sid for sid in ss.eval_schedule if sid not in ss.trials_done), None)


def keeps_previous_memories(scenario_id: str) -> bool:
    """A follow-up task (data/replay.json, evaluation_carry_over) keeps the memories as the
    task it follows left them, if that task was the one just finished."""
    source = carries_from(scenario_id)
    return bool(source) and bool(ss.trials_done) and ss.trials_done[-1] == source


def start_task(scenario_id: str) -> None:
    """Evaluation mode: start from the original memories (unless this is a declared follow-up),
    put the scheduled prompt in the chat and ask for a prediction."""
    if ss.eval_task is not None or ss.awaiting_prediction or scenario_id != next_scenario():
        return
    carried = keeps_previous_memories(scenario_id)
    if not carried:
        store.reset()      # every independent task starts from the same seeded memories
    log("task_started", detail=f"scenario={scenario_id};memories={'carried' if carried else 'reset'}")
    prompt = scenario(scenario_id)["prompt"]
    ss.messages.append({"role": "user", "text": prompt})
    ss.awaiting_prediction = prompt
    ss.awaiting_scenario = scenario_id


def open_questionnaire() -> None:
    if next_scenario() is None:            # only after every scheduled task
        ss.questionnaire = "open"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _task_rows(trial_id: str) -> list[dict]:
    """This session's log rows for one task: actions on its reply, on rewrites of that
    reply, and in the memory panel while the task was open. Nothing from other tasks."""
    return [r for r in read_log(log_path())
            if r["session_id"] == ss.session_id and r.get("trial_id") == trial_id
            and r["event"] not in ("prediction_made", "reply_shown")]


def finish_task() -> None:
    """Evaluation mode: the user has finished reacting to the reply. Log what they
    did (behavioural reliance), then ask the fit question about the memories used."""
    task = ss.eval_task
    msg = ss.messages[task["msg_index"]]
    used = set(task["stimulus"])
    truth = truth_for(task["scenario"])
    rows = _task_rows(task["trial_id"])
    behaviour = summarise_task(rows, task["started"], _now())
    misapplied = used & set(truth["misapplied"]) if truth else set()
    fitting = used & set(truth["fits"]) if truth else set()
    outcome = reliance(misapplied, fitting, behaviour)
    # what each correction was, and where each memory in the reply ended up (after any undo)
    actions = correction_actions(rows)
    states = {mid: final_state(start, store.get(mid), msg["task"]) for mid, start in task["stimulus"].items()}
    correction = correction_outcome(misapplied, fitting, states, actions)
    msg["task_summary"] = {**behaviour, "reliance": outcome, "actions": actions,
                           "final": states, "correction": correction}
    log("task_finished", turn_id=task["turn_id"], task=msg["task"],
        detail=f"scenario={truth['scenario'] if truth else 'unknown'};"
               f"decision={behaviour['decision']};"
               f"accepted={behaviour['accepted']};flagged={behaviour['flagged']};"
               f"corrected={','.join(behaviour['corrected']) or 'none'};"
               f"sources_opened={behaviour['sources_opened']};seconds={behaviour['seconds']};"
               f"misapplied_shown={','.join(sorted(misapplied)) or 'none'};reliance={outcome};"
               f"actions={','.join(actions) or 'none'};"
               f"final={','.join(f'{m}:{v}' for m, v in sorted(states.items())) or 'none'};"
               f"correction={correction}")
    ss.trials_done.append(task["scenario"])
    if fit_questions():
        ss.awaiting_reason = True           # neutral question first, then the fit question
    else:                                   # no memory was used, so there is nothing to judge
        ss.eval_task = None


def fit_questions() -> list[dict]:
    """The memories the fit question asks about: those the assistant used in the original
    reply of the open task, worded as they were then. Taken from the snapshot made when the
    task started, so editing or forgetting a memory can't remove or change the question.
    The snapshot is dropped when the task closes."""
    task = ss.eval_task
    return [{"id": mid, "text": m["text"]} for mid, m in (task or {}).get("stimulus", {}).items()]


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


# What Forget actually does (forget() below), so the confirmation promises no more than that.
FORGET_EXPLAINED = ("Deletes this memory from storage and blanks its wording wherever this chat "
                    "recorded it: what was sent with past replies, check results and notices. "
                    "Replies it already shaped stay as they are, and the conversation it was saved "
                    "from isn't deleted. This can't be undone.")
FORGET_DURING_TASK = "During a task, its wording is kept until the task's last question, then cleared."


def render_edit_forget(cols, mem: dict, key: str, turn_id: str = "") -> None:
    with cols[0].popover("Edit"):
        st.text_area("Memory wording", value=mem["text"], key=f"edit_text_{key}")
        if MODE == "replay":
            st.caption("Replay mode: replies are scripted, so new wording won't change their text.")
        st.button("Save", key=f"edit_save_{key}", on_click=save_edit,
                  args=(mem["id"], f"edit_text_{key}", turn_id))
    with cols[1].popover("Forget"):
        st.caption(FORGET_EXPLAINED + (" " + FORGET_DURING_TASK if ss.evaluation_mode else ""))
        st.button("Forget this memory", key=f"forget_{key}", type="primary",
                  on_click=forget, args=(mem["id"], turn_id))


def reply_is_active(msg: dict) -> bool:
    """Outside evaluation mode every reply can be acted on. In evaluation mode only the replies
    of the open task can, and only until the user says they're done with it."""
    if not ss.evaluation_mode:
        return True
    task = ss.eval_task
    return reacting() and msg.get("trial_id") == task["trial_id"]


def panel_is_active() -> bool:
    """In evaluation mode the memory panel can only change memories while a task is open."""
    return not ss.evaluation_mode or reacting()


def reacting() -> bool:
    """An evaluation task is open and the user hasn't said they're done with the reply yet."""
    return ss.eval_task is not None and not ss.awaiting_reason and not ss.awaiting_fit


def render_memory_chip(msg_index: int, msg: dict, used: dict, note: str | None = None,
                       interactive: bool = True) -> None:
    """One memory under a reply, with its controls. Used for memories the assistant
    claimed and for memories that were sent but not claimed."""
    mem = store.get(used["id"])
    key = f"{msg['turn_id']}:{used['id']}"
    # The wording shown is the wording that was sent with THIS reply (its record), so editing a
    # memory later doesn't rewrite the history. The current wording lives in the memory panel.
    sent = next((m["text"] for m in msg["record"].memories_sent if m["id"] == used["id"]), None)
    with st.container(border=True):
        if mem is None or sent == "[forgotten]":
            st.caption("A memory sent with this reply has since been forgotten.")
            return
        shown = sent if sent is not None else mem["text"]
        st.markdown(f"🧠 **{shown}**")
        if mem["text"] != shown:
            st.caption("You've edited this memory since this reply. Its current wording is in the memory panel.")
        if used["how"]:
            st.caption(f"How it was applied: {used['how']}")
        if note:
            st.caption(note)
        if not interactive:
            return
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
    active = reply_is_active(msg)
    st.markdown(tame_headings(reply.answer))
    if reply.note:
        st.warning(reply.note)
    # 1. memories the assistant says it used
    if reply.used:
        st.markdown("**Memories used in this reply** _(as reported by the assistant)_")
        for used in reply.used:
            render_memory_chip(msg_index, msg, used, note=check_note(msg, used["id"]), interactive=active)
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
                                   note=check_note(msg, m["id"]), interactive=active)
    # 3. memories that were not sent, e.g. because the user restricted them
    for w in rec.memories_withheld:
        text = w.get("text") or "[forgotten]"          # as it was when this reply was written
        if text == "[forgotten]" or store.get(w["id"]) is None:
            text = "a memory you've since forgotten"
        st.caption(f"⛔ Not sent: {text} ({w['reason']})")
    verdict = msg.get("verdict")
    if verdict:
        st.caption("You marked this reply as working for you." if verdict == "accepted"
                   else "You flagged this reply.")
    elif not active:
        if msg.get("trial_id"):
            st.caption("This task is finished, so this reply can't be changed any more.")
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
            st.markdown(f"- **Corrections:** {', '.join(t.get('actions') or []) or 'none'} · final state "
                        f"{', '.join(f'{m}: {v}' for m, v in (t.get('final') or {}).items()) or '–'} "
                        f"→ **{t.get('correction', '–')}**")
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
    active = panel_is_active()
    if not active:
        st.caption("You can change these while a task is open.")
    for mem in memories:
        key = f"panel:{mem['id']}"
        with st.container(border=True):
            st.markdown(mem["text"] + ("  _(edited)_" if mem.get("edited") else ""))
            # The same line in both settings. In setting A the conversation's title and turns
            # appear only once the user opens the source, so what differs between A and B is
            # access to the source, not a title that is always on screen.
            st.caption(f"Saved {mem['created']}")
            for task in mem["excluded_tasks"]:
                st.caption(f"⛔ Not used for {label(task)}")
                if active:
                    st.button(f"Use for {label(task)} again", key=f"again_{mem['id']}_{task}",
                              on_click=use_again, args=(mem["id"], task), use_container_width=True)
            if not active:
                continue
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
# In evaluation mode the tasks are given by the schedule, so the user can't type their own prompts.
LOCKED = bool(ss.awaiting_prediction) or ss.evaluation_mode
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
if MODE == "replay" and not ss.evaluation_mode:
    st.info("**Replay mode:** replies are scripted (simulated) for the demo prompts below. "
            "Which memories are sent still depends on your corrections, so the reply changes as it would live.")
    cols = st.columns(len(model.scenarios))
    for col, scenario in zip(cols, model.scenarios):
        col.button(scenario["prompt"], key=f"demo_{scenario['id']}", on_click=queue_prompt,
                   args=(scenario["prompt"],), use_container_width=True,
                   disabled=LOCKED)

for i, msg in enumerate(ss.messages):
    if msg["role"] == "notice":
        with st.chat_message("assistant", avatar="🛠️"):
            what = f"“{msg['memory_text']}”" if msg.get("memory_text") else "that memory (since forgotten)"
            st.success(f"Got it. I won't use {what} for {label(msg['task'])}. Here's that reply again without it.")
            notice_active = not ss.evaluation_mode or (
                reacting() and trial_of(msg.get("turn_id", "")) == ss.eval_task["trial_id"])
            if msg["kind"] == "excluded" and not msg["undone"] and notice_active:
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
    prompt, scenario_id = ss.awaiting_prediction, ss.awaiting_scenario
    ss.trial_count += 1                         # per session; clearing the chat doesn't reuse an id
    trial_id = f"trial{ss.trial_count}"
    with st.spinner("Thinking…"):
        # each task starts without chat history, so every session gets the same reply to it
        reply_msg = respond(prompt, [], trial_id=trial_id)
    used = {u["id"] for u in reply_msg["reply"].used}
    score = score_prediction(predicted, used)
    truth = truth_for(scenario_id)
    reply_msg["scenario"] = scenario_id
    reply_msg["prediction"] = {"predicted": sorted(predicted), **score}
    ss.messages.append(reply_msg)
    log("prediction_made", turn_id=reply_msg["turn_id"], task=reply_msg["task"],
        detail=f"predicted={','.join(sorted(predicted)) or 'none'};used={','.join(sorted(used)) or 'none'};"
               f"exact={score['exact']};jaccard={score['jaccard']};"
               f"scenario={truth['scenario'] if truth else 'unknown'}")
    for key in [f"pred_{m['id']}" for m in memories] + ["pred_none"]:
        ss.pop(key, None)
    ss.awaiting_prediction = None
    ss.awaiting_scenario = None
    # The memories the reply used, as they were at the start of the task (for the fit question and
    # for scoring where each one ended up). Held only while the task is open.
    sent = {m["id"]: m["text"] for m in reply_msg["record"].memories_sent}
    stimulus = {mid: {"text": sent[mid], "excluded_tasks": list((store.get(mid) or {}).get("excluded_tasks", []))}
                for mid in sorted(used) if mid in sent}
    ss.eval_task = {"msg_index": len(ss.messages) - 1, "turn_id": reply_msg["turn_id"],
                    "trial_id": trial_id, "scenario": scenario_id, "started": _now(), "stimulus": stimulus}
    st.rerun()


def render_task_controls() -> None:
    """Evaluation mode, after the reply: free reaction first, then a neutral question about
    why they did what they did, then the fit question about the original reply."""
    task = ss.eval_task
    msg = ss.messages[task["msg_index"]]
    if reacting():
        with st.chat_message("assistant", avatar="🔮"):
            st.markdown("Use the reply however you would normally: accept it, flag it, open sources "
                        "or correct a memory. When you've finished with it, continue.")
            st.button("I'm done with this reply →", key="task_done", on_click=finish_task, type="primary")
        return
    if ss.awaiting_reason:
        with st.chat_message("assistant", avatar="🔮"):
            with st.form("reason_form"):
                st.text_area("What made you choose what you did with that reply?", key="reason_text",
                             help="A sentence or two in your own words.")
                submitted = st.form_submit_button("Continue →")
        if not submitted:
            return
        reason = clean_comment(ss.get("reason_text", ""))
        if not reason:
            st.warning("Please write a sentence or two before you continue.")
            return
        log("reason_given", turn_id=task["turn_id"], task=msg["task"], detail=f"text={reason}")
        ss.pop("reason_text", None)
        ss.awaiting_reason = False
        ss.awaiting_fit = True
        st.rerun()
    questions = fit_questions()
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
    truth = truth_for(task["scenario"])
    msg["fit"] = {}
    for mid, answer in answers.items():
        score = score_fit(answer, mid, truth)
        msg["fit"][mid] = {"answer": answer, **score}
        log("fit_judged", turn_id=task["turn_id"], task=msg["task"], memory_id=mid,
            detail=f"answer={answer};truth={score['truth']};correct={score['correct']};"
                   f"scenario={truth['scenario'] if truth else 'unknown'}")
        ss.pop(f"fit_{mid}", None)
    ss.eval_task = None                     # drops the snapshot of the memories' wording
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
        upcoming = next_scenario()
        total = len(ss.eval_schedule)
        with st.chat_message("assistant", avatar="🔮"):
            if upcoming is not None:
                n = len(ss.trials_done) + 1
                st.markdown(f"**Task {n} of {total}.** Imagine you're asking your assistant this:")
                st.markdown(f"> {scenario(upcoming)['prompt']}")
                st.caption("This task keeps any changes you made to your memories in the last task."
                           if keeps_previous_memories(upcoming) else
                           "Each task starts from your original memories, so changes from earlier tasks "
                           "are undone when it starts.")
                st.button(f"Start task {n} of {total} →", key="start_task", type="primary",
                          on_click=start_task, args=(upcoming,))
            else:
                st.caption(f"Tasks finished: {tasks_finished()} of {total}. That's all the tasks.")
                st.button("Finish the session →", key="finish_session", on_click=open_questionnaire,
                          type="primary")
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
    log("session_finished", detail=f"tasks={tasks_finished()};scheduled={len(ss.eval_schedule)}")
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

typed = st.chat_input("Ask your assistant anything…" if not ss.evaluation_mode
                      else "In evaluation mode, start each task with the button above.", disabled=LOCKED)
prompt = typed or ss.pending_prompt
ss.pending_prompt = None
if prompt and not LOCKED:
    history = history_before(len(ss.messages))
    ss.messages.append({"role": "user", "text": prompt})
    with st.spinner("Thinking…"):
        ss.messages.append(respond(prompt, history))
    st.rerun()
