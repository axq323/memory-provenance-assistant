# Memory Provenance Assistant

A general everyday chat assistant that shows which stored memories shaped each reply. The person can open the conversation each memory came from, then keep, restrict, edit or forget the memory and see the rewritten reply. Prototype for the Intelligent Interactive Systems project (Assignment 2).

> README prose word count: **about 1,460** (code blocks, commands and tables of file paths excluded)
> Submitted version: tag [`v1.0-submission`](https://github.com/axq323/memory-provenance-assistant/releases/tag/v1.0-submission)

## Project status

**Implemented**
- Streamlit chat with a memory panel ("What I remember about you").
- Keyword task classifier: each request gets a task type before the model call.
- Context assembly: leaves out memories restricted for this task type and records what was sent, withheld and why.
- Reply validator: a memory chip appears only if the model claims the memory **and** it was sent.
- Under each reply: memories reported, sent but not reported, and withheld.
- Memory controls: *Show where this came from*, *Not for [task type]* (with Undo and a rewritten reply), *Edit*, *Forget*.
- Replay mode (scripted, default) and live mode for any OpenAI-compatible endpoint (tested with Ollama and qwen2.5:7b), switchable in the developer controls.
- Memory influence check: regenerates the reply without each memory; in live mode a judge model checks each reply.
- Evaluation mode: four fixed tasks (predict, react, explain, judge fit) and a short questionnaire, with sources shown (A) or hidden (B), per-session logs and analysis scripts.
- Benchmark: 14 cases (4 held out), criteria C1–C4 and a prompt log.

**Simulated**
- The four seeded memories and their source conversations (`data/`) are fictional.
- Replay replies (`data/replay.json`) are scripted. Which one appears depends on the memories actually sent, so corrections still change it.
- `eval/sample_session_log_SIMULATED.csv` holds four scripted sessions. No real people took part.

**Not done (planned)**
- Evaluation mode has not been run with users.
- No memory creation from conversation: no confirm-before-saving, expiry or ask-before-applying. Left out so that errors in creating memories don't confound the comparison.
- In replay mode, editing a memory's wording doesn't change the scripted text.
- The keyword classifier can miss unfamiliar phrasing, so a restriction may not apply.

## Technology stack

| Part | What it does |
|---|---|
| Python 3.10+ | Everything |
| Streamlit ≥ 1.40 | Interface; `streamlit.testing` runs headless UI tests |
| `openai` SDK ≥ 1.40 | Client for any OpenAI-compatible chat endpoint |
| Ollama + qwen2.5:7b | Local assistant model and default judge (free, no key) |
| python-dotenv | Reads settings from `.env` |
| pytest | Tests |
| JSON / CSV files | Seed data, per-session memory copies, interaction logs |

- I tried llama3.2:3b first and dropped it: it followed the vegetarian memory in 12/12 runs but reported it in 0/12 (see the development record).
- **Still undecided:** the judge. By default it is the same model as the assistant (`JUDGE_MODEL=qwen2.5:7b`), so the assistant's own model judges its replies. llama3.2:3b as judge gave false positives, and a stronger independent judge is untested. Also undecided: whether a model-based classifier would be worth losing the keyword rule's predictability.
- With no `.env`, the app uses prompt v2. `.env.example` sets v4, the version adopted after evaluation.

## Running or inspecting the work

```bash
git clone https://github.com/axq323/memory-provenance-assistant.git
cd memory-provenance-assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
streamlit run app.py
```

**Replay mode** is the default and needs no model or key. Click a demo prompt above the chat. A short walkthrough:
1. **Birthday message:** the reply is too formal. Open *Show where this came from*, click *Not for personal messages*, then compare the rewrite. Undo is offered.
2. **Work email:** the formal tone is still used, because the correction was scoped.
3. **Solar panels:** a terse bullet list. Click *Not for explanations* to get the step-by-step version.
4. **Dinner:** both memories fit. Click *Check which memories mattered*.

Turn on *Developer view* under *Demo and evaluation controls* to see exactly what was sent to the model.

**Live mode** (local, free). Start Ollama, then either pick *Live* under *Demo and evaluation controls → Model*, or start the app in live mode:
```bash
ollama pull qwen2.5:7b
# in .env: MODE=live
streamlit run app.py
```
To use a hosted model, set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in `.env` to your own values. `.env` is gitignored; no key is included in this repository.

**Evaluation mode as a participant sees it:** `http://localhost:8501/?evaluate=1` (sources shown) or `?evaluate=1&condition=B` (sources hidden).

**Tests** (replay mode, no model needed):
```bash
pytest            # 221 tests
```

**Benchmark and analysis:**
```bash
python eval/run_eval.py --repeat 3 --prompt v4      # live: 14 cases; replay: the 4 scripted ones
python eval/manual_check.py make eval/results/<run>.csv
python eval/inspect_case.py E04                      # what was sent and the raw reply
python eval/run_diagnostics.py --models llama3.2:3b qwen2.5:7b
python eval/summarise_sessions.py eval/sample_session_log_SIMULATED.csv
python eval/code_reasons.py make                     # blind coding sheet for written reasons
```

| Path | Contents |
|---|---|
| `app.py` | Streamlit interface |
| `mpa/context.py` | Context assembly and system prompts v2–v4 |
| `mpa/validate.py` | Checks the model's claims against what was sent |
| `mpa/llm.py` | Replay and live model back-ends, judge, live availability check |
| `mpa/counterfactual.py` | Memory influence check |
| `mpa/task_types.py` | Keyword classifier |
| `mpa/memory_store.py` | Memories, sources, edit / restrict / forget |
| `mpa/evaluation.py`, `analysis.py`, `reason_coding.py` | Evaluation mode scoring and summaries |
| `data/`, `eval/`, `tests/` | Seed data, benchmark and scripts, tests |

## Development record

Grouped by stage, with links to the commits.

**1. Core pipeline (17 Nov).** [46e827c](https://github.com/axq323/memory-provenance-assistant/commit/46e827c) → [e16e397](https://github.com/axq323/memory-provenance-assistant/commit/e16e397)
- Built the memory store, classifier, context assembly, validator, replay/live back-ends, chips, source view and corrections.
- Decision: a keyword classifier rather than a model, so that a scoped correction is applied *before* the model call and behaves predictably.
- Decision: replay mode, so reviewers need no key and every session sees the same misapplication.
- I first scoped the prototype as a study assistant. On 17 Nov I broadened it to everyday requests: misapplication is clearest when one person's requests cross contexts (a manager email, then a friend's birthday).

**2. The model's self-report proved unreliable (18–21 Nov).** [f345627](https://github.com/axq323/memory-provenance-assistant/commit/f345627), [1c4b78b](https://github.com/axq323/memory-provenance-assistant/commit/1c4b78b), [6afcc3d](https://github.com/axq323/memory-provenance-assistant/commit/6afcc3d), [8494ab5](https://github.com/axq323/memory-provenance-assistant/commit/8494ab5)
- In live runs, llama3.2:3b reported no memories at all.
- To find out whether my code was the cause, I wrote `run_diagnostics.py`, which switches JSON mode and message layout and removes memories, one change at a time. Over three repeats, llama reported memories 0/12 under every setting and qwen2.5:7b 12/12, so it was a model limit. I switched to qwen.

**3. Measuring influence instead of trusting the list (21–22 Nov).** [4e6cecd](https://github.com/axq323/memory-provenance-assistant/commit/4e6cecd), [15d04cb](https://github.com/axq323/memory-provenance-assistant/commit/15d04cb), [345b664](https://github.com/axq323/memory-provenance-assistant/commit/345b664), [b2da9dd](https://github.com/axq323/memory-provenance-assistant/commit/b2da9dd)
- Added the counterfactual check: regenerate without each memory and compare.
- The first judge compared two whole replies and blamed unrelated wording on the removed memory, so I redesigned it to ask one yes/no question about one reply at a time (prompt j3: reason before answer).
- Memories sent but not listed now appear under every reply, because the assistant may use them without saying so.

**4. Fixing history and correction bugs (22 Nov).** [8edf585](https://github.com/axq323/memory-provenance-assistant/commit/8edf585)
- A rewrite now reuses the history the original reply was sent with, and a corrected reply no longer leaks into later turns.
- The influence check tests the memories as they were sent.
- The caption now says "the assistant didn't report using any memories" instead of claiming that none were used.

**5. Prompt versions, tested on held-out cases (22 Nov).** [2ad9a76](https://github.com/axq323/memory-provenance-assistant/commit/2ad9a76), [2b0289c](https://github.com/axq323/memory-provenance-assistant/commit/2b0289c), [bc3e5ce](https://github.com/axq323/memory-provenance-assistant/commit/bc3e5ce), [1f910ca](https://github.com/axq323/memory-provenance-assistant/commit/1f910ca)
- Live runs over-applied memories in E07 and E09. I added held-out cases E11–E14 *before* writing prompt v3, so I could tell whether v3 was tuned to the test.
- Over three runs on 14 cases: v2 8.3, v3 (rules) 8.0, v4 (rules plus each memory's source title) 10.3. I adopted v4 in `.env.example`.

**6. Interaction refinements (23 Nov).** [b49aa18](https://github.com/axq323/memory-provenance-assistant/commit/b49aa18), [9919082](https://github.com/axq323/memory-provenance-assistant/commit/9919082), [6da5fcf](https://github.com/axq323/memory-provenance-assistant/commit/6da5fcf)
- Past replies now show each memory as it was worded at the time, and Forget blanks its wording everywhere in the chat's records.
- Widened the personal-message keywords after the flatmate text (E12) fell through to "other", which meant a scoped correction didn't apply to it.
- Added the work email as a scripted scenario, so the scoped correction can be shown surviving.

**7. Evaluation mode (22–23 Nov).** [9fb4ab5](https://github.com/axq323/memory-provenance-assistant/commit/9fb4ab5), [a5a2770](https://github.com/axq323/memory-provenance-assistant/commit/a5a2770), [66315fc](https://github.com/axq323/memory-provenance-assistant/commit/66315fc), [14625dc](https://github.com/axq323/memory-provenance-assistant/commit/14625dc), [730501a](https://github.com/axq323/memory-provenance-assistant/commit/730501a), [de9a0b2](https://github.com/axq323/memory-provenance-assistant/commit/de9a0b2), [5a8f6ab](https://github.com/axq323/memory-provenance-assistant/commit/5a8f6ab), [9bc8cea](https://github.com/axq323/memory-provenance-assistant/commit/9bc8cea), [0248a9c](https://github.com/axq323/memory-provenance-assistant/commit/0248a9c)
- The fit question comes after the reaction, so it can't prime it; the questionnaire is logged apart from behaviour.
- Each session has its own memories and log. Each task starts from the seeded memories, except the work email, which keeps the birthday task's changes.
- The sources setting is pinned per session, after a test showed a mid-session URL change could mix the two.
- Scoring fixes: noticing a mismatch is coded apart from naming the scope, and correcting a memory the reply didn't use isn't scored as over-reliance.

**8. Being honest about what the benchmark measures (23 Nov).** [060cf87](https://github.com/axq323/memory-provenance-assistant/commit/060cf87)
- C1–C4 score the model's *report*, not the reply. A non-answer that lists the right memories passes. Tests now demonstrate both blind spots, results keep the reply text, and `manual_check.py` supports checking replies by hand.

**9. Replay/live switch (24 Nov).** [6cc8bf0](https://github.com/axq323/memory-provenance-assistant/commit/6cc8bf0815806a10f0fe8b142d948d5510770907)
- Added a replay/live switch to the developer controls (before, it meant editing `.env` and restarting). It goes live only if the endpoint answers, and switching clears the chat so a reply and its influence check always come from the same model.
- Each reply is labelled scripted or live; the switch is fixed during evaluation mode.
- Checked with 8 new tests in `tests/test_mode_switch.py` (endpoint faked) and by hand with Ollama running and stopped.

**How I checked the work:** headless UI and unit tests for each change (221 at submission), repeated live runs with results in `eval/results/` and `eval/prompt_log.md`, and four scripted sessions run end to end through the analysis.

## AI assistance

- **Claude (Anthropic), 23–24 Nov:**
  - Reviewed the report draft against the brief and rubric. I adopted its suggested rationale table, scope statement and the account of the study-assistant change, and checked each claim in the report against the code, commits and `eval/prompt_log.md`.
  - Helped draft this README from the code and commit history. I checked the status, commands and development record against the repository.
  - Wrote the replay/live switch (`mpa/llm.py`, `app.py`) and its tests (`tests/test_mode_switch.py`). I reviewed the code in VS Code, added type-checker suppressions (`# type: ignore`) in two test files, ran `pytest` – 221 passed.
- The design decisions, the scenarios and the evaluation design are my own, and every result in `eval/` comes from my own runs.

## Additional files

- `eval/results/` – CSV output of every benchmark and diagnostic run, logged in `eval/prompt_log.md`.
- `eval/cases.json` – the 14 benchmark cases and criteria. `eval/reason_rubric.md` – rubric for coding written reasons.
- `eval/sample_session_log_SIMULATED.csv` – four scripted evaluation sessions (no participants), used to test the analysis.
- `data/interaction_log_1.csv` – my own development-testing log from 17–22 Nov (replay and live). It is not participant data, and it predates the per-task `trial_id` column.
- No ZIP file is needed.
