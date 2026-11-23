# Prompt log

Each run of `eval/run_eval.py` appends one row: prompt version, model, score, and what changed. Change one thing at a time, then rerun the same cases (Lecture 09).

**What the score measures.** The criteria (C1-C4, written before tuning, in `eval/cases.json`) check the model's *report* of which memories it used, not the reply itself. A score here is a score on those automated reporting criteria. For example, the v2 → v4 change from mean 8.3/14 to 10.3/14 means v4's reports matched the criteria more often. It does not show by itself that v4 applied memories better. Known blind spots: a reply that doesn't answer the request passes if it lists the expected memories, and a misapplied-style reply passes C4 if it reports no memories. From 23 Nov 2026 the results CSV keeps each reply, so a run can be checked by hand with `eval/manual_check.py`; see "Manual checks" below.

**Replay vs live.** Replay mode runs only the scripted demo cases, so it checks the pipeline, not model judgement: three cases until 22 Nov 2026, four from 23 Nov (E04, the work email, became a replay scenario). Live runs score the model on all cases (10 until 22 Nov, 14 after the held-out cases E11-E14 were added).

**Comparability.** On 23 Nov 2026 the task-type keywords were widened (texts and messages to friends, flatmates and family are personal messages). This changes the `corrected` condition for E12 (the flatmate text), so `corrected` runs before and after that date aren't directly comparable.

| Run | Prompt | Model | Score | Change made | Results |
|---|---|---|---|---|---|
| 20261118-230901 | v1 | replay: replay (scripted demo replies) | 1/3 | Replay baseline | `eval\results\20261118-230901_replay.csv` |
| 20261119-012031 | v1 | live: live (llama3.2:3b at http://localhost:11434/v1) | 5/10 | First live run, llama3.2:3b, prompt v1 | `eval\results\20261119-012031_live.csv` |
| 20261119-223747 | v1 | live: live (llama3.2:3b at http://localhost:11434/v1) | 5/10 | Second live run, llama3.2:3b, prompt v1 | `eval\results\20261119-223747_live.csv` |
| 20261119-232300 | v2 | live: live (llama3.2:3b at http://localhost:11434/v1) | 5/10 | v2: memory list first, MUST-list rule, example, no <br> | `eval\results\20261119-232300_live.csv` |
| 20261121-034722 | v2 | live: live (qwen2.5:7b at http://localhost:11434/v1) | 8/10 | v2 prompt, qwen2.5:7b | `eval\results\20261121-034722_live.csv` |
| 20261122-035646 | v2 | live: live (qwen2.5:7b at http://localhost:11434/v1) | 7/10 | qwen reporting rate, run  | `eval\results\20261122-035646_live.csv` |
| 20261122-040000 | v2 | live: live (qwen2.5:7b at http://localhost:11434/v1) | 8/10 | qwen reporting rate, run  | `eval\results\20261122-040000_live.csv` |
| 20261122-040346 | v2 | live: live (qwen2.5:7b at http://localhost:11434/v1) | 6/10 | qwen reporting rate, run  | `eval\results\20261122-040346_live.csv` |
| 20261122-212719 | v2 | live: live (qwen2.5:7b at http://localhost:11434/v1) | mean 8.3/14 (runs: 8, 10, 7) | v2 baseline, 14 cases | `eval\results\20261122-212719_live_v2_default.csv` |
| 20261122-213744 | v3 | live: live (qwen2.5:7b at http://localhost:11434/v1) | mean 8.0/14 (runs: 8, 8, 8) | v3 rules | `eval\results\20261122-213744_live_v3_default.csv` |
| 20261122-214817 | v4 | live: live (qwen2.5:7b at http://localhost:11434/v1) | mean 10.3/14 (runs: 10, 10, 11) | v4 rules + source context | `eval\results\20261122-214817_live_v4_default.csv` |
| 20261122-215754 | v2 | live: live (qwen2.5:7b at http://localhost:11434/v1) | mean 9.0/14 (runs: 9, 9, 9) | [corrected] v2 with test D corrections | `eval\results\20261122-215754_live_v2_corrected.csv` |

## Manual checks

A manual check reads the actual replies of one run against three questions: does it answer the request, does it follow the memories that fit, does it avoid the ones that don't. `python eval/manual_check.py score <sheet>` reports the manual pass rate and how often it agrees with the automated pass/fail.

| Sheet | From run | Replies checked | Manual pass | Agrees with automated |
|---|---|---|---|---|

