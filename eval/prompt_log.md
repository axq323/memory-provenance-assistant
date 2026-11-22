# Prompt log

Each run of `eval/run_eval.py` appends one row: prompt version, model, score, and what changed. Change one thing at a time, then rerun the same cases (Lecture 09).

Criteria (written before tuning) are in `eval/cases.json`. Replay mode runs only the three scripted demo cases, so it checks the pipeline, not model judgement. Live runs score the model on all ten.

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
