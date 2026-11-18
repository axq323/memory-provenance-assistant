# Prompt log

Each run of `eval/run_eval.py` appends one row: prompt version, model, score, and what changed. Change one thing at a time, then rerun the same cases (Lecture 09).

Criteria (written before tuning) are in `eval/cases.json`. Replay mode runs only the three scripted demo cases, so it checks the pipeline, not model judgement. Live runs score the model on all ten.

| Run | Prompt | Model | Score | Change made | Results |
|---|---|---|---|---|---|
| 20261118-230901 | v1 | replay: replay (scripted demo replies) | 1/3 | Replay baseline | `eval\results\20261118-230901_replay.csv` |
