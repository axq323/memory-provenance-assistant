"""Run every diagnostic switch on one eval case, for one or more models.

Each run changes one thing, to tell model limits apart from choices in
our own code (JSON mode, how messages are split, whether memories are
sent at all). Prints a comparison table and saves it as a CSV.

Usage (live mode settings come from .env):
    python eval/run_diagnostics.py                                   # E06, model from .env
    python eval/run_diagnostics.py --models llama3.2:3b qwen2.5:7b   # compare models
    python eval/run_diagnostics.py --case E04 --repeat 2             # another case, 2 runs each
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from mpa.context import PROMPT_VERSION, assemble  # noqa: E402
from mpa.llm import LiveModel  # noqa: E402
from mpa.memory_store import MemoryStore  # noqa: E402
from mpa.task_types import classify  # noqa: E402
from mpa.validate import validate  # noqa: E402

# name -> (send memories?, force JSON mode?, one system message?)
RUNS = {
    "normal":               (True,  True,  False),
    "no-memories":          (False, True,  False),
    "no-json":              (True,  False, False),
    "one-system":           (True,  True,  True),
    "one-system + no-json": (True,  False, True),
}


def run_once(model: LiveModel, case: dict, send_memories: bool, json_mode: bool,
             one_system: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(path=Path(tmp) / "memories.json")
        task, trigger = classify(case["prompt"])
        record = assemble(case["prompt"], store.all() if send_memories else [], task, trigger)

    messages = record.messages
    if one_system:
        system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        messages = [{"role": "system", "content": system_text}] + \
                   [m for m in messages if m["role"] != "system"]

    kwargs = {"model": model.model, "messages": messages, "temperature": model.temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        raw = model.client.chat.completions.create(**kwargs).choices[0].message.content or ""
    except Exception as exc:  # show the error in the table instead of stopping
        return {"used": "", "parse_ok": False, "dropped": "", "answer": f"ERROR: {exc}"}

    reply = validate(raw, record.sent_ids)
    return {"used": ",".join(u["id"] for u in reply.used) or "-",
            "parse_ok": reply.parse_ok,
            "dropped": ",".join(d["id"] for d in reply.dropped) or "-",
            "answer": " ".join(reply.answer.split())[:90]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="E06", help="eval case id (default E06, packed lunch)")
    parser.add_argument("--models", nargs="+", default=[os.getenv("LLM_MODEL", "llama3.2:3b")],
                        help="one or more Ollama model names")
    parser.add_argument("--repeat", type=int, default=1, help="runs per switch (replies vary)")
    args = parser.parse_args()

    cases = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))["cases"]
    case = next(c for c in cases if c["id"] == args.case)
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "ollama")

    print(f"Case {case['id']}: {case['prompt']}")
    print(f"Should use: {case['should_use']} | should not use: {case['should_not_use']} "
          f"| prompt {PROMPT_VERSION}\n")

    rows = []
    for model_name in args.models:
        model = LiveModel(base_url=base_url, model=model_name, api_key=api_key)
        for run_name, (send_memories, json_mode, one_system) in RUNS.items():
            for i in range(args.repeat):
                print(f"  running {model_name} · {run_name}" + (f" #{i + 1}" if args.repeat > 1 else "") + " …",
                      flush=True)
                result = run_once(model, case, send_memories, json_mode, one_system)
                rows.append({"model": model_name, "switches": run_name, "repeat": i + 1, **result})

    print(f"\n{'model':14} {'switches':22} {'used':8} {'json ok':8} {'dropped':8} answer starts")
    for r in rows:
        print(f"{r['model']:14} {r['switches']:22} {r['used']:8} {str(r['parse_ok']):8} "
              f"{r['dropped']:8} {r['answer']}")

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}_diagnostics_{case['id']}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()