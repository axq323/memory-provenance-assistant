"""Show what was sent to the model and its raw reply for one eval case.

Usage:  python eval/inspect_case.py E04
"""

import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from mpa.context import assemble  # noqa: E402
from mpa.llm import get_model  # noqa: E402
from mpa.memory_store import MemoryStore  # noqa: E402
from mpa.task_types import classify  # noqa: E402
from mpa.validate import validate  # noqa: E402

case_id = sys.argv[1] if len(sys.argv) > 1 else "E04"
cases = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))["cases"]
case = next(c for c in cases if c["id"] == case_id)

with tempfile.TemporaryDirectory() as tmp:
    store = MemoryStore(path=Path(tmp) / "memories.json")
    task, trigger = classify(case["prompt"])
    record = assemble(case["prompt"], store.all(), task, trigger)
    raw = get_model().generate(record, case["prompt"])

print("PROMPT:", case["prompt"])
print("TASK:", task, "| MEMORIES SENT:", sorted(record.sent_ids))
print("SHOULD USE:", case["should_use"], "| SHOULD NOT USE:", case["should_not_use"])
print("\nRAW MODEL REPLY:\n" + raw)
reply = validate(raw, record.sent_ids)
print("\nVALIDATED -> used:", reply.used, "| dropped:", reply.dropped, "| parse_ok:", reply.parse_ok)