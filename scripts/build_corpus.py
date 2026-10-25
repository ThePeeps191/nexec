"""Generate src/nexec/data/corpus_v1.jsonl from the segment source scripts/corpus_v1_data.py."""

import json
import runpy
from pathlib import Path

from nexec.corpus import build, default_path

if __name__ == "__main__":
    data = runpy.run_path(str(Path(__file__).with_name("corpus_v1_data.py")))
    rows = build(data["EXAMPLES"])
    path = default_path()
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} examples to {path}")
