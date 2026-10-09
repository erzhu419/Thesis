"""Use the same complete paired-result validation as v5."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_usage_v3"))
from usage_common import module, write
base = module("decision_scale_results", HERE.parent/"mcwm_explore_v5/collect_results.py")
base.HERE = HERE
base.P = dict(base.P, **json.loads((HERE/"protocol.json").read_text()))
summarize = base.summarize


if __name__ == "__main__":
    result = summarize()
    write(HERE/"summary.json", result)
    print(json.dumps({k: result[k] for k in ("complete", "training_ready", "evaluation_ready")}))
