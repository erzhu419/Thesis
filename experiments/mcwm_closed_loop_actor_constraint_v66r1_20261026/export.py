"""Losslessly split v66's existing analysis; no new learning or native work."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.analyze import analyze

SOURCE = HERE.parent/"mcwm_closed_loop_actor_constraint_v66_20261026"
MAX_BYTES = 48000


def split_summary(value, *, drift_file="policy_drift.json"):
    """Move every line/window diagnostic row, retaining all other evidence."""
    main = deepcopy(value)
    drift = dict(candidate=value["candidate"], stage=value["stage"],
        policy_drift_columns=deepcopy(value["policy_drift_columns"]),
        policy_drift_precision=value["policy_drift_precision"], native_evidence=[])
    for row in main["native_evidence"]:
        drift["native_evidence"].append(dict(seed=row["seed"], scenario=row["scenario"],
                                            policy_drift=row.pop("policy_drift")))
    main["policy_drift_file"] = drift_file
    return main, drift


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()+b"\n"


def export(source_root, out):
    """Recompute the unchanged frozen gate once, then write two small JSONs."""
    source_root, out = Path(source_root), Path(out)
    main, drift = split_summary(analyze(source_root))
    payloads = {out: encoded(main), out.with_name("policy_drift.json"): encoded(drift)}
    for path, raw in payloads.items():
        if len(raw) > MAX_BYTES:
            raise ValueError(f"{path.name} exceeds {MAX_BYTES} bytes: {len(raw)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, raw in payloads.items():
        with path.open("xb") as handle:
            handle.write(raw)
    return dict(status=main["status"], decision=main["decision"],
        files={path.name: len(raw) for path, raw in payloads.items()},
        source_root=str(source_root), new_native_days=0, new_feedback_slots=0, new_policy_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source_root, args.out), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
