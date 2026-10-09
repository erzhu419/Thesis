#!/usr/bin/env python3
"""Collect only the service summaries needed by the paired controller analysis."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
HPC_ROOT = Path("/home/zhengliang01/scheduleurm_work/Thesis/workpapers/最终控制器同条件比较_20261006")
HPC_NODES = {f"node{i:03d}" for i in range(1, 8)}
LAB_NODES = {"jtl110gpu", "jtl110gpu2", "jtl311linux"}
SCIENTIFIC_FIELDS = (
    "comparison_case", "sumo_seed", "scenario", "policy", "episode",
    "passenger_welfare", "passengers", "operations", "service",
)


# This function is also sent verbatim to the server; no simulator or torch is imported.
def small_result(data):
    return {
        "comparison_case": data["comparison_case"],
        "sumo_seed": data["sumo_seed"],
        "scenario": {key: data["scenario"][key] for key in ("selection_seed", "demand_scale")},
        "policy": {"training_metadata": {"training_seed": data["policy"]["training_metadata"]["training_seed"]}},
        "episode": data["episode"],
        "passenger_welfare": {
            "accrued_generalized_seconds_per_departed_passenger":
                data["passenger_welfare"]["accrued_generalized_seconds_per_departed_passenger"]
        },
        "passengers": {key: data["passengers"][key] for key in (
            "completion_rate", "accrued_wait_seconds_per_departed_passenger",
            "accrued_in_vehicle_seconds_per_departed_passenger", "unfinished_passengers",
            "departed_passengers",
        )},
        "operations": {
            "hold_seconds": {"mean": data["operations"]["hold_seconds"]["mean"]},
            "nonzero_hold_rate": data["operations"]["nonzero_hold_rate"],
        },
        "service": {"completion_rate": data["service"]["completion_rate"]},
    }


def remote_script(root):
    import inspect

    return ("import json\nfrom pathlib import Path\n" + inspect.getsource(small_result)
            + "\nrows = []\n"
            + f"for path in sorted(Path({str(root / 'results')!r}).glob('*.json')):\n"
            + "    data = json.loads(path.read_text(encoding='utf-8'))\n"
            + "    if 'comparison_case' not in data:\n"
            + "        continue  # The evaluator has not published its final record yet.\n"
            + "    result = small_result(data)\n"
            + "    result['server_result_path'] = str(path)\n"
            + "    rows.append({'name': path.name, 'result': result})\n"
            + "print(json.dumps(rows, ensure_ascii=False))\n")


def store_rows(rows, destination, node):
    destination.mkdir(parents=True, exist_ok=True)
    copied = reused = 0
    for row in rows:
        path = destination / row["name"]
        result = row["result"]
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            differing = [field for field in SCIENTIFIC_FIELDS if previous[field] != result[field]]
            if differing:
                raise ValueError(
                    f"Conflicting scientific results for {row['name']}: "
                    f"{previous.get('collection_node', 'existing local result')} vs {node}; "
                    f"differing fields: {', '.join(differing)}"
                )
            reused += 1
            continue
        result["collection_node"] = node
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        copied += 1
    return {"node": node, "available": len(rows), "copied": copied, "reused_identical": reused}


def collect(node, destination, scheduler):
    root = HPC_ROOT if node in HPC_NODES else HERE
    python = scheduler.NODES[node].get("resume_scan_python") or "python3"
    command = shlex.quote(python) + " - <<'PY'\n" + remote_script(root) + "PY\n"
    _, output, _ = scheduler.run_on(node, command, timeout=60, check=True)
    summary = store_rows(json.loads(output), destination, node)
    summary["transfer_bytes"] = len(output.encode("utf-8"))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", action="append", required=True, choices=sorted(HPC_NODES | LAB_NODES))
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "analysis.json", help="Analysis JSON path when --analyze is set")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
    import scheduler

    shared_collected = False
    for node in dict.fromkeys(args.node):
        if node in HPC_NODES and shared_collected:
            continue
        collect(node, args.results, scheduler)
        shared_collected |= node in HPC_NODES
    if args.analyze:
        subprocess.run([sys.executable, str(HERE / "analyze.py"),
                        "--results", str(args.results), "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
