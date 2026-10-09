"""Pull only the JSON fields needed for paired analysis, leaving full runs remote."""
import argparse
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

HPC_BASE = Path("/home/zhengliang01/scheduleurm_work/thesis-h2o-source-guard-20261006")


def collect(node, replay=False):
    hpc = node in {f"node{i:03d}" for i in range(1, 7)}
    remote = HPC_BASE / "campaign" if hpc else HERE
    script = """
import json
from pathlib import Path
root = Path(ROOT)
rows = []
for p in sorted(root.glob(PATTERN)):
    d = json.loads(p.read_text())
    if REPLAY:
        small = d
    else:
        small = {k:d[k] for k in ('method','sumo_seed','episode','service')}
        small['policy'] = {
            'training_metadata':{'training_seed':d['policy']['training_metadata']['training_seed']},
            'execution_diagnostic':d['policy']['execution_diagnostic']}
        small['scenario'] = {k:d['scenario'][k] for k in ('demand_scale','selection_seed')}
        small['passenger_welfare'] = {k:d['passenger_welfare'][k] for k in ('accrued_generalized_seconds_per_departed_passenger',)}
        small['passengers'] = {k:d['passengers'][k] for k in (
            'completion_rate','accrued_wait_seconds_per_departed_passenger',
            'accrued_in_vehicle_seconds_per_departed_passenger',
            'unfinished_passengers','departed_passengers')}
        small['operations'] = {k:d['operations'][k] for k in ('hold_seconds','nonzero_hold_rate')}
        small['server_result_path'] = str(p)
    rows.append({'name':p.name,'result':small})
print(json.dumps(rows))
"""
    script = script.replace("ROOT", repr(str(remote / ("replay" if replay else "results"))))
    script = script.replace("PATTERN", repr("replay_raw*.json" if replay else "sac_guarded*.json"))
    script = script.replace("REPLAY", repr(replay))
    python = (scheduler.NODES[node]["resume_scan_python"] if hpc else
              "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python")
    cmd = python + " - <<'PY'\n" + script + "\nPY"
    code, output, error = scheduler.run_on(node, cmd, timeout=60, check=True)
    rows = json.loads(output)
    destination = HERE / ("replay_" + node if replay else "results")
    destination.mkdir(exist_ok=True)
    for row in rows:
        row["result"]["evaluation_node"] = "hpc-shared" if hpc else node
        (destination / row["name"]).write_text(json.dumps(row["result"], ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"node":node,"copied":len(rows),"transfer_bytes":len(output.encode()),"directory":str(destination)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    shared_collected = False
    for node in dict.fromkeys(args.node):
        hpc = node in {f"node{i:03d}" for i in range(1, 7)}
        if hpc and shared_collected:
            continue
        collect(node, args.replay)
        shared_collected |= hpc
