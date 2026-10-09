"""Reassign only unstarted fixed cases after exact replay validation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

DEPLOYMENT = "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/h2oplus_v20_formal_confirmation_evaluation_v4"
PYTHON_ENV = "/home/erzhu419/anaconda3/envs/LSTM-RL"
HPC_BASE = "/home/zhengliang01/scheduleurm_work/thesis-h2o-source-guard-20261006"
PREFIX = "thesis-h2o-source-guard-20261006/jtl110gpu/sac_guarded"
HPC_NODES = [f"node{i:03d}" for i in range(1, 7)]


def placement(task, node, dataset_stat):
    hpc = node.startswith("node")
    base = HPC_BASE if hpc else str(HERE / "runtime")
    campaign = base + "/campaign" if hpc else str(HERE)
    args = shlex.split(task["cmd"])
    args = [arg.replace(DEPLOYMENT, base).replace(str(HERE), campaign) for arg in args]
    for option, value in (("--verified-dataset-size-bytes", dataset_stat["size_bytes"]),
                          ("--verified-dataset-mtime-ns", dataset_stat["mtime_ns"])):
        if option in args:
            args[args.index(option) + 1] = str(value)
    prefix = [base + "/proot", "-R", base + "/guestroot", "-b", "/home"]
    if hpc:
        prefix += ["-b", base + "/env:" + PYTHON_ENV]
    prefix += ["-w", campaign, "/usr/bin/env",
               "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcurl.so.4"]
    env = dict(task["extra_env"])
    env.pop("LD_PRELOAD", None)
    env["PYTHONPATH"] = base + "/workspace/H2Oplus"
    env["TMPDIR"] = base + "/tmp/" + node
    changes = {"cmd": shlex.join(prefix + args), "cwd": campaign,
            "extra_env": env, "require_node": node, "preferred_node": None,
            "skip_launch_staging": True, "last_block_reason": None}
    if hpc and task.get("id"):
        env["TMPDIR"] = base + "/tmp/" + task["id"]
        changes.update(require_node=None, preferred_node=node,
                       allowed_nodes=list(HPC_NODES), allowed_nodes_user_explicit=True,
                       allowed_nodes_submitted=list(HPC_NODES))
        changes["cmd"] = ("mkdir -p " + shlex.quote(env["TMPDIR"]) +
                          " && exec " + changes["cmd"])
    return changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", action="append", required=True,
                        help="Validated node and available task slots, e.g. node001:30")
    args = parser.parse_args()
    slots = dict((node, int(count)) for node, count in (x.split(":") for x in args.capacity))
    stats = {}
    for node in slots:
        check = HERE / ("replay_check_" + ("node001" if node.startswith("node") else node) + ".json")
        if not json.loads(check.read_text())["passed"]:
            raise SystemExit("Exact replay failed: " + node)
        base = HPC_BASE if node.startswith("node") else str(HERE / "runtime")
        dataset = base + "/workspace/H2Oplus/bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
        script = ("import json,os; s=os.stat(" + repr(dataset) +
                  "); print(json.dumps({'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns}))")
        python = scheduler.NODES[node].get("resume_scan_python", "python3")
        _, output, _ = scheduler.run_on(node, "mkdir -p " + shlex.quote(base + "/tmp/" + node) +
                                        " && " + shlex.quote(python) + " -c " + shlex.quote(script),
                                        timeout=20, check=True)
        stats[node] = json.loads(output)
    record_path = HERE / "distributed_placement.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {"changes": [], "result_nodes": []}
    assigned = Counter()
    with scheduler.state_lock(timeout_s=15):
        state = scheduler.load_state()
        for task in state["tasks"]:
            if (task["status"] != "queued" or not task.get("signature", "").startswith(PREFIX)
                    or task.get("require_node") != "jtl110gpu"):
                continue
            available = [node for node in slots if assigned[node] < slots[node]]
            if not available:
                break
            node = min(available, key=lambda name: assigned[name] / slots[name])
            changes = placement(task, node, stats[node])
            record["changes"].append({"id": task["id"], "node": node,
                                       "previous": {key: task.get(key) for key in changes}})
            task.update(changes)
            assigned[node] += 1
        record["result_nodes"] = sorted(set(record["result_nodes"]) | set(assigned))
        record["updated_at"] = time.time()
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        scheduler.save_state(state)
    print(json.dumps({"reassigned": dict(assigned), "record": str(record_path)}, ensure_ascii=False))
