"""Reuse resident inputs; GPU training and independently gated CPU test days."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_usage_v3"))
from usage_common import P as PREVIOUS, module, write
previous = module("explore_launcher_infra", HERE.parent/"mcwm_usage_v3/launch.py")
infra = previous.infra
P = dict(PREVIOUS, **json.loads((HERE/"protocol.json").read_text()))
TRAIN_NODES = ["node007", "jtl110gpu", "jtl110gpu2"]


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


def placements(smoke=False):
    items = []
    for seed in P["training_seeds"][:1] if smoke else P["training_seeds"]:
        for scene in [P["scenarios"][-1]] if smoke else P["scenarios"]:
            for arm in P["arms"]:
                name = f"{scene['name']}/{arm}_seed{seed}"
                items.append(dict(name=name, scenario=scene["name"], arm=arm, seed=seed,
                    node=None, index=len(items), smoke=smoke))
    return items


def located_items(smoke=False):
    items = placements(smoke)
    submission = HERE/("smoke_submission.json" if smoke else "campaign_submission.json")
    if not submission.exists():
        return items
    trains = [r for r in json.loads(submission.read_text())["submitted"] if r["signature"].endswith("/train")]
    records = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "status", "--readonly",
        "--json", "--brief", "--lock-timeout", "30", "--ids", ",".join(r["id"] for r in trains)], text=True))["tasks"]
    nodes = {r["signature"]: r["node"] for r in records}
    for item in items:
        signature = f"{P['name']}/{'smoke' if smoke else 'runs'}/{item['name']}/train"
        item["node"] = nodes.get(signature)
    return items


def gate(item):
    return HERE/"gates"/(("smoke_" if item["smoke"] else "")+item["name"].replace("/", "_")+".json")


def task(item, case=None):
    evaluation = case is not None
    folder = Path("smoke" if item["smoke"] else "runs")/item["name"]
    cmd = (f"{HERE}/node_python -u {HERE}/node_run.py --arm {item['arm']} "
        f"--scenario {item['scenario']} --seed {item['seed']} --out {folder}")
    if item["smoke"]:
        cmd += " --smoke"
    if evaluation:
        cmd += f" --evaluate --case {case}"
    phase = f"eval{case}" if evaluation else "train"
    family = P["name"]+("/eval" if evaluation else "/smoke" if item["smoke"] else "/train")
    row = dict(project="Thesis", description=f"MC-WM exploration {item['name']} {phase}",
        signature=f"{P['name']}/{'smoke' if item['smoke'] else 'runs'}/{item['name']}/{phase}",
        cwd=str(HERE), cmd=cmd, allowed_nodes=infra.CPUS if evaluation else TRAIN_NODES, cpu=1,
        ram_mb=2500 if evaluation else 4000 if item["smoke"] else 13000,
        vram=0 if evaluation else 1024,
        env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"), resource_family=family,
        ram_resource_family=family, vram_resource_family=family,
        skip_launch_staging=True, stage_excludes=["runs", "smoke", "gates"],
        allow_no_ckpt=True, allow_no_resume=True, allow_gpu_over_one_third=True)
    if evaluation:
        row["wait_for_files"] = [str(gate(item))]
    return row


def submit(tasks, name):
    output = HERE/(name+"_submission.json")
    if output.exists():
        raise ValueError("Submission already recorded: "+str(output))
    path = HERE/(name+"_jobs.json")
    write(path, tasks)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(path),
        "--trusted", "--json"], text=True))
    write(output, result)
    print(json.dumps(dict(batch=name, ids=[r["id"] for r in result["submitted"]])), flush=True)


def stage():
    for node in TRAIN_NODES:
        infra.sync(node, str(HERE)+"/", remote(node), ["--include=*.py", "--include=protocol.json", "--exclude=*"])
        config = dict(train=dict(python=infra.executable(node), env=infra.environment(node)))
        if node == "node007":
            config["evaluate"] = dict(python=infra.executable("node001"), env=infra.environment("node001", True))
        path = HERE/"deployments"/node/"node_runtime.json"
        write(path, config)
        infra.sync(node, path, remote(node))
        infra.scheduler.run_on(node, "ln -sfn "+shlex.quote(infra.executable(node))+" "+
            shlex.quote(remote(node)+"/node_python"), timeout=40, check=True)
        path = remote(node, HERE.parent/"mcwm_alignment_v2/inputs")
        infra.scheduler.run_on(node, "test -s "+shlex.quote(path+"/h2oplus_seed6201.pt")+
            " -a -s "+shlex.quote(path+"/h2oplus_seed6202.pt"), timeout=40, check=True)
        print("staged "+node, flush=True)


def collect(items):
    nodes = ["node001"]+sorted({i["node"] for i in items if i["node"]
        and not i["node"].startswith("node") and not gate(i).exists()})
    for node in nodes:
        for part in sorted(set("smoke" if i["smoke"] else "runs" for i in items)):
            (HERE/part).mkdir(exist_ok=True)
            result = subprocess.run(["rsync", "-az", "--include=*/", "--include=*.json", "--exclude=*", "-e",
                infra.scheduler._ssh_rsync_shell_for_node(node),
                infra.scheduler._ssh_target_for_node(node)+":"+remote(node)+f"/{part}/", str(HERE/part)+"/"],
                capture_output=True, text=True, timeout=60)
            if result.returncode not in (0, 23):
                raise RuntimeError(f"JSON collection failed {node}: {result.stderr[-300:]}")


def handoff(items):
    for item in items:
        folder = HERE/("smoke" if item["smoke"] else "runs")/item["name"]
        if gate(item).exists() or not (folder/"results/ready.json").exists():
            continue
        assert item["node"], f"Actual execution node missing for {item['name']}"
        if not item["node"].startswith("node"):
            previous.old_launch.relay(item["node"], remote(item["node"], folder), ["policy.pt"],
                "node001", remote("node001", folder))
        write(gate(item), item)


def watch():
    summarize = module("explore_collector", HERE/"collect_results.py").summarize
    deadline = time.monotonic()+24*3600
    while time.monotonic() < deadline:
        try:
            items = located_items()
            collect(items)
            handoff(items)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-500:], flush=True)
        result = summarize()
        write(HERE/"summary.json", result)
        print(json.dumps({k: result[k] for k in ("complete", "training_ready", "evaluation_ready")}), flush=True)
        if result["complete"]:
            return
        time.sleep(60)
    raise TimeoutError("Three-direction study remains incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-eval", "submit", "collect", "watch"])
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "smoke":
        submit([task(i) for i in placements(True)], "smoke")
    elif mode == "smoke-eval":
        items = located_items(True)
        collect(items)
        handoff(items)
        assert all(gate(i).exists() for i in items)
        submit([task(i, 0) for i in items], "smoke_eval")
    elif mode == "collect":
        items = located_items()
        collect(items)
        handoff(items)
    elif mode == "watch":
        watch()
    else:
        items = placements()
        write(HERE/"locations.json", items)
        submit([task(i) for i in items]+[task(i, c) for i in items for c in range(10)], "campaign")
        submit([dict(project="Thesis", description="MC-WM exploration result handoff", signature=P["name"]+"/watch",
            cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1,
            ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], "watch")
