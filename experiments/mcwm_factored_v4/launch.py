"""Reuse resident datasets and distribute full-day evaluations after handoff."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_usage_v3"))
from usage_common import P as PREVIOUS, module, write
previous = module("factored_launcher_infra", HERE.parent/"mcwm_usage_v3/launch.py")
infra = previous.infra
P = dict(PREVIOUS, **json.loads((HERE/"protocol.json").read_text()))


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


def placements(smoke=False):
    items = []
    # jtl311linux currently has no free CPU slots; keep each four-arm group together.
    nodes = ["node007", "jtl110gpu", "node007", "node007", "jtl110gpu2", "node007"]
    for seed_index, seed in enumerate(P["training_seeds"][:1] if smoke else P["training_seeds"]):
        for scene_index, scene in enumerate([P["scenarios"][-1]] if smoke else P["scenarios"]):
            for structure in P["structures"]:
                for arm in P["arms"]:
                    name = f"{scene['name']}/{structure}_{arm}_seed{seed}"
                    items.append(dict(name=name, scenario=scene["name"], structure=structure, arm=arm,
                        seed=seed, node="node007" if smoke else nodes[3*seed_index+scene_index],
                        index=len(items), smoke=smoke))
    return items


def gate(item):
    return HERE/"gates"/(item["name"].replace("/", "_")+".json")


def task(item, case=None):
    evaluation = case is not None
    node = infra.CPUS[(item["index"]*10+case) % 6] if evaluation else item["node"]
    folder = HERE/("smoke" if item["smoke"] else "runs")/item["name"]
    cmd = (f"{infra.executable(node)} -u {remote(node)}/run.py --structure {item['structure']} "
           f"--arm {item['arm']} --scenario {item['scenario']} --seed {item['seed']} --out {remote(node, folder)}")
    if item["smoke"]:
        cmd += " --smoke"
    if evaluation:
        cmd += f" --evaluate --case {case}"
    phase = f"eval{case}" if evaluation else "train"
    family = P["name"]+("/eval" if evaluation else "/smoke" if item["smoke"] else "/train")
    row = dict(project="Thesis", description=f"MC-WM mechanisms {item['name']} {phase}",
        signature=f"{P['name']}/{'smoke' if item['smoke'] else 'runs'}/{item['name']}/{phase}",
        cwd=str(HERE), cmd=cmd, require_node=node, cpu=1,
        ram_mb=2500 if evaluation else 4000 if item["smoke"] else 10000 if node == "node007" else 13000,
        vram=0 if evaluation else 3000 if node == "jtl110gpu" else 1024 if node == "node007" else 1536,
        env=infra.environment(node, evaluation), resource_family=family,
        ram_resource_family=family, vram_resource_family=family+"/"+node,
        skip_launch_staging=True, stage_excludes=["runs", "smoke", "gates"],
        allow_no_ckpt=True, allow_no_resume=True, allow_gpu_over_one_third=True)
    if evaluation and not item["smoke"]:
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
    for node in infra.GPUS:
        infra.sync(node, str(HERE)+"/", remote(node), ["--include=*.py", "--include=protocol.json", "--exclude=*"])
        print("staged "+node, flush=True)


def collect(items):
    nodes = ["node001"] + [n for n in infra.GPUS if not n.startswith("node")
                           and any(i["node"] == n and not gate(i).exists() for i in items)]
    for node in nodes:
        try:
            infra.scheduler.run_on(node, "true", timeout=30, check=True)
            for part in ("runs", "smoke"):
                (HERE/part).mkdir(exist_ok=True)
                result = subprocess.run(["rsync", "-az", "--include=*/", "--include=*.json", "--exclude=*", "-e",
                    infra.scheduler._ssh_rsync_shell_for_node(node),
                    infra.scheduler._ssh_target_for_node(node)+":"+remote(node)+f"/{part}/", str(HERE/part)+"/"],
                    capture_output=True, text=True, timeout=60)
                if result.returncode not in (0, 23):
                    print(f"collection pending {node}: {result.stderr[-300:]}", flush=True)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(f"collection pending {node}: {error}", flush=True)


def watch():
    items = placements()
    summarize = module("factored_collector", HERE/"collect_results.py").summarize
    deadline = time.monotonic()+24*3600
    while time.monotonic() < deadline:
        collect(items)
        for item in items:
            folder = HERE/"runs"/item["name"]
            if gate(item).exists() or not (folder/"results/ready.json").exists():
                continue
            try:
                if not item["node"].startswith("node"):
                    previous.old_launch.relay(item["node"], remote(item["node"], folder), ["policy.pt"],
                                              "node001", remote("node001", folder))
                write(gate(item), item)
            except (RuntimeError, subprocess.SubprocessError) as error:
                print(f"handoff pending {item['name']}: {error}", flush=True)
        result = summarize()
        write(HERE/"summary.json", result)
        print(json.dumps({k:result[k] for k in ("complete", "training_ready", "evaluation_ready")}), flush=True)
        if result["complete"]:
            return
        time.sleep(60)
    raise TimeoutError("Mechanism study remains incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-eval", "submit", "collect", "watch"])
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "smoke":
        submit([task(i) for i in placements(True)], "smoke")
    elif mode == "smoke-eval":
        submit([task(i, 0) for i in placements(True)], "smoke_eval")
    elif mode == "collect":
        collect(placements())
    elif mode == "watch":
        watch()
    else:
        items = placements()
        write(HERE/"locations.json", items)
        submit([task(i) for i in items]+[task(i, c) for i in items for c in range(10)], "campaign")
        submit([dict(project="Thesis", description="MC-WM mechanisms result handoff", signature=P["name"]+"/watch",
            cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1,
            ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], "watch")
