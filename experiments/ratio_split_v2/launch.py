"""Independent corrected runs; small JSON collection and server-to-server policies."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "enhancement_v1"))
import deploy as infra
sys.path.insert(0, str(HERE.parent / "mcwm_alignment_v2"))
from launch import relay

PREFIX = "thesis-ratio-split-v2"
ARMS = ["mixed_nomodel", "fixed_direct", "fixed_residual", "calibrated_selection"]


def remote(node, path):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


def write(path, value):
    infra.write(path, value)


def locations():
    nodes = ["node007", "jtl110gpu", "jtl110gpu2", "node007", "jtl311linux"]
    return [dict(arm=arm, seed=seed, node=nodes[i], name=f"{arm}_seed{seed}")
            for i, seed in enumerate(infra.P["training_seeds"]) for arm in ARMS]


def job(item, case=None, smoke=False, diagnostic=False):
    evaluation = case is not None
    node = infra.CPUS[((item["seed"] - 22101)*10 + case) % 6] if evaluation else item["node"]
    folder = HERE / ("smoke" if smoke else "runs") / item["name"]
    phase = f"diagnostic{case}" if diagnostic else f"eval{case}" if evaluation else "train"
    args = f"--arm {item['arm']} --seed {item['seed']} --out {remote(node, folder)}"
    script = HERE / "run.py"
    if diagnostic:
        script = HERE.parent / "review_round3/ratio.py"
        args = (f"--mode heldout --seed {item['seed']} --case {case} "
                f"--checkpoint {remote(node, folder / 'ratio_export.pt')} "
                f"--out {remote(node, folder / f'heldout/case{case}')}")
    elif evaluation:
        args += f" --evaluate --case {case}"
    if smoke:
        args += " --smoke"
    resources = infra.resource_spec("mc", item["arm"], node, evaluation=evaluation, smoke=smoke)
    family = f"{PREFIX}/{'diagnostic' if diagnostic else 'eval' if evaluation else 'smoke' if smoke else 'train'}"
    resources.update(resource_family=family, ram_resource_family=family,
                     vram_resource_family=family+"/"+node)
    if diagnostic:
        resources["ram_mb"] = 12000
    task = dict(project="Thesis", description=f"Corrected source split {item['name']} {phase}",
                signature=f"{PREFIX}/{'smoke' if smoke else 'runs'}/{item['name']}/{phase}",
                cwd=str(HERE), cmd=f"{infra.executable(node)} -u {remote(node, script)} {args}",
                require_node=node, cpu=1, env=infra.environment(node, evaluation and not diagnostic),
                skip_launch_staging=True, stage_excludes=["runs", "smoke", "gates"],
                allow_no_ckpt=True, allow_no_resume=True, allow_gpu_over_one_third=True, **resources)
    if evaluation:
        task["wait_for_files"] = [str(HERE / "gates" / (item["name"]+".json"))]
    if diagnostic:
        task.update(allow_cpu_training=True, cpu_training_justification="Frozen classifier and actor diagnostic on fresh source days; no gradient updates.")
    return task


def submit(tasks, name):
    path = HERE / (name+"_jobs.json")
    write(path, tasks)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file",
                                                str(path), "--trusted", "--json"], text=True))
    write(HERE / (name+"_submission.json"), result)
    print(json.dumps(dict(batch=name, submitted=len(result["submitted"]),
                          ids=[t["id"] for t in result["submitted"]])), flush=True)


def stage():
    for node in infra.GPUS:
        infra.sync(node, str(HERE)+"/", remote(node, HERE), ["--include=*.py", "--exclude=*"])
        for file in [HERE.parent / "enhancement_v1/mc.py", HERE.parent / "mcwm_bus/run_control.py"]:
            infra.sync(node, file, remote(node, file.parent))
        print("staged " + node, flush=True)


def collect():
    for node in ["node001", "jtl110gpu", "jtl110gpu2", "jtl311linux"]:
        try:
            # Resolve the working jump route before rsync uses the SSH arguments.
            infra.scheduler.run_on(node, "true", timeout=40, check=True)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(f"collection pending {node}: {error}", flush=True)
            continue
        for part in ["runs", "smoke"]:
            (HERE / part).mkdir(exist_ok=True)
            result = subprocess.run(["rsync", "-az", "--include=*/", "--include=*.json", "--exclude=*", "-e",
                        infra.scheduler._ssh_rsync_shell_for_node(node),
                        infra.scheduler._ssh_target_for_node(node)+":"+remote(node, HERE / part)+"/",
                        str(HERE / part)+"/"], text=True, capture_output=True, timeout=60)
            if result.returncode not in (0, 23):
                print(result.stderr[-500:], flush=True)


def watch():
    deadline = time.monotonic()+6*3600
    while time.monotonic() < deadline:
        collect()
        for item in locations():
            folder = HERE / "runs" / item["name"]
            gate = HERE / "gates" / (item["name"]+".json")
            if gate.exists() or not (folder / "results/ready.json").exists():
                continue
            if not item["node"].startswith("node"):
                try:
                    infra.scheduler.run_on(item["node"], "true", timeout=40, check=True)
                    relay(item["node"], remote(item["node"], folder), ["policy.pt", "ratio_export.pt"],
                          "node001", remote("node001", folder))
                except (RuntimeError, subprocess.SubprocessError) as error:
                    print(f"handoff pending {item['name']}: {error}", flush=True)
                    continue
            write(gate, item)
        counts = dict(training=len(list((HERE / "runs").glob("*/results/ready.json"))),
                      evaluations=len(list((HERE / "runs").glob("*/results/eval_*.json"))),
                      diagnostics=len(list((HERE / "runs").glob("*/heldout/case*/results/ready.json"))))
        write(HERE / "status.json", counts)
        print(json.dumps(counts), flush=True)
        if counts == dict(training=20, evaluations=200, diagnostics=15):
            return
        time.sleep(45)
    raise TimeoutError("Corrected experiment remains incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "submit", "collect", "watch"])
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "smoke":
        submit([job(dict(arm="calibrated_selection", seed=22000, node="node007", name="calibrated_selection_seed22000"), smoke=True)], "smoke")
    elif mode == "collect":
        collect()
    elif mode == "watch":
        watch()
    else:
        items = locations()
        write(HERE / "locations.json", items)
        jobs = [job(i) for i in items] + [job(i, c) for i in items for c in range(10)]
        jobs += [job(i, c, diagnostic=True) for i in items if i["arm"] == "mixed_nomodel" for c in range(3)]
        submit(jobs, "campaign")
        submit([dict(project="Thesis", description="Corrected split results and policy handoff", signature=PREFIX+"/watch",
                     cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1,
                     ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], "watch")
