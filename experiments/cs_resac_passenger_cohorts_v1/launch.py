"""Stage frozen inputs server-to-server and dispatch independent CPU evaluations."""

import argparse
from itertools import product
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
EXPERIMENTS = HERE.parent
SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"
sys.path.insert(0, str(Path(SCHEDULER).parent))
import scheduler

HPC = "/home/zhengliang01/scheduleurm_work"
HPC_PYTHON = HPC + "/conda_envs/csbapr-gpu-py310/bin/python"
STORES = ("node007", "jtl110gpu", "jtl110gpu2", "jtl311linux")


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", HPC) if node.startswith("node") else str(HERE)


def ssh(node, command):
    return shlex.split(scheduler._ssh_rsync_shell_for_node(node)) + [
        scheduler._ssh_target_for_node(node), command
    ]


def copy_between(source_node, destination_node, folder):
    sender = subprocess.Popen(ssh(source_node,
        f"tar -C {shlex.quote(root(source_node))} -czf - {folder}"), stdout=subprocess.PIPE)
    try:
        receiver = subprocess.run(ssh(destination_node,
            f"tar -C {shlex.quote(root(destination_node))} -xzf -"),
            stdin=sender.stdout, timeout=180, check=True)
    finally:
        sender.stdout.close()
    if sender.wait(timeout=30):
        raise RuntimeError("Server input transfer failed")
    return receiver.returncode


def sync(node, source, destination):
    subprocess.run(["rsync", "-az", "--timeout=30", "-e",
                    scheduler._ssh_rsync_shell_for_node(node), str(source),
                    scheduler._ssh_target_for_node(node) + ":" + destination], check=True, timeout=90)


def stage(code_only=False):
    for node in STORES:
        scheduler.run_on(node, "mkdir -p " + shlex.quote(root(node)), timeout=45, check=True)
        for path in sorted(HERE.glob("*.py")) + [HERE / "protocol.json"]:
            sync(node, path, root(node) + "/")
        key = "node007" if node.startswith("node") else node
        deployment = json.loads((EXPERIMENTS / f"mcwm_bus/deployment_{key}.json").read_text())
        runtime = dict(python=HPC_PYTHON if node.startswith("node") else deployment["python"],
                       env=deployment["env"])
        local = HERE / "deployments" / key / "node_runtime.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(runtime, indent=2) + "\n")
        sync(node, local, root(node) + "/node_runtime.json")
        bootstrap = local.parent / "boot.sh"
        bootstrap.write_text(
            "#!/bin/sh\nexec " + shlex.quote(runtime["python"]) +
            ' -u "$(dirname "$0")/node_run.py" "$@"\n'
        )
        sync(node, bootstrap, root(node) + "/boot.sh")
    if code_only:
        print("Updated evaluation code and each store's interpreter bootstrap", flush=True)
        return
    code = """from pathlib import Path
import json,shutil
root=Path(ROOT)
experiments=root.parent
source=experiments/'enhancement_v1/runtime/cs'
out=root/'inputs'
out.mkdir(exist_ok=True)
for relative in ['CS-BAPR/csbapr','offline-sumo/env/sim_core','offline-sumo/env/calibrated_env/data']:
 shutil.copytree(source/relative,out/'runtime'/relative,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','_line_envs'))
target=out/'runtime/CS-BAPR/scripts'
target.mkdir(parents=True,exist_ok=True)
shutil.copy2(source/'CS-BAPR/scripts/test_multiline_convergence.py',target)
shutil.copy2(source/'offline-sumo/env/calibrated_env/config.json',out/'runtime/offline-sumo/env/calibrated_env/config.json')
shutil.copy2(experiments/'legacy_bus/runtime/BAPR/bus_experiments/frozen_policy_bank_core.py',out/'model_core.py')
p=json.loads((root/'protocol.json').read_text())
(out/'policies').mkdir(exist_ok=True)
for method in p['methods']:
 for seed in p['training_seeds']:
  path=experiments/p['checkpoints'][method].format(seed=seed)
  shutil.copy2(path,out/'policies'/f'{method}_seed{seed}.pt')
print('Copied frozen runtime and four actors inside the new server-only directory')
""".replace("ROOT", repr(root("node007")))
    scheduler.run_on("node007", HPC_PYTHON + " -c " + shlex.quote(code), timeout=90, check=True)
    for node in STORES[1:]:
        copy_between("node007", node, "inputs")
    print("Staged the shared seven-node store and three independent Linux stores", flush=True)


def task(method, seed, day, scene, phase="evaluate", node=None):
    label = f"{phase}/{method}/{seed}/{day}/{scene}" + (f"/{node}" if node else "")
    fields = f"--phase {phase} --method {method} --seed {seed} --evaluation-seed {day} --scene {scene}"
    return dict(project="Thesis", description=P["name"] + " " + label,
                signature=P["name"] + "/" + label, cwd=str(HERE),
                cmd=f"bash {HERE}/boot.sh {fields}",
                allowed_nodes=[node] if node else P["nodes"], cpu=1,
                ram_mb=850 if phase == "preflight" else 1500, vram=0,
                resource_family=P["name"] + "/eval", allow_no_ckpt=True,
                allow_no_resume=True, skip_launch_staging=True,
                env=dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))


def submit(phase):
    output = HERE / (phase + "_submission.json")
    if output.exists():
        raise ValueError("This batch has already been submitted")
    if phase == "preflight":
        nodes = ("node001", *STORES[1:])
        rows = [task(method, 64001, 78101, "normal" if method == "resac" else "burst50",
                     phase, node) for node, method in product(nodes, P["methods"])]
    else:
        rows = [task(method, seed, day, scene) for method, seed, day, scene in product(
            P["methods"], P["training_seeds"], P["evaluation_seeds"], P["scenes"])]
    jobs = HERE / (phase + "_jobs.json")
    jobs.write_text(json.dumps(rows, indent=2) + "\n")
    result = json.loads(subprocess.check_output(["python3", SCHEDULER, "submit-jsonl", "--file",
                                               str(jobs), "--trusted", "--json"], text=True))
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(phase=phase, submitted=len(result["submitted"]),
                          ids=[t["id"] for t in result["submitted"]])))


def collect():
    for node in STORES[1:]:
        ret = scheduler.run_on(node, "test -d " + shlex.quote(root(node) + "/evaluate"),
                               timeout=45, check=False)
        if ret[0] == 0:
            copy_between(node, "node007", "evaluate")
    scheduler.run_on("node007", HPC_PYTHON + " " + shlex.quote(root("node007") + "/collect_results.py"),
                     timeout=90, check=True)
    subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
                    scheduler._ssh_target_for_node("node007") + ":" + root("node007") + "/analysis_summary.json",
                    str(HERE) + "/"], check=True, timeout=60)
    report = json.loads((HERE / "analysis_summary.json").read_text())
    print(json.dumps({k: report[k] for k in ("evaluations", "expected", "complete", "maximum_wall_seconds", "maximum_rss_mb")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "stage-code", "preflight", "submit", "collect"))
    args = parser.parse_args()
    if args.mode in ("stage", "stage-code"):
        stage(code_only=args.mode == "stage-code")
    elif args.mode == "collect":
        collect()
    else:
        submit("evaluate" if args.mode == "submit" else "preflight")
