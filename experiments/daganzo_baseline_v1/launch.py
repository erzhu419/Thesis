"""CPU-only rule evaluations on the six existing shared-storage nodes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcwm_usage_v3"))
from usage_common import module, write

infra = module("daganzo_infra", HERE.parent / "enhancement_v1/deploy.py")
P = json.loads((HERE / "protocol.json").read_text())


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def jobs(smoke):
    rows = []
    for method in P["methods"]:
        for layer in P["execution_layers"]:
            for case in ([0] if smoke else range(10)):
                node = infra.CPUS[len(rows) % len(infra.CPUS)]
                phase = "smoke" if smoke else "runs"
                label = f"{phase}/{method}/{layer}/case{case:02d}"
                command = (f"{infra.executable(node)} -u {remote()}/run.py "
                           f"--method {method} --layer {layer} --case {case}" + (" --smoke" if smoke else ""))
                family = P["name"] + "/" + phase
                rows.append(dict(project="Thesis", description="Daganzo reference " + label,
                    signature=P["name"] + "/" + label, cwd=str(HERE), cmd=command,
                    require_node=node, cpu=1, ram_mb=2500, vram=0,
                    env=infra.environment(node, evaluation=True),
                    resource_family=family, ram_resource_family=family,
                    skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True))
    return rows


def submit(smoke):
    name = "smoke" if smoke else "campaign"
    submission = HERE / (name + "_submission.json")
    if submission.exists():
        raise ValueError(f"Already submitted: {submission}")
    path = HERE / (name + "_jobs.json")
    write(path, jobs(smoke))
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl",
        "--file", str(path), "--trusted", "--json"], text=True))
    write(submission, result)
    print(json.dumps(dict(batch=name, ids=[r["id"] for r in result["submitted"]])), flush=True)


def collect():
    node = "node001"
    infra.scheduler.run_on(node, "true", timeout=30, check=True)
    for part in ("smoke", "runs"):
        (HERE / part).mkdir(exist_ok=True)
        result = subprocess.run(["rsync", "-az", "--include=*/", "--include=*.json", "--exclude=*", "-e",
            infra.scheduler._ssh_rsync_shell_for_node(node),
            infra.scheduler._ssh_target_for_node(node) + ":" + remote() + f"/{part}/",
            str(HERE / part) + "/"], capture_output=True, text=True, timeout=60)
        if result.returncode not in (0, 23):
            raise RuntimeError(result.stderr)
        print(f"collected {part}: rsync status {result.returncode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "smoke", "submit", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        infra.sync("node001", str(HERE) + "/", remote(),
                   ["--include=*.py", "--include=protocol.json", "--exclude=*"])
    elif mode == "collect":
        collect()
    else:
        submit(mode == "smoke")
