"""Export small frozen classifiers on their training hosts, then release CPU diagnostics."""

import argparse
import json
import time
import shlex
from launch import HERE, P, infra, remote, old_launch, submit, cpu_task, write, collect


def gate(seed):
    return HERE / "gates" / f"ratio_holdout_seed{seed}.json"


def jobs():
    result = []
    for i, seed in enumerate(P["ratio"]["enhancement_seeds"]):
        checkpoint = remote("node001", HERE / f"runs/ratio/enhancement_seed{seed}/ratio_export.pt")
        for case in range(3):
            row = cpu_task("ratio", f"ratio_heldout/seed{seed}/case{case}",
                          f"--mode heldout --seed {seed} --case {case} --checkpoint {checkpoint}", i*3+case)
            row["env"] = infra.environment("node001")
            row["ram_mb"] = 12000
            row["wait_for_files"] = [str(gate(seed))]
            result.append(row)
    return result


def watch():
    items = [i for i in json.loads((HERE / "locations.json").read_text()) if i["kind"] == "ratio"]
    deadline = time.monotonic()+72*3600
    while time.monotonic() < deadline:
        collect()
        for item in items:
            if gate(item["seed"]).exists():
                continue
            folder = HERE / "runs" / item["name"]
            if not (folder / "results/ready.json").exists():
                continue
            node = item["node"]
            env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in infra.environment(node).items())
            command = (f"{env} {infra.executable(node)} {remote(node)}/ratio.py --mode export "
                       f"--out {remote(node,folder)} --checkpoint {remote(node,folder)}/ratio_checkpoint.pt")
            infra.scheduler.run_on(node, command, timeout=180, check=True)
            if not node.startswith("node"):
                old_launch.relay(node, remote(node, folder), ["ratio_export.pt"], "node001", remote("node001",folder))
            write(gate(item["seed"]), dict(ready=True, gradient_updates_in_diagnostics=0))
        completed = list((HERE / "runs/ratio_heldout").glob("seed*/case*/results/ready.json"))
        if len(completed) == 15:
            return
        time.sleep(90)
    raise TimeoutError("Classifier holdout evaluation remains incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["submit", "watch"])
    args = parser.parse_args()
    if args.mode == "watch":
        watch()
    else:
        submit(jobs(), "ratio_holdout")
        submit([dict(project="Thesis", description="Frozen ratio export and independent source holdout handoff",
                     signature=P["name"]+"/ratio_handoff", cwd=str(HERE), require_node="local",
                     cmd=f"python3 -u {HERE}/handoff_ratio.py watch", cpu=1, ram_mb=512, vram=0,
                     skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True)], "ratio_handoff")
