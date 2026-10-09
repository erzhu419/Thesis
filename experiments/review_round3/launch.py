"""Small-result collection and independent Linux placements for review experiments."""

from review_common import HERE, EXPERIMENTS, P, module, write
import argparse
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(EXPERIMENTS / "enhancement_v1"))
import deploy as infra
old_launch = module("round3_old_launch", EXPERIMENTS / "mcwm_alignment_v2/launch.py")
GPUS = ["node007", "jtl110gpu", "jtl110gpu2", "jtl311linux"]
PREFIX = P["name"]


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


def stage():
    for node in GPUS:
        infra.sync(node, str(HERE)+"/", remote(node),
                   ["--include=*.py", "--include=protocol.json", "--exclude=*"])
        source = remote(node, EXPERIMENTS / "mcwm_alignment_v2/inputs")
        destination = remote(node, HERE / "inputs")
        infra.scheduler.run_on(node, f"mkdir -p {destination} && cp -n {source}/h2oplus_seed*.pt {destination}/",
                               timeout=60, check=True)
        print("staged " + node, flush=True)


def placements():
    items = []
    # Three arms sharing a training seed and scenario always share the same host.
    nodes = ["node007", "jtl110gpu", "jtl110gpu2",
             "node007", "jtl110gpu", "jtl110gpu2",
             "node007", "jtl110gpu", "jtl110gpu2"]
    for i, seed in enumerate(P["mcwm"]["seeds"]):
        for j, scenario in enumerate(P["mcwm"]["scenarios"]):
            for arm in P["mcwm"]["arms"]:
                items.append(dict(kind="mcwm", seed=seed, scenario=scenario, arm=arm,
                                  node=nodes[i*3+j], name=f"mcwm/{scenario}/{arm}_seed{seed}", evaluations=10))
    for i, seed in enumerate(P["ensemble"]["seeds"]):
        for size in P["ensemble"]["sizes"]:
            node = ["node007", "node007", "jtl110gpu", "jtl110gpu2", "jtl311linux"][i]
            items.append(dict(kind="legacy", seed=seed, size=size, arm="ensemble", node=node,
                              name=f"ensemble/k{size}_seed{seed}", evaluations=6))
    for i, seed in enumerate(P["ratio"]["enhancement_seeds"]):
        items.append(dict(kind="ratio", seed=seed, mode="enhancement", node=GPUS[i % len(GPUS)],
                          name=f"ratio/enhancement_seed{seed}", evaluations=0))
    return items


def gate(item):
    return HERE / "gates" / (item["name"].replace("/", "_")+".json")


def task(item, case=None, smoke=False):
    evaluation = case is not None
    node = item["node"] if not evaluation else infra.CPUS[case % len(infra.CPUS)]
    folder = HERE / ("smoke" if smoke else "runs") / item["name"]
    args = f"--seed {item['seed']} --out {remote(node, folder)}"
    if item["kind"] == "mcwm":
        args += f" --arm {item['arm']} --scenario {item['scenario']}"
    elif item["kind"] == "legacy":
        args += f" --size {item['size']}"
    elif item["kind"] == "ratio":
        args += f" --mode {item['mode']}"
    if evaluation:
        args += f" --case {case}"
        if item["kind"] == "mcwm":
            args += " --evaluate"
    if smoke:
        args += " --smoke"
    phase = f"eval{case}" if evaluation else "train"
    family = f"{PREFIX}/{item['kind']}/{phase if evaluation else 'train'}"
    row = dict(project="Thesis", description=f"Review R3 {item['name']} {phase}",
               signature=f"{PREFIX}/{'smoke' if smoke else 'runs'}/{item['name']}/{phase}",
               cwd=str(HERE), cmd=f"{infra.executable(node)} -u {remote(node)}/{item['kind']}.py {args}",
               require_node=node, cpu=1, ram_mb=2500 if evaluation else 16000 if item["kind"] != "legacy" else 4000,
               vram=0 if evaluation else (1600 if item["kind"] == "legacy" else 2200) if node == "jtl110gpu" else 1024,
               env=infra.environment(node, evaluation), resource_family=family,
               vram_resource_family=family+"/"+node, ram_resource_family=family,
               skip_launch_staging=True, stage_excludes=["runs", "smoke", "inputs", "gates"],
               allow_no_ckpt=True, allow_no_resume=True, allow_gpu_over_one_third=True)
    if evaluation and not smoke:
        row["wait_for_files"] = [str(gate(item))]
    return row


def cpu_task(kind, name, arguments, index, *, smoke=False):
    node = infra.CPUS[index % len(infra.CPUS)]
    folder = HERE / ("smoke" if smoke else "runs") / name
    cmd = (f"{infra.executable(node)} -u {remote(node)}/{kind}.py "
           f"{arguments} --out {remote(node, folder)}" + (" --smoke" if smoke else ""))
    return dict(project="Thesis", description=f"Review R3 {name}",
                signature=f"{PREFIX}/{'smoke' if smoke else 'runs'}/{name}", cwd=str(HERE), cmd=cmd,
                require_node=node, cpu=1, ram_mb=3500, vram=0, env=infra.environment(node, True),
                resource_family=f"{PREFIX}/{kind}/cpu", skip_launch_staging=True,
                stage_excludes=["runs", "smoke", "inputs", "gates"], allow_no_ckpt=True, allow_no_resume=True,
                allow_cpu_training=True,
                cpu_training_justification="Frozen-policy service-day evaluation or saved classifier diagnostics; no policy gradient training.")


def cpu_tasks(smoke=False):
    rows = []
    for kind, count in [("guard", 72), ("response", 54)]:
        cases = [0, 27] if smoke and kind == "response" else [0] if smoke else range(count)
        for case in cases:
            rows.append(cpu_task(kind, f"{kind}/case{case:03d}", f"--case {case}", len(rows), smoke=smoke))
    for arm in P["reward"]["arms"][:1] if smoke else P["reward"]["arms"]:
        for seed in P["reward"]["training_seeds"][:1] if smoke else P["reward"]["training_seeds"]:
            controller = remote("node001", EXPERIMENTS / f"legacy_bus/runs/reward_v1/{arm}_seed{seed}/bundle/controller.pt")
            for case in range(1 if smoke else 6):
                rows.append(cpu_task("legacy", f"reward/{arm}_seed{seed}/case{case:02d}",
                            f"--arm {arm} --seed {seed} --case {case} --controller {controller}", len(rows), smoke=smoke))
    if not smoke:
        folder = EXPERIMENTS / "mcwm_bus/runs/campaign_v3/blocks6_calibrated_h2o"
        checkpoint = remote("node001", folder / "checkpoints/latest.pt")
        dataset = remote("node001", folder / "offline_view.h5")
        row = cpu_task("ratio", "ratio/short_budget", f"--mode short --checkpoint {checkpoint} --dataset {dataset}", len(rows))
        row["env"] = infra.environment("node001")
        row["ram_mb"] = 12000
        rows.append(row)
    return rows


def submit(rows, name):
    file = HERE / f"{name}_jobs.json"
    write(file, rows)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl",
                                                "--file", str(file), "--trusted", "--json"], text=True))
    write(HERE / f"{name}_submission.json", result)
    print(json.dumps(dict(batch=name, submitted=len(result["submitted"]),
                          ids=[r["id"] for r in result["submitted"]])), flush=True)


def collect():
    for node in ["node001", "jtl110gpu", "jtl110gpu2", "jtl311linux"]:
        for directory in ["runs", "smoke"]:
            target = HERE / directory
            target.mkdir(exist_ok=True)
            result = subprocess.run(["rsync", "-az", "--include=*/", "--include=*.json", "--exclude=*",
                                     "-e", infra.scheduler._ssh_rsync_shell_for_node(node),
                                     infra.scheduler._ssh_target_for_node(node)+":"+remote(node)+f"/{directory}/",
                                     str(target)+"/"], text=True, capture_output=True)
            if result.returncode not in (0, 23):
                print(json.dumps(dict(collect_error=node, detail=result.stderr[-300:])), flush=True)


def watch():
    items = json.loads((HERE / "locations.json").read_text())
    deadline = time.monotonic()+72*3600
    while time.monotonic() < deadline:
        collect()
        for item in items:
            folder = HERE / "runs" / item["name"]
            if not item["evaluations"] or gate(item).exists() or not (folder / "results/ready.json").exists():
                continue
            if not item["node"].startswith("node"):
                files = ["bundle/controller.pt"] if item["kind"] == "legacy" else ["policy.pt"]
                old_launch.relay(item["node"], remote(item["node"], folder), files,
                                 "node001", remote("node001", folder))
            write(gate(item), dict(ready=True, node=item["node"]))
        complete = sum((HERE / "runs" / i["name"] / "results/ready.json").exists() for i in items)
        evaluations = sum(len(list((HERE / "runs" / i["name"] / "results").glob("eval*.json"))) for i in items)
        write(HERE / "status.json", dict(training_ready=complete, training_expected=len(items),
              evaluations_ready=evaluations, evaluations_expected=sum(i["evaluations"] for i in items)))
        if complete == len(items) and evaluations == sum(i["evaluations"] for i in items):
            return
        time.sleep(60)
    raise TimeoutError("Review supplements have not all completed; inspect tasks and preserved outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "cpu-smoke", "cpu-submit", "submit", "collect", "watch"])
    args = parser.parse_args()
    if args.mode == "stage":
        stage()
    elif args.mode == "collect":
        collect()
    elif args.mode == "watch":
        watch()
    elif args.mode in ("cpu-smoke", "cpu-submit"):
        submit(cpu_tasks(args.mode == "cpu-smoke"), args.mode)
    elif args.mode == "smoke":
        items = [dict(kind="mcwm", seed=6201, scenario="calibrated", arm=arm, node="node007", name="mcwm/"+arm)
                 for arm in P["mcwm"]["arms"]]
        items.append(dict(kind="legacy", seed=32101, size=20, arm="ensemble", node="jtl110gpu2", name="legacy/k20"))
        items.append(dict(kind="ratio", seed=22101, mode="enhancement", node="jtl110gpu", name="ratio/enhancement"))
        submit([task(i, smoke=True) for i in items], "smoke")
    else:
        items = placements()
        write(HERE / "locations.json", items)
        submit([task(i) for i in items] + [task(i, case) for i in items for case in range(i["evaluations"])], "training")
        watcher = dict(project="Thesis", description="Review R3 small-result collection and policy handoff",
                       signature=PREFIX+"/watch", cwd=str(HERE),
                       cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1, ram_mb=512, vram=0,
                       allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)
        submit([watcher], "watch")


if __name__ == "__main__":
    main()
    print("DONE review supplement", flush=True)
