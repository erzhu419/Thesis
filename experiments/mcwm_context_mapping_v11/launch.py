"""Fresh CPU controls; all training jobs and per-day SUMO shards are unpinned."""

import argparse
import json
import shlex
import subprocess
import time

from mapping_common import HERE, P, CONTEXTS, bundle_path, module, write

base = module("mapping_launcher", HERE.parent / "mcwm_explore_v5/launch.py")
base.HERE, base.P, base.TRAIN_NODES = HERE, P, P["training_nodes"]
infra = base.infra


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


base.remote = remote


def task(item, case=None):
    row = base.task(item, case)
    row["description"] = f"Context mapping {item['name']} "+("train" if case is None else f"eval{case}")
    if case is None:
        row.update(vram=0, allow_cpu_training=True,
            cpu_training_justification="Matched single-thread v10 CPU learner; 30 independent controls distributed over six 192-core nodes.")
    return row


def stage():
    for seed in P["training_seeds"]:
        original = json.loads((HERE.parent / "mcwm_context_control_v10/adaptation" / f"seed{seed}.json").read_text())
        assert original["consumer_seed"] == seed and original["provider_seed"] != seed
        write(bundle_path(seed), original)
    node = "node007"
    infra.sync(node, str(HERE)+"/", remote(node), ["--include=*.py", "--include=protocol.json", "--exclude=*"])
    infra.sync(node, str(HERE / "adaptation")+"/", remote(node, HERE / "adaptation"))
    runtime = dict(train=dict(python=infra.executable(node), env=infra.environment(node)),
        evaluate=dict(python=infra.executable("node001"), env=infra.environment("node001", True)))
    path = HERE / "deployments/node007/node_runtime.json"
    write(path, runtime)
    infra.sync(node, path, remote(node))
    command = "ln -sfn "+shlex.quote(infra.executable(node))+" "+shlex.quote(remote(node)+"/node_python")
    infra.scheduler.run_on(node, command, timeout=40, check=True)
    paths = [HERE.parent / "mcwm_alignment_v2/inputs" / f"h2oplus_seed{seed}.pt" for seed in P["training_seeds"]]
    paths += [HERE.parent / folder / file for folder, file in
        (("mcwm_context_control_v10", "run.py"), ("mcwm_context_filter_v9", "filters.py"), ("mcwm_context_v7", "run.py"))]
    infra.scheduler.run_on(node, "test "+" -a ".join("-s "+shlex.quote(remote(node, p)) for p in paths), timeout=40, check=True)
    print("Staged isolated code; shared CPU storage, no datasets or checkpoints transferred", flush=True)


def collect(items):
    node = next((i["node"] for i in items if i["node"] and i["node"].startswith("node")), "node001")
    infra.scheduler.run_on(node, "true", timeout=30, check=True)
    for part in sorted({"smoke" if i["smoke"] else "runs" for i in items}):
        (HERE / part).mkdir(exist_ok=True)
        result = subprocess.run(["rsync", "-az", "--exclude=evaluation/", "--exclude=models/",
            "--exclude=context_blocks.json", "--include=*/", "--include=*.json", "--exclude=*", "-e",
            infra.scheduler._ssh_rsync_shell_for_node(node),
            infra.scheduler._ssh_target_for_node(node)+":"+remote(node)+f"/{part}/", str(HERE / part)+"/"],
            capture_output=True, text=True, timeout=60)
        if result.returncode not in (0, 23):
            raise RuntimeError("JSON collection failed: "+result.stderr[-300:])


def watch():
    summarize = module("mapping_collector", HERE / "collect_results.py").summarize
    deadline = time.monotonic()+48*3600
    while time.monotonic() < deadline:
        try:
            items = base.located_items()
            collect(items)
            base.handoff(items)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-500:], flush=True)
        report = summarize()
        write(HERE / "summary.json", report)
        print(json.dumps({k:report[k] for k in ("complete", "training_ready", "evaluation_ready")}), flush=True)
        if report["complete"]:
            return
        time.sleep(60)
    raise TimeoutError("Context mapping experiment remains incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-eval", "submit", "collect", "watch"])
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "smoke":
        base.submit([task(i) for i in base.placements(True)], "smoke")
    elif mode == "smoke-eval":
        items = base.located_items(True)
        collect(items)
        base.handoff(items)
        assert all(base.gate(i).exists() for i in items)
        base.submit([task(i, 10) for i in items], "smoke_eval")
    elif mode == "collect":
        items = base.located_items()
        collect(items)
        base.handoff(items)
    elif mode == "watch":
        watch()
    else:
        items = base.placements()
        write(HERE / "locations.json", items)
        base.submit([task(i) for i in items]+[task(i, c) for i in items for c in range(10, 20)], "campaign")
        base.submit([dict(project="Thesis", description="Context mapping result handoff", signature=P["name"]+"/watch",
            cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1,
            ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], "watch")
