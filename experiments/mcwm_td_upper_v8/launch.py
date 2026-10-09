"""Unpinned, same-GPU-family training and independently sharded CPU days."""

import argparse
import json
import shlex
import subprocess
import sys
import time

from td_common import HERE, P, module, write
base = module("td_upper_launcher", HERE.parent / "mcwm_explore_v5/launch.py")
base.HERE, base.P, base.TRAIN_NODES = HERE, P, P["training_nodes"]
infra = base.infra
sys.path.insert(0, str(HERE))


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


base.remote = remote


def task(item, case=None):
    r = base.task(item, case)
    r["description"] = f"Fixed TD upper {item['name']} "+("train" if case is None else f"eval{case}")
    if item["smoke"] and case is None:
        r.update(allowed_nodes=infra.CPUS, vram=0, allow_cpu_training=True)
    return r


def stage():
    primary = base.TRAIN_NODES
    base.TRAIN_NODES = ["node007"]+primary
    try:
        base.stage()
        for node in base.TRAIN_NODES:
            path = remote(node, HERE.parent / "mcwm_alignment_v2/inputs/h2oplus_seed6203.pt")
            infra.scheduler.run_on(node, "test -s "+shlex.quote(path), timeout=35, check=True)
    finally:
        base.TRAIN_NODES = primary


def watch():
    summarize = module("td_upper_collector", HERE / "collect_results.py").summarize
    end = time.monotonic()+48*3600
    while time.monotonic() < end:
        try:
            items = base.located_items()
            base.collect(items)
            base.handoff(items)
        except (RuntimeError, subprocess.SubprocessError) as e:
            print(str(e)[-500:], flush=True)
        r = summarize()
        write(HERE / "summary.json", r)
        print(json.dumps({k:r[k] for k in ["complete", "training_ready", "evaluation_ready"]}), flush=True)
        if r["complete"]:
            return
        time.sleep(60)
    raise TimeoutError("Fixed TD study remains incomplete")


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
        base.collect(items)
        base.handoff(items)
        assert all(base.gate(i).exists() for i in items)
        base.submit([task(i, 10) for i in items], "smoke_eval")
    elif mode == "collect":
        items = base.located_items()
        base.collect(items)
        base.handoff(items)
    elif mode == "watch":
        watch()
    else:
        items = base.placements()
        write(HERE / "locations.json", items)
        base.submit([task(i) for i in items]+[task(i,c) for i in items for c in range(20)], "campaign")
        base.submit([dict(project="Thesis", description="Fixed TD result handoff", signature=P["name"]+"/watch",
            cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1,
            ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], "watch")
