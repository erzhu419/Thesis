"""Cross-fitted stages with unpinned GPU training and independent CPU days."""

import argparse
import json
import subprocess
import sys
import time

from context_common import HERE, P, FIXED, CONTEXTS, cases, bundle_path, module, write

base = module("context_launcher_base", HERE.parent / "mcwm_explore_v5/launch.py")
base.HERE, base.P = HERE, P
infra = base.infra
sys.path.insert(0, str(HERE))


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


base.remote = remote


def task(item, case=None):
    row = base.task(item, case)
    row["description"] = f"MC-WM context {item['name']} "+("train" if case is None else f"eval{case}")
    if item["smoke"] and case is None:
        row.update(allowed_nodes=infra.CPUS, vram=0, allow_cpu_training=True)
    elif case is None and item["arm"] not in FIXED:
        row["wait_for_files"] = [str(HERE / "gates" / f"adaptation_seed{item['seed']}.json")]
    return row


def stage():
    base.stage()
    synthetic = dict(consumer_seed=6201, provider_seed=6202, smoke_only=True,
        coefficients=dict(zip(CONTEXTS, [0., .125, .25])),
        recognizer=dict(classes=CONTEXTS, mean=[0.]*8, scale=[1.]*8,
            centers=[[0.]*8 for _ in CONTEXTS], variance=[1.]*8))
    write(bundle_path(6201, True), synthetic)
    for node in base.TRAIN_NODES:
        infra.sync(node, bundle_path(6201, True), remote(node))


def distribute_bundles():
    from selection import prepare
    prepare()
    for seed in P["training_seeds"]:
        gate = HERE / "gates" / f"adaptation_seed{seed}.json"
        if gate.exists() or not bundle_path(seed).exists():
            continue
        for node in base.TRAIN_NODES:
            infra.sync(node, bundle_path(seed), remote(node, HERE / "adaptation"))
        write(gate, dict(seed=seed, provider_seed=json.loads(bundle_path(seed).read_text())["provider_seed"],
            distributed_nodes=base.TRAIN_NODES))


def watch():
    summarize = module("context_collector", HERE / "collect_results.py").summarize
    deadline = time.monotonic()+48*3600
    while time.monotonic() < deadline:
        try:
            items = base.located_items()
            base.collect(items)
            base.handoff(items)
            distribute_bundles()
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-500:], flush=True)
        report = summarize()
        write(HERE / "summary.json", report)
        print(json.dumps({k: report[k] for k in ("complete", "training_ready", "evaluation_ready")}), flush=True)
        if report["complete"]:
            return
        time.sleep(60)
    raise TimeoutError("Context study remains incomplete")


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
        distribute_bundles()
    elif mode == "watch":
        watch()
    else:
        items = base.placements()
        write(HERE / "locations.json", items)
        rows = [task(i) for i in items]+[task(i, c) for i in items for c in cases(i["arm"])]
        base.submit(rows, "campaign")
        base.submit([dict(project="Thesis", description="Context cross-fit and results handoff",
            signature=P["name"]+"/watch", cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch",
            require_node="local", cpu=1, ram_mb=512, vram=0, allow_no_ckpt=True,
            allow_no_resume=True, skip_launch_staging=True)], "watch")
