"""Submit adaptive training now, with no dependency on the fixed-TD campaign."""

import argparse
import json
import shlex
import subprocess
import sys
import time

from control_common import HERE, P, CONTEXTS, bundle_path, module, write

base = module("context_control_launcher", HERE.parent / "mcwm_explore_v5/launch.py")
base.HERE, base.P, base.TRAIN_NODES = HERE, P, P["training_nodes"]
infra = base.infra
sys.path.insert(0, str(HERE))


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


base.remote = remote


def located_items(smoke=False):
    items = base.placements(smoke)
    path = HERE / ("smoke_submission.json" if smoke else "cpu_train_submission.json")
    if not path.exists():
        return items
    trains = json.loads(path.read_text())["submitted"]
    trains = [r for r in trains if r["signature"].endswith("/train")]
    records = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "status", "--readonly",
        "--json", "--brief", "--lock-timeout", "30", "--ids", ",".join(r["id"] for r in trains)], text=True))["tasks"]
    nodes = {r["signature"]:r["node"] for r in records}
    for item in items:
        signature = f"{P['name']}/{'smoke' if smoke else 'runs'}/{item['name']}/train"
        item["node"] = nodes.get(signature)
    return items


base.located_items = located_items


def prepare_bundles():
    context = module("control_fit_context", HERE.parent / "mcwm_context_v7/context.py")
    for consumer in P["training_seeds"]:
        old = json.loads((HERE.parent / "mcwm_context_v7/adaptation" / f"seed{consumer}.json").read_text())
        provider = old["provider_seed"]
        assert provider != consumer and old["consumer_seed"] == consumer and not old["test_outcomes_used"]
        labeled = {}
        for scene in CONTEXTS:
            rows = []
            for arm in ("fixed_zero", "fixed_mid", "fixed_original"):
                source = HERE.parent / "mcwm_context_diagnosis_v8/inputs" / scene / f"{arm}_seed{provider}.json"
                row = json.loads(source.read_text())
                assert row["seed"] == provider
                rows.extend([b["features"][:3] for b in row["blocks"]])
            labeled[scene] = rows
        bundle = {key:old[key] for key in ("consumer_seed", "provider_seed", "coefficients", "recognizer")}
        bundle.update(physical_recognizer=context.fit_recognizer(labeled, CONTEXTS, P["context_variance_floor"]),
            global_coefficient={"fixed_zero":0., "fixed_mid":.125, "fixed_original":.25}[old["best_global_fixed_arm"]],
            selection_scope=P["selection"], test_outcomes_used=False)
        write(bundle_path(consumer), bundle)


def task(item, case=None):
    row = base.task(item, case)
    row["description"] = f"Adaptive context {item['name']} "+("train" if case is None else f"eval{case}")
    if case is None:
        row.update(allowed_nodes=infra.CPUS, vram=0, allow_cpu_training=True,
            cpu_training_justification="Measured CPU policy update about 0.22 seconds; run all matched arms on the six 192-core nodes while both GPU nodes are at concurrency limits.")
        family = P["name"] + ("/smoke/cpu" if item["smoke"] else "/train/cpu")
        row.update(resource_family=family, ram_resource_family=family, vram_resource_family=family)
    return row


def stage():
    prepare_bundles()
    primary = base.TRAIN_NODES
    base.TRAIN_NODES = ["node007"]
    try:
        base.stage()
        for node in base.TRAIN_NODES:
            infra.sync(node, HERE.parent / "mcwm_context_filter_v9/filters.py", remote(node, HERE.parent / "mcwm_context_filter_v9"))
            infra.sync(node, str(HERE / "adaptation")+"/", remote(node, HERE / "adaptation"))
    finally:
        base.TRAIN_NODES = primary


def watch():
    summarize = module("context_control_collector", HERE / "collect_results.py").summarize
    deadline = time.monotonic()+48*3600
    while time.monotonic() < deadline:
        try:
            items = base.located_items()
            base.collect(items)
            base.handoff(items)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-500:], flush=True)
        report = summarize()
        write(HERE / "summary.json", report)
        print(json.dumps({k:report[k] for k in ("complete", "training_ready", "evaluation_ready")}), flush=True)
        if report["complete"]:
            return
        time.sleep(60)
    raise TimeoutError("Adaptive context study remains incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-eval", "cpu-train", "collect", "watch"])
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
        base.submit([task(i) for i in items], "cpu_train")
        base.submit([dict(project="Thesis", description="Adaptive context results handoff", signature=P["name"]+"/watch",
            cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1,
            ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], "cpu_watch")
