"""Stage feedback-only changes and dispatch independent all-node CPU evaluations."""

import argparse
from collections import Counter
import importlib.util
from itertools import product
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("temperature_deployment", HERE.parent / "cs_source_anchor_v1/launch.py")
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)
deployment.HERE, deployment.P = HERE, P
shared, scheduler = deployment.shared, deployment.scheduler


def stage():
    records = []
    for node in deployment.STORES:
        destination = deployment.root(node)
        scheduler.run_on(node, "mkdir -p " + shlex.quote(destination), timeout=45, check=True)
        folder = HERE / "deployments" / node
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["run.py", "collect.py", "protocol.json"]:
            shutil.copy2(HERE / name, folder / name)
        (folder / "boot.sh").write_text("#!/bin/sh\nexec " + shlex.quote(shared.python(node)) + ' -u "$(dirname "$0")/run.py" "$@"\n')
        subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
            str(folder) + "/", scheduler._ssh_target_for_node(node) + ":" + destination + "/"], check=True, timeout=60)
        command = shared.python(node) + " " + shlex.quote(destination + "/run.py") + " contracts"
        result = json.loads(scheduler.run_on(node, command, timeout=60, check=True)[1])
        records.append(dict(node=node, **result))
        print("Feedback contracts passed: " + node, flush=True)
    deployment.write(HERE / "deployment_checks.json", dict(passed=True, stores=records))


def smoke():
    assert json.loads((HERE / "deployment_checks.json").read_text())["passed"]
    rows = [deployment.task("smoke/identity", "smoke --variant identity --seed 64001 --domain target --scene normal --day 78101")]
    rows += [deployment.task("smoke/" + variant, f"smoke --variant {variant} --seed 64002 --domain target --scene static10 --day 78101") for variant in P["variants"]]
    deployment.submit_rows(rows, "smoke")


def smoke_report():
    rows = [r for r in deployment.tasks() if "/smoke/" in r["signature"]]
    assert len(rows) == 4 and all(r["status"] == "done" for r in rows)
    for node in deployment.STORES[1:]:
        if scheduler.run_on(node, "test -d " + shlex.quote(deployment.root(node) + "/smoke"), timeout=30, check=False)[0] == 0:
            deployment.relay(node, "node007", "smoke")
    result = scheduler.run_on("node007", shared.python("node007") + " " + shlex.quote(deployment.root("node007") + "/collect.py") + " smoke-report", timeout=60, check=True)[1]
    deployment.write(HERE / "smoke_passed.json", json.loads(result))
    print(result, flush=True)


def submit():
    assert json.loads((HERE / "smoke_passed.json").read_text())["passed"]
    rows = []
    for domain in ["source", "target"]:
        for variant, seed, scene, day in product(P["variants"], P["training_seeds"], P[domain + "_scenes"], P[domain + "_days"]):
            label = f"eval/{domain}/{variant}/seed{seed}/{scene}/day{day}"
            command = f"eval --variant {variant} --seed {seed} --domain {domain} --scene {scene} --day {day}"
            rows.append(deployment.task(label, command))
    assert len(rows) == 204
    rows.append(dict(project="Thesis", signature=P["name"] + "/watch", description=P["name"] + " server analysis",
        cwd=str(HERE), cmd=f"{sys.executable} -u {HERE}/launch.py watch", require_node="local", cpu=1, ram_mb=256,
        vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True,
        allow_cpu_training=True, cpu_training_justification="CPU-only server result collection, no learning."))
    deployment.submit_rows(rows, "formal")


def watch():
    previous_counts = None
    while True:
        rows = [r for r in deployment.tasks() if "/eval/" in r["signature"]]
        counts = dict(Counter(r["status"] for r in rows))
        if counts != previous_counts:
            deployment.write(HERE / "handoff_status.json", dict(status_counts=counts))
            print(json.dumps(counts), flush=True)
            previous_counts = counts
        failures = [r["id"] for r in rows if r["status"] in ["failed", "error"]]
        if failures:
            raise RuntimeError("Temperature evaluations failed: " + str(failures))
        if len(rows) == 204 and all(r["status"] == "done" for r in rows):
            deployment.collect()
            return
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-report", "submit", "status", "watch", "collect"])
    mode = parser.parse_args().mode
    if mode == "status":
        rows = deployment.tasks()
        print(json.dumps(dict(status_counts=dict(Counter(r["status"] for r in rows)),
            remaining=[{k: r.get(k) for k in ["id", "status", "node", "signature"]} for r in rows if r["status"] != "done"])))
    else:
        {"stage": stage, "smoke": smoke, "smoke-report": smoke_report, "submit": submit,
         "watch": watch, "collect": deployment.collect}[mode]()
