"""Stage only this new code and reuse previously frozen server inputs."""

import argparse
from collections import Counter
from itertools import product
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("source_envelope_launch", HERE.parent / P["frozen_campaign"] / "launch.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
scheduler = old.scheduler


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work") if node.startswith("node") else str(HERE)


def stage():
    for node in old.STORES:
        scheduler.run_on(node, "mkdir -p " + shlex.quote(root(node)), timeout=45, check=True)
        deployment = HERE / "deployments" / node
        deployment.mkdir(parents=True, exist_ok=True)
        (deployment / "boot.sh").write_text("#!/bin/sh\nexec " + shlex.quote(old.python(node)) + ' -u "$(dirname "$0")/run.py" "$@"\n')
        for name in ["run.py", "collect.py", "phase_total.py", "protocol.json"]:
            shutil.copy2(HERE / name, deployment / name)
        subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
            str(deployment) + "/", scheduler._ssh_target_for_node(node) + ":" + root(node) + "/"], check=True, timeout=60)
        print("Staged new code only: " + node, flush=True)


def tasks():
    latest = {}
    for row in sorted(scheduler.load_state()["tasks"], key=lambda r: int(r["id"][1:])):
        if row.get("signature", "").startswith(P["name"] + "/"):
            latest[row["signature"]] = row
    return list(latest.values())


def submit(smoke):
    batch = "smoke" if smoke else "formal"
    out = HERE / f"{batch}_submission.json"
    assert not out.exists(), "Batch already submitted"
    if smoke:
        cases = [(64002, 78101, "normal"), (64001, 78101, "static10")]
    else:
        assert json.loads((HERE / "smoke_passed.json").read_text())["passed"]
        cases = product(P["seeds"], P["evaluation_days"], P["scenes"])
    jobs = []
    for seed, day, scene in cases:
        label = f"{seed}/{day}/{scene}"
        command = f"--seed {seed} --day {day} --scene {scene}" + (" --smoke" if smoke else "")
        jobs.append(dict(project="Thesis", signature=P["name"] + "/" + batch + "/" + label,
            description=P["name"] + " " + batch + " " + label, cwd=str(HERE),
            cmd=f"bash {HERE}/boot.sh {command}", allowed_nodes=P["nodes"], cpu=1,
            ram_mb=1600 if scene == "static10" else 1300, vram=0,
            resource_family=P["name"] + "/eval", ram_resource_family=P["name"] + "/eval",
            allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True))
    path = HERE / f"{batch}_jobs.json"
    path.write_text(json.dumps(jobs, indent=2) + "\n")
    result = json.loads(subprocess.check_output([sys.executable, old.SCHEDULER, "submit-jsonl", "--file", str(path), "--trusted", "--json"], text=True))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(submitted=len(result["submitted"]), ids=[r["id"] for r in result["submitted"]])))


def gather(folder):
    for node in old.STORES[1:]:
        if scheduler.run_on(node, "test -d " + shlex.quote(root(node) + "/" + folder), timeout=30, check=False)[0] != 0:
            continue
        source = subprocess.Popen(old.ssh(node, "tar -C " + shlex.quote(root(node)) + " -czf - " + folder), stdout=subprocess.PIPE)
        try:
            subprocess.run(old.ssh("node007", "tar -C " + shlex.quote(root("node007")) + " -xzf -"), stdin=source.stdout, check=True, timeout=120)
        finally:
            source.stdout.close()
        assert source.wait(timeout=30) == 0


def smoke_report():
    rows = [r for r in tasks() if "/smoke/" in r["signature"]]
    assert len(rows) == 2 and all(r["status"] == "done" for r in rows)
    gather("smoke")
    code = f'''import json
from pathlib import Path
root=Path({root("node007")!r})/'smoke'
normal=json.loads((root/'seed64002/day78101_normal.json').read_text())
stress=json.loads((root/'seed64001/day78101_static10.json').read_text())
assert normal['unchanged_rollout_matched']
assert stress['adaptation']['changed_decisions']>0 and stress['rollout']['timestamp_cost_identity_passed']
print(json.dumps(dict(passed=True,normal_unchanged=True,stress_cost=stress['rollout']['cost'],
stress_holding=stress['rollout']['mean_holding_seconds'],stress_changed_fraction=stress['adaptation']['changed_fraction'])))
'''
    result = json.loads(scheduler.run_on("node007", old.python("node007") + " -c " + shlex.quote(code), timeout=45, check=True)[1])
    (HERE / "smoke_passed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def collect():
    gather("results")
    print(scheduler.run_on("node007", old.python("node007") + " " + shlex.quote(root("node007") + "/collect.py"), timeout=60, check=True)[1])
    code = "from pathlib import Path;print(Path(" + repr(root("node007") + "/analysis_summary.json") + ").read_text())"
    raw = scheduler.run_on("node007", old.python("node007") + " -c " + shlex.quote(code), timeout=60, check=True)[1]
    assert json.loads(raw)["complete"]
    (HERE / "analysis_summary.json").write_text(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-report", "submit", "status", "collect"])
    mode = parser.parse_args().mode
    if mode in ["smoke", "submit"]:
        submit(mode == "smoke")
    elif mode == "status":
        rows = tasks()
        print(json.dumps(dict(status_counts=dict(Counter(r["status"] for r in rows)),
            remaining=[{k: r.get(k) for k in ["id", "status", "node", "signature"]} for r in rows if r["status"] != "done"])))
    else:
        {"stage": stage, "smoke-report": smoke_report, "collect": collect}[mode]()
