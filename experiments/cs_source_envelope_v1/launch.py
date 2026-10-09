"""Small staging and source-only calibration followed by parallel CPU tests."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

STORES = ["node007", "jtl110gpu", "jtl110gpu2", "jtl311linux"]
SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work") if node.startswith("node") else str(HERE)


def python(node):
    path = HERE.parent / "cs_resac_passenger_cohorts_v1/deployments" / node / "node_runtime.json"
    return json.loads(path.read_text())["python"]


def ssh(node, command):
    return shlex.split(scheduler._ssh_rsync_shell_for_node(node)) + [scheduler._ssh_target_for_node(node), command]


def stage():
    for node in STORES:
        scheduler.run_on(node, "mkdir -p " + shlex.quote(root(node)), timeout=45, check=True)
        deployment = HERE / "deployments" / node
        deployment.mkdir(parents=True, exist_ok=True)
        (deployment / "boot.sh").write_text("#!/bin/sh\nexec " + shlex.quote(python(node)) + ' -u "$(dirname "$0")/run.py" "$@"\n')
        for name in ["run.py", "collect.py", "envelope.py", "protocol.json"]:
            shutil.copy2(HERE / name, deployment / name)
        subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
            str(deployment) + "/", scheduler._ssh_target_for_node(node) + ":" + root(node) + "/"], check=True, timeout=60)
        print("Staged code only: " + node, flush=True)


def tasks():
    latest = {}
    for row in sorted(scheduler.load_state()["tasks"], key=lambda r: int(r["id"][1:])):
        if row.get("signature", "").startswith(P["name"] + "/"):
            latest[row["signature"]] = row
    return list(latest.values())


def submit(batch):
    out = HERE / f"{batch}_submission.json"
    if out.exists():
        raise ValueError("Already submitted: " + str(out))
    jobs = []
    if batch == "probe":
        fields = [(f"{scene}/{day}", f"probe --scene {scene} --day {day}")
                  for scene, day in product(P["source_scenes"], P["source_probe_days"])]
    elif batch == "smoke":
        assert (HERE / "fit_summary.json").exists()
        fields = [(f"{variant}/{scene}", f"eval --variant {variant} --scene {scene} --day 78101 --seed 64002 --smoke")
                  for variant, scene in [("identity", "static10"), ("daily_total", "normal"), ("phase_envelope", "static10")]]
    else:
        assert json.loads((HERE / "smoke_passed.json").read_text())["passed"]
        fields = [(f"{variant}/{seed}/{scene}/{day}", f"eval --variant {variant} --seed {seed} --scene {scene} --day {day}")
                  for variant, seed, scene, day in product(P["variants"], P["actors"], P["scenes"], P["evaluation_days"])]
    for label, command in fields:
        jobs.append(dict(project="Thesis", signature=P["name"] + "/" + batch + "/" + label,
            description=P["name"] + " " + batch + " " + label, cwd=str(HERE),
            cmd=f"bash {HERE}/boot.sh {command}", allowed_nodes=P["nodes"], cpu=1,
            ram_mb=1600 if "static10" in command else 1300, vram=0,
            resource_family=P["name"] + "/eval", ram_resource_family=P["name"] + "/eval",
            allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True))
    path = HERE / f"{batch}_jobs.json"
    path.write_text(json.dumps(jobs, indent=2) + "\n")
    result = json.loads(subprocess.check_output([sys.executable, SCHEDULER, "submit-jsonl", "--file", str(path), "--trusted", "--json"], text=True))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(submitted=len(result["submitted"]), ids=[r["id"] for r in result["submitted"]])))


def relay(sender, receiver, folder):
    source = subprocess.Popen(ssh(sender, "tar -C " + shlex.quote(root(sender)) + " -czf - " + folder), stdout=subprocess.PIPE)
    try:
        subprocess.run(ssh(receiver, "tar -C " + shlex.quote(root(receiver)) + " -xzf -"), stdin=source.stdout, check=True, timeout=120)
    finally:
        source.stdout.close()
    assert source.wait(timeout=30) == 0


def gather(folder):
    for node in STORES[1:]:
        if scheduler.run_on(node, "test -d " + shlex.quote(root(node) + "/" + folder), timeout=30, check=False)[0] == 0:
            relay(node, "node007", folder)


def small_result(name):
    code = "from pathlib import Path;print(Path(" + repr(root("node007") + "/" + name) + ").read_text())"
    raw = scheduler.run_on("node007", python("node007") + " -c " + shlex.quote(code), timeout=60, check=True)[1]
    result = json.loads(raw)
    (HERE / name).write_text(raw)
    return result


def fit():
    gather("probes")
    print(scheduler.run_on("node007", python("node007") + " " + shlex.quote(root("node007") + "/collect.py") + " fit", timeout=60, check=True)[1])
    for receiver in STORES[1:]:
        relay("node007", receiver, "inputs")
    small_result("fit_summary.json")


def smoke_report():
    rows = [r for r in tasks() if "/smoke/" in r["signature"]]
    assert len(rows) == 3 and all(r["status"] == "done" for r in rows)
    gather("smoke")
    code = f'''import json
from pathlib import Path
root=Path({root("node007")!r})/'smoke'
a=json.loads((root/'identity/seed64002/day78101_static10.json').read_text())
b=json.loads((root/'daily_total/seed64002/day78101_normal.json').read_text())
c=json.loads((root/'phase_envelope/seed64002/day78101_static10.json').read_text())
assert a['unchanged_rollout_matched'] and b['unchanged_rollout_matched']
assert c['adaptation']['changed_decisions']>0 and c['rollout']['timestamp_cost_identity_passed']
print(json.dumps(dict(passed=True,identity_cost=a['rollout']['cost'],normal_unchanged=True,
phase_cost=c['rollout']['cost'],phase_holding=c['rollout']['mean_holding_seconds'],
phase_changed_fraction=c['adaptation']['changed_fraction'])))
'''
    result = json.loads(scheduler.run_on("node007", python("node007") + " -c " + shlex.quote(code), timeout=45, check=True)[1])
    (HERE / "smoke_passed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def collect():
    gather("results")
    print(scheduler.run_on("node007", python("node007") + " " + shlex.quote(root("node007") + "/collect.py") + " collect", timeout=60, check=True)[1])
    result = small_result("analysis_summary.json")
    assert result["complete"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "probe", "fit", "smoke", "smoke-report", "submit", "status", "collect"])
    mode = parser.parse_args().mode
    if mode in ["probe", "smoke", "submit"]:
        submit("formal" if mode == "submit" else mode)
    elif mode == "status":
        rows = tasks()
        print(json.dumps(dict(status_counts=dict(Counter(r["status"] for r in rows)),
            remaining=[{k: r.get(k) for k in ["id", "status", "node", "signature"]} for r in rows if r["status"] != "done"])))
    else:
        {"stage": stage, "fit": fit, "smoke-report": smoke_report, "collect": collect}[mode]()
