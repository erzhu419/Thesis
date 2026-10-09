"""Reuse existing deployment stores; run all diagnostics without node binding."""

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

STORES = ["node007", "jtl110gpu", "jtl311linux"]
SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work") if node.startswith("node") else str(HERE)


def runtime(node):
    return HERE.parent / "cs_resac_passenger_cohorts_v1/deployments" / node / "node_runtime.json"


def stage():
    python = json.loads(runtime("node007").read_text())["python"]
    code = f'''import json
from pathlib import Path
root=Path({root("node007")!r}).parent/'cs_terminal_coverage_v1/results/global_coverage'
refs={{}}
for path in root.glob('seed*/day*.json'):
 r=json.loads(path.read_text())
 if r['scene'] in ['normal','static10','burst50']:
  refs[str(r['seed'])+'/'+str(r['evaluation_seed'])+'/'+r['scene']]={{k:r[k] for k in ['cost','mean_holding_seconds','decisions','exact_cost']}}
assert len(refs)==18
print(json.dumps(refs))
'''
    reference = json.loads(scheduler.run_on("node007", python + " -c " + shlex.quote(code), timeout=45, check=True)[1])
    (HERE / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    for node in STORES:
        scheduler.run_on(node, "mkdir -p " + shlex.quote(root(node)), timeout=45, check=True)
        deployment = HERE / "deployments" / node
        deployment.mkdir(parents=True, exist_ok=True)
        python = json.loads(runtime(node).read_text())["python"]
        (deployment / "boot.sh").write_text("#!/bin/sh\nexec " + shlex.quote(python) + ' -u "$(dirname "$0")/run.py" "$@"\n')
        for name in ["run.py", "collect.py", "protocol.json", "reference.json"]:
            shutil.copy2(HERE / name, deployment / name)
        subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
            str(deployment) + "/", scheduler._ssh_target_for_node(node) + ":" + root(node) + "/"], check=True, timeout=60)
        print("Staged diagnostic code only: " + node, flush=True)


def tasks():
    latest = {}
    for row in sorted(scheduler.load_state()["tasks"], key=lambda r: int(r["id"][1:])):
        if row.get("signature", "").startswith(P["name"] + "/"):
            latest[row["signature"]] = row
    return list(latest.values())


def submit(smoke=False):
    batch = "smoke" if smoke else "formal"
    out = HERE / f"{batch}_submission.json"
    if out.exists():
        raise ValueError("Already submitted: " + str(out))
    if not smoke:
        assert json.loads((HERE / "smoke_passed.json").read_text())["passed"]
    cases = [("coverage_64002", "static10", 78101)] if smoke else product(P["drivers"], P["scenes"], P["days"])
    jobs = []
    for driver, scene, day in cases:
        label = f"{batch}/{driver}/{scene}/{day}"
        jobs.append(dict(project="Thesis", signature=P["name"] + "/" + label,
            description=P["name"] + " " + label, cwd=str(HERE),
            cmd=f"bash {HERE}/boot.sh --driver {driver} --scene {scene} --day {day}" + (" --smoke" if smoke else ""),
            allowed_nodes=P["nodes"], cpu=1, ram_mb=1800 if scene == "static10" else 1500, vram=0,
            resource_family=P["name"] + "/eval", ram_resource_family=P["name"] + "/eval",
            allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True))
    path = HERE / f"{batch}_jobs.json"
    path.write_text(json.dumps(jobs, indent=2) + "\n")
    result = json.loads(subprocess.check_output([sys.executable, SCHEDULER, "submit-jsonl", "--file", str(path), "--trusted", "--json"], text=True))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(submitted=len(result["submitted"]), ids=[r["id"] for r in result["submitted"]])))


def smoke_report():
    rows = [r for r in tasks() if "/smoke/" in r["signature"]]
    assert len(rows) == 1 and rows[0]["status"] == "done"
    node = "node007" if rows[0]["node"].startswith("node") else rows[0]["node"]
    python = json.loads(runtime(node).read_text())["python"]
    code = f'''import json
from pathlib import Path
r=json.loads((Path({root(node)!r})/'smoke/coverage_64002/static10_day78101.json').read_text())
assert r['native_actor_equivalence_checked'] and r['previous_rollout_matched'] and r['learner_updates']==0
assert r['rollout']['timestamp_cost_identity_passed']
print(json.dumps(dict(passed=True,cost=r['rollout']['cost'],holding=r['rollout']['mean_holding_seconds'],
wall_seconds=r['wall_seconds'],peak_rss_mb=r['peak_rss_mb'])))
'''
    result = json.loads(scheduler.run_on(node, python + " -c " + shlex.quote(code), timeout=45, check=True)[1])
    (HERE / "smoke_passed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def ssh(node, command):
    return shlex.split(scheduler._ssh_rsync_shell_for_node(node)) + [scheduler._ssh_target_for_node(node), command]


def collect():
    for node in STORES[1:]:
        if scheduler.run_on(node, "test -d " + shlex.quote(root(node) + "/results"), timeout=30, check=False)[0] != 0:
            continue
        sender = subprocess.Popen(ssh(node, "tar -C " + shlex.quote(root(node)) + " -czf - results"), stdout=subprocess.PIPE)
        try:
            subprocess.run(ssh("node007", "tar -C " + shlex.quote(root("node007")) + " -xzf -"), stdin=sender.stdout, check=True, timeout=120)
        finally:
            sender.stdout.close()
        assert sender.wait(timeout=30) == 0
    python = json.loads(runtime("node007").read_text())["python"]
    print(scheduler.run_on("node007", python + " " + shlex.quote(root("node007") + "/collect.py"), timeout=60, check=True)[1])
    code = "from pathlib import Path;print(Path(" + repr(root("node007") + "/analysis_summary.json") + ").read_text())"
    summary = scheduler.run_on("node007", python + " -c " + shlex.quote(code), timeout=60, check=True)[1]
    assert json.loads(summary)["complete"]
    (HERE / "analysis_summary.json").write_text(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-report", "submit", "status", "collect"])
    mode = parser.parse_args().mode
    if mode in ["smoke", "submit"]:
        submit(mode == "smoke")
    elif mode == "status":
        rows = tasks()
        print(json.dumps(dict(status_counts=dict(Counter(r["status"] for r in rows)),
            jobs=[{k: r.get(k) for k in ["id", "status", "node", "signature"]} for r in rows])))
    else:
        {"stage": stage, "smoke-report": smoke_report, "collect": collect}[mode]()
