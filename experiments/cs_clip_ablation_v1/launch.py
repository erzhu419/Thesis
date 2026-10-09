"""Parallel CPU evaluation of already replicated actors, with small summaries."""

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

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("clip_source_launch", HERE.parent / "cs_source_envelope_v1/launch.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
scheduler, STORES = shared.scheduler, shared.STORES


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work") if node.startswith("node") else str(HERE)


def stage():
    for node in STORES:
        scheduler.run_on(node, "mkdir -p " + shlex.quote(root(node)), timeout=45, check=True)
        deployment = HERE / "deployments" / node
        deployment.mkdir(parents=True, exist_ok=True)
        (deployment / "boot.sh").write_text("#!/bin/sh\nexec " + shlex.quote(shared.python(node)) + ' -u "$(dirname "$0")/run.py" "$@"\n')
        for name in ["run.py", "collect.py", "linear.py", "protocol.json"]:
            shutil.copy2(HERE / name, deployment / name)
        subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
            str(deployment) + "/", scheduler._ssh_target_for_node(node) + ":" + root(node) + "/"], check=True, timeout=60)
        print("Staged code only: " + node, flush=True)
    prepare_references()


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
        cases = [("hard_clip", 64002, 78101, "static10"), ("linear_total", 64001, 78101, "static10")]
    else:
        assert json.loads((HERE / "smoke_passed.json").read_text())["passed"]
        cases = [("linear_total", seed, day, scene) for seed, day, scene in product(P["training_seeds"], P["evaluation_days"], P["scenes"])]
    jobs = []
    for variant, seed, day, scene in cases:
        label = f"{variant}/{seed}/{day}/{scene}"
        command = f"--variant {variant} --seed {seed} --day {day} --scene {scene}" + (" --smoke" if smoke else "")
        jobs.append(dict(project="Thesis", signature=P["name"] + "/" + batch + "/" + label,
            description=P["name"] + " " + batch + " " + label, cwd=str(HERE), cmd=f"bash {HERE}/boot.sh {command}",
            allowed_nodes=P["nodes"], cpu=1, ram_mb=1600 if scene == "static10" else 1300, vram=0,
            resource_family=P["name"] + "/eval", ram_resource_family=P["name"] + "/eval",
            allow_cpu_training=True, cpu_training_justification="Frozen CPU-only policy evaluation, no learner updates or GPU requirement.",
            allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True))
    path = HERE / f"{batch}_jobs.json"
    path.write_text(json.dumps(jobs, indent=2) + "\n")
    result = json.loads(subprocess.check_output([sys.executable, shared.SCHEDULER, "submit-jsonl", "--file", str(path), "--trusted", "--json"], text=True))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(submitted=len(result["submitted"]), ids=[r["id"] for r in result["submitted"]])), flush=True)


def relay(sender, receiver, folder):
    source = subprocess.Popen(shared.ssh(sender, "tar -C " + shlex.quote(root(sender)) + " -czf - " + folder), stdout=subprocess.PIPE)
    try:
        subprocess.run(shared.ssh(receiver, "tar -C " + shlex.quote(root(receiver)) + " -xzf -"), stdin=source.stdout, check=True, timeout=120)
    finally:
        source.stdout.close()
    assert source.wait(timeout=30) == 0


def prepare_references():
    code = f'''import json
from pathlib import Path
from itertools import product
root=Path({root("node007")!r})
source=root.parent/{P['policy_campaign']!r}/'results'/{P['policy_arm']!r}
refs={{}}
for seed,day,scene in product({P['training_seeds']!r},{P['evaluation_days']!r},{list(P['scenes'])!r}):
 row=json.loads((source/('seed'+str(seed))/('day'+str(day)+'_'+scene+'.json')).read_text())
 refs[str(seed)+'/'+str(day)+'/'+scene]={{k:row[k] for k in ['cost','exact_cost','decisions','mean_holding_seconds','groups']}}
(root/'inputs').mkdir(exist_ok=True)
raw=json.dumps(refs,indent=2)+'\\n'
(root/'inputs/reference.json').write_text(raw)
print(json.dumps(dict(reference_cases=len(refs),bytes=len(raw.encode()))))
'''
    print(scheduler.run_on("node007", shared.python("node007") + " -c " + shlex.quote(code), timeout=45, check=True)[1], flush=True)
    for receiver in STORES[1:]:
        relay("node007", receiver, "inputs")


def gather(folder):
    for node in STORES[1:]:
        if scheduler.run_on(node, "test -d " + shlex.quote(root(node) + "/" + folder), timeout=30, check=False)[0] == 0:
            relay(node, "node007", folder)


def smoke_report():
    rows = [r for r in tasks() if "/smoke/" in r["signature"]]
    assert len(rows) == 2 and all(r["status"] == "done" for r in rows)
    gather("smoke")
    code = f'''import json
from pathlib import Path
root=Path({root("node007")!r})/'smoke'
hard=json.loads((root/'hard_clip/seed64002/day78101_static10.json').read_text())
linear=json.loads((root/'linear_total/seed64001/day78101_static10.json').read_text())
assert hard['reference_rollout_matched']
assert linear['input_stats']['source_cap_exceeded_fraction']>0 and linear['input_stats']['transformed_total_maximum']>1
assert hard['updates']==linear['updates']==0
print(json.dumps(dict(passed=True,hard_reference_matched=True,linear_cost=linear['cost'],linear_holding=linear['mean_holding_seconds'],linear_peak=linear['input_stats']['transformed_total_maximum'])))
'''
    result = json.loads(scheduler.run_on("node007", shared.python("node007") + " -c " + shlex.quote(code), timeout=45, check=True)[1])
    (HERE / "smoke_passed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def collect():
    gather("results")
    print(scheduler.run_on("node007", shared.python("node007") + " " + shlex.quote(root("node007") + "/collect.py"), timeout=60, check=True)[1])
    code = "from pathlib import Path;print(Path(" + repr(root("node007") + "/analysis_summary.json") + ").read_text())"
    raw = scheduler.run_on("node007", shared.python("node007") + " -c " + shlex.quote(code), timeout=60, check=True)[1]
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
            remaining=[{k:r.get(k) for k in ['id','status','node','signature']} for r in rows if r['status']!='done'])))
    else:
        {"stage": stage, "smoke-report": smoke_report, "collect": collect}[mode]()
