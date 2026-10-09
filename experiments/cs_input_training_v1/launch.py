"""Parallel training with replicated actors and all-node evaluation gates."""

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
spec = importlib.util.spec_from_file_location("input_source_deployment", HERE.parent / P["threshold_campaign"] / "launch.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
scheduler, STORES = shared.scheduler, shared.STORES


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work") if node.startswith("node") else str(HERE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def stage():
    for node in STORES:
        scheduler.run_on(node, "mkdir -p " + shlex.quote(root(node)), timeout=45, check=True)
        deployment = HERE / "deployments" / node
        deployment.mkdir(parents=True, exist_ok=True)
        (deployment / "boot.sh").write_text("#!/bin/sh\nexec " + shlex.quote(shared.python(node)) + ' -u "$(dirname "$0")/run.py" "$@"\n')
        for name in ["run.py", "collect.py", "representation.py", "protocol.json"]:
            shutil.copy2(HERE / name, deployment / name)
        subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node), str(deployment) + "/",
            scheduler._ssh_target_for_node(node) + ":" + root(node) + "/"], check=True, timeout=60)
        print("Staged new code only: " + node, flush=True)


def tasks():
    latest = {}
    for row in sorted(scheduler.load_state()["tasks"], key=lambda r: int(r["id"][1:])):
        if row.get("signature", "").startswith(P["name"] + "/"):
            latest[row["signature"]] = row
    return list(latest.values())


def task(label, command, train=False, gate=None):
    family = P["name"] + ("/train" if train else "/eval")
    row = dict(project="Thesis", signature=P["name"] + "/" + label, description=P["name"] + " " + label,
        cwd=str(HERE), cmd=f"bash {HERE}/boot.sh {command}", allowed_nodes=P["nodes"], cpu=1,
        ram_mb=2200 if train else (1600 if "static10" in command else 1300), vram=0,
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True,
        allow_cpu_training=True,
        cpu_training_justification="Intentional CPU-only evaluation; campaign path also matches the scheduler training heuristic.",
        resource_family=family, ram_resource_family=family)
    if train:
        row.update(allow_cpu_training=True, cpu_training_justification="Matched original single-thread small learner, no GPU requirement.")
    if gate:
        row["wait_for_files"] = [str(HERE / "gates" / f"{gate}.json")]
    return row


def submit_rows(rows, batch):
    path = HERE / f"{batch}_submission.json"
    assert not path.exists(), "Batch already submitted"
    jobs = HERE / f"{batch}_jobs.json"
    write(jobs, rows)
    result = json.loads(subprocess.check_output([sys.executable, shared.SCHEDULER, "submit-jsonl", "--file", str(jobs), "--trusted", "--json"], text=True))
    write(path, result)
    print(json.dumps(dict(batch=batch, submitted=len(result["submitted"]), ids=[r["id"] for r in result["submitted"]])), flush=True)


def preflight():
    submit_rows([task("preflight/" + arm, f"preflight --arm {arm} --seed 64001", train=True) for arm in ["identity"] + P["arms"]], "preflight")


def preflight_report():
    jobs = [r for r in tasks() if "/preflight/" in r["signature"]]
    assert len(jobs) == 3 and all(r["status"] == "done" for r in jobs)
    records = []
    for job in jobs:
        node = "node007" if job["node"].startswith("node") else job["node"]
        arm = job["signature"].rsplit("/", 1)[1]
        code = f'''import json
from pathlib import Path
root=Path({root(node)!r})
arm={arm!r}
folder=root/'preflight'/arm/'seed64001'
settings=json.loads((folder/'settings.json').read_text())
train=json.loads((folder/'train_summary.json').read_text())
assert len(train['history'])==2 and train['training_steps']==80
assert settings['config']['weight_reg']==0 and settings['config']['gamma']==1
assert [r['source_regime'] for r in train['history']]==['pulse3','repeated3']
assert all(r['timestamp_cost_identity_passed'] and r['updates']==40 and r['credit']['cost_identity_passed'] for r in train['history'])
if arm=='identity':
 assert json.loads((folder/'native_equivalence.json').read_text())['passed']
else:
 assert all(r['input_stats']['changed_fraction']>0 for r in train['history'])
rows=[json.loads((root/'preflight_results'/arm/'seed64001'/('day78101_'+scene+'.json')).read_text()) for scene in ['normal','static10','burst50']]
assert all(r['updates']==0 and r['horizon_seconds']==18000 and r['timestamp_cost_identity_passed'] for r in rows)
if arm=='source_clip':
 assert all(r['input_stats']['transformed_total_maximum']<=1.000001 for r in rows)
print(json.dumps(dict(arm=arm,passed=True,training_wall_seconds=train['training_wall_seconds'],
 input_stats=[r['input_stats'] for r in train['history']],
 evaluations=[{{k:r[k] for k in ['scene','cost','mean_holding_seconds','input_stats']}} for r in rows])))
'''
        records.append(json.loads(scheduler.run_on(node, shared.python(node) + " -c " + shlex.quote(code), timeout=45, check=True)[1]))
    write(HERE / "preflight_summary.json", dict(passed=True, records=records))
    print(json.dumps(dict(passed=True, records=records)))


def submit():
    assert json.loads((HERE / "preflight_summary.json").read_text())["passed"]
    rows = []
    for arm, seed in product(P["arms"], P["training_seeds"]):
        rows.append(task(f"train/{arm}/seed{seed}", f"train --arm {arm} --seed {seed}", train=True))
        for scene, day in product(P["scenes"], P["evaluation_days"]):
            rows.append(task(f"eval/{arm}/seed{seed}/{scene}/day{day}",
                f"eval --arm {arm} --seed {seed} --scene {scene} --day {day}", gate=f"{arm}_seed{seed}"))
    rows.append(dict(project="Thesis", signature=P["name"] + "/watch", description=P["name"] + " actor handoff and summary",
        cwd=str(HERE), cmd=f"{sys.executable} -u {HERE}/launch.py watch", require_node="local", cpu=1, ram_mb=256,
        vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True,
        allow_cpu_training=True, cpu_training_justification="CPU-only actor distribution and result collection, not model training."))
    submit_rows(rows, "formal")


def relay(sender, receiver, folder):
    source = subprocess.Popen(shared.ssh(sender, "tar -C " + shlex.quote(root(sender)) + " -czf - " + shlex.quote(folder)), stdout=subprocess.PIPE)
    try:
        subprocess.run(shared.ssh(receiver, "tar -C " + shlex.quote(root(receiver)) + " -xzf -"), stdin=source.stdout, check=True, timeout=120)
    finally:
        source.stdout.close()
    assert source.wait(timeout=30) == 0


def publish(row):
    arm, seed = row["signature"].split("/train/", 1)[1].split("/")
    gate = HERE / "gates" / f"{arm}_{seed}.json"
    if gate.exists():
        return
    source = "node007" if row["node"].startswith("node") else row["node"]
    folder = f"runs/{arm}/{seed}"
    scheduler.run_on(source, "test -s " + shlex.quote(root(source) + "/" + folder + "/ready.json"), timeout=30, check=True)
    if source != "node007":
        relay(source, "node007", folder)
    for receiver in STORES[1:]:
        if receiver != source:
            relay("node007", receiver, folder)
    write(gate, dict(source_node=row["node"], replicated_stores=STORES, ready=True))
    print("Released 9 independent evaluations: " + folder, flush=True)


def collect():
    for node in STORES[1:]:
        if scheduler.run_on(node, "test -d " + shlex.quote(root(node) + "/results"), timeout=30, check=False)[0] == 0:
            relay(node, "node007", "results")
    print(scheduler.run_on("node007", shared.python("node007") + " " + shlex.quote(root("node007") + "/collect.py"), timeout=60, check=True)[1])
    code = "from pathlib import Path;print(Path(" + repr(root("node007") + "/analysis_summary.json") + ").read_text())"
    raw = scheduler.run_on("node007", shared.python("node007") + " -c " + shlex.quote(code), timeout=60, check=True)[1]
    assert json.loads(raw)["complete"]
    (HERE / "analysis_summary.json").write_text(raw)


def watch():
    previous = None
    while True:
        try:
            rows = [r for r in tasks() if "/preflight/" not in r["signature"] and not r["signature"].endswith("/watch")]
            for row in rows:
                if "/train/" in row["signature"] and row["status"] == "done":
                    publish(row)
            counts = dict(Counter(r["status"] for r in rows))
            if counts != previous:
                print(json.dumps(counts), flush=True)
                previous = counts
                write(HERE / "handoff_status.json", dict(status_counts=counts))
            if rows and all(r["status"] == "done" for r in rows):
                collect()
                return
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-500:], flush=True)
        time.sleep(40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "preflight", "preflight-report", "submit", "status", "watch", "collect"])
    mode = parser.parse_args().mode
    if mode == "status":
        rows = tasks()
        print(json.dumps(dict(status_counts=dict(Counter(r["status"] for r in rows)),
            training=[{k:r.get(k) for k in ['id','status','node','signature']} for r in rows if '/train/' in r['signature'] or '/preflight/' in r['signature']])))
    else:
        {"stage": stage, "preflight": preflight, "preflight-report": preflight_report, "submit": submit, "watch": watch, "collect": collect}[mode]()
