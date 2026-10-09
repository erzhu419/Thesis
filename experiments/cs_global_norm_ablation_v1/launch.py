"""Small code staging, parallel CPU jobs, server-to-server actor handoffs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

STORES = ["node007", "jtl110gpu", "jtl110gpu2", "jtl311linux"]
HPC = "/home/zhengliang01/scheduleurm_work"
SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", HPC) if node.startswith("node") else str(HERE)


def ssh(node, command):
    return shlex.split(scheduler._ssh_rsync_shell_for_node(node)) + [scheduler._ssh_target_for_node(node), command]


def relay(sender, receiver, folder):
    source = subprocess.Popen(ssh(sender, f"tar -C {shlex.quote(root(sender))} -czf - {shlex.quote(folder)}"), stdout=subprocess.PIPE)
    try:
        subprocess.run(ssh(receiver, f"tar -C {shlex.quote(root(receiver))} -xzf -"),
                       stdin=source.stdout, check=True, timeout=120)
    finally:
        source.stdout.close()
    if source.wait(timeout=30):
        raise RuntimeError("Server-to-server relay failed")


def stage():
    for node in STORES:
        scheduler.run_on(node, "mkdir -p " + shlex.quote(root(node)), timeout=45, check=True)
        runtime = HERE.parent / "cs_resac_passenger_cohorts_v1/deployments" / node / "node_runtime.json"
        python = json.loads(runtime.read_text())["python"]
        deployment = HERE / "deployments" / node
        deployment.mkdir(parents=True, exist_ok=True)
        boot = deployment / "boot.sh"
        boot.write_text("#!/bin/sh\nexec " + shlex.quote(python) +
                        ' -u "$(dirname "$0")/run.py" "$@"\n')
        for name in ["run.py", "contract.py", "collect.py", "protocol.json"]:
            shutil.copy2(HERE / name, deployment / name)
        shutil.copy2(runtime, deployment / "node_runtime.json")
        subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
            str(deployment) + "/", scheduler._ssh_target_for_node(node) + ":" + root(node) + "/"], check=True, timeout=60)
        print("Staged code only: " + node, flush=True)


def task(label, fields, *, train=False, ram=1800, gate=None, node=None):
    family = P["name"] + ("/train" if train else "/eval")
    row = dict(project="Thesis", description=P["name"] + " " + label,
        signature=P["name"] + "/" + label, cwd=str(HERE),
        cmd=f"bash {HERE}/boot.sh {fields}",
        allowed_nodes=P["nodes"], cpu=1, ram_mb=ram, vram=0,
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True,
        resource_family=family, ram_resource_family=family)
    if node:
        row["allowed_nodes"] = [node]
    if train:
        row.update(allow_cpu_training=True,
            cpu_training_justification="Independent single-thread small CS learners, no GPU requirement.")
    if gate:
        row["wait_for_files"] = [str(HERE / "gates" / f"{gate}.json")]
    return row


def submit_rows(rows, batch):
    path = HERE / f"{batch}_submission.json"
    if path.exists():
        raise ValueError("Already submitted: " + str(path))
    jobs = HERE / f"{batch}_jobs.json"
    write(jobs, rows)
    result = json.loads(subprocess.check_output(["python3", SCHEDULER, "submit-jsonl",
                "--file", str(jobs), "--trusted", "--json"], text=True))
    write(path, result)
    print(json.dumps(dict(batch=batch, submitted=len(result["submitted"]),
        ids=[r["id"] for r in result["submitted"]])), flush=True)


def preflight():
    submit_rows([task("preflight/global_event", "preflight --arm global_event --seed 64001",
                      train=True, ram=2500)], "preflight")


def preflight_report():
    jobs = [r for r in campaign_tasks() if "/preflight/" in r["signature"]]
    assert len(jobs) == 1 and jobs[0]["status"] == "done", "Preflight not complete"
    node = "node007" if jobs[0]["node"].startswith("node") else jobs[0]["node"]
    python = json.loads((HERE.parent / "cs_resac_passenger_cohorts_v1/deployments" / node / "node_runtime.json").read_text())["python"]
    code = f'''import json
from pathlib import Path
root=Path({root(node)!r})
arm='global_event'
i=json.loads((root/'initialization_check.json').read_text())
assert i['passed'] and len(i['seeds'])==2
t=json.loads((root/'preflight'/arm/'seed64001/train_summary.json').read_text())
s=json.loads((root/'preflight'/arm/'seed64001/settings.json').read_text())
assert s['config']['weight_reg']==0 and s['config']['gamma']==1
assert len(t['history'])==2 and t['training_steps']==80
assert all(r['timestamp_cost_identity_passed'] and r['updates']==40 for r in t['history'])
assert all(r['credit']['cost_identity_passed'] and r['credit']['terminal_transitions']==1 for r in t['history'])
e=[json.loads((root/'preflight_results'/arm/(s+'.json')).read_text()) for s in ['normal','burst50']]
assert all(r['horizon_seconds']==18000 and r['timestamp_cost_identity_passed'] for r in e)
assert e[0]['groups']['baseline']['passengers']==e[1]['groups']['baseline']['passengers']
print(json.dumps(dict(complete=True,initialization=i,config_difference=['weight_reg'],
 training_wall_seconds=t['training_wall_seconds'],
 peak_rss_mb=max(r['peak_rss_mb'] for r in e),
 final_update=t['history'][-1]['update_last'],credit=t['history'][-1]['credit'],
 evaluation_counts={{r['scene']:{{k:r['groups'][k]['passengers'] for k in ['all','baseline','extra']}} for r in e}})))
'''
    record = json.loads(scheduler.run_on(node, python + " -c " + shlex.quote(code), timeout=45, check=True)[1])
    write(HERE / "preflight_summary.json", record)
    print(json.dumps(record))


def submit():
    assert json.loads((HERE / "preflight_summary.json").read_text())["complete"]
    rows = []
    for arm in P["arms"]:
        for seed in P["training_seeds"]:
            key = f"{arm}_seed{seed}"
            rows.append(task(f"train/{arm}/seed{seed}", f"train --arm {arm} --seed {seed}", train=True, ram=1600))
            for scene in P["scenes"]:
                for day in P["evaluation_seeds"]:
                    ram = 1800 if scene == "static10" else 1500
                    label = f"eval/{arm}/seed{seed}/{scene}/day{day}"
                    rows.append(task(label, f"eval --arm {arm} --seed {seed} --scene {scene} --day {day}", ram=ram, gate=key))
    rows.append(dict(project="Thesis", description=P["name"] + " actor handoff",
        signature=P["name"] + "/watch", cwd=str(HERE), cmd=f"{sys.executable} -u {HERE}/launch.py watch",
        require_node="local", cpu=1, ram_mb=256, vram=0, allow_no_ckpt=True,
        allow_no_resume=True, skip_launch_staging=True))
    submit_rows(rows, "formal")


def campaign_tasks():
    rows = [r for r in scheduler.load_state()["tasks"] if r.get("signature", "").startswith(P["name"] + "/")]
    # Automatic retries retain the same signature; old failed attempts are not dependencies.
    latest = {}
    for row in sorted(rows, key=lambda r: int(r["id"][1:])):
        latest[row["signature"]] = row
    return list(latest.values())


def publish(row):
    label = row["signature"].split("/train/", 1)[1]
    arm, seed = label.split("/")
    gate = HERE / "gates" / f"{arm}_{seed}.json"
    if gate.exists():
        return
    node = row["node"]
    source = "node007" if node.startswith("node") else node
    folder = "runs/" + label
    command = "test -s " + shlex.quote(root(source) + "/" + folder + "/ready.json")
    scheduler.run_on(source, command, timeout=30, check=True)
    if source != "node007":
        relay(source, "node007", folder)
    for receiver in STORES[1:]:
        if receiver != source:
            relay("node007", receiver, folder)
    write(gate, dict(ready=label, source_node=node, replicated_stores=STORES))
    print("Released 18 distributed evaluations: " + label, flush=True)


def collect():
    for node in STORES[1:]:
        if scheduler.run_on(node, "test -d " + shlex.quote(root(node) + "/results"), timeout=30, check=False)[0] == 0:
            relay(node, "node007", "results")
    python = json.loads((HERE.parent / "cs_resac_passenger_cohorts_v1/deployments/node007/node_runtime.json").read_text())["python"]
    scheduler.run_on("node007", python + " " + shlex.quote(root("node007") + "/collect.py"), timeout=60, check=True)
    subprocess.run(["rsync", "-az", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        scheduler._ssh_target_for_node("node007") + ":" + root("node007") + "/analysis_summary.json", str(HERE) + "/"],
        check=True, timeout=60)
    result = json.loads((HERE / "analysis_summary.json").read_text())
    print(json.dumps({k: result[k] for k in ["complete", "evaluations", "training_runs"]}), flush=True)
    return result


def watch():
    previous = None
    while True:
        try:
            rows = [r for r in campaign_tasks() if "/preflight/" not in r["signature"] and not r["signature"].endswith("/watch")]
            for row in rows:
                if "/train/" in row["signature"] and row["status"] == "done":
                    publish(row)
            counts = dict(Counter(r["status"] for r in rows))
            if counts != previous:
                print(json.dumps(counts), flush=True)
                previous = counts
                write(HERE / "handoff_status.json", dict(status_counts=counts))
            if rows and all(r["status"] == "done" for r in rows):
                assert collect()["complete"]
                return
            failed = [r["id"] for r in rows if r["status"] in ["failed", "killed"]]
            if failed:
                write(HERE / "handoff_status.json", dict(status_counts=counts, failed=failed))
        except (RuntimeError, subprocess.SubprocessError, OSError) as error:
            print(str(error)[-600:], flush=True)
        time.sleep(40)


def status():
    rows = campaign_tasks()
    print(json.dumps(dict(counts=dict(Counter(r["status"] for r in rows)),
        active=[{k: r.get(k) for k in ["id", "node", "status", "signature"]}
                for r in rows if "/train/" in r["signature"] or "/preflight/" in r["signature"]]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "preflight", "preflight_report", "submit", "watch", "collect", "status"])
    globals()[parser.parse_args().mode]()
