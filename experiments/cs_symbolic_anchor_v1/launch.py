"""Stage a frozen symbolic runtime and reuse all-node actor handoff."""

import argparse
import importlib.util
from itertools import product
import json
from pathlib import Path
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
path = HERE.parent / "cs_source_anchor_v1/launch.py"
spec = importlib.util.spec_from_file_location("symbolic_anchor_deployment", path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.HERE, previous.P = HERE, P
shared, scheduler = previous.shared, previous.scheduler


def stage():
    for node in previous.STORES:
        destination = previous.root(node)
        scheduler.run_on(node, "mkdir -p " + shlex.quote(destination), timeout=45, check=True)
        deployment = HERE / "deployments" / node
        deployment.mkdir(parents=True, exist_ok=True)
        for name in ["run.py", "policy.py", "collect.py", "protocol.json"]:
            shutil.copy2(HERE / name, deployment / name)
        shutil.copytree(HERE.parent / "csbapr_symbolic_bus_v2/runtime/csbapr", deployment / "runtime/csbapr",
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (deployment / "boot.sh").write_text("#!/bin/sh\nexec " + shlex.quote(shared.python(node)) + ' -u "$(dirname "$0")/run.py" "$@"\n')
        subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
                        str(deployment) + "/", scheduler._ssh_target_for_node(node) + ":" + destination + "/"], check=True, timeout=60)
        print("Staged frozen code only: " + node, flush=True)
    code = f'''import json,shutil
from pathlib import Path
r=Path({previous.root('node007')!r})
prior=r.parent/'cs_source_anchor_v1'
shutil.copytree(prior/'inputs',r/'inputs',dirs_exist_ok=True)
source=json.loads((prior/'analysis_summary.json').read_text())
controls=json.loads((r/'inputs/controls.json').read_text())
controls['mlp_anchor_groups']=source['groups']
(r/'inputs/controls.json').write_text(json.dumps(controls))
records=[]
for seed in {P['training_seeds']!r}:
 src=r.parent/'csbapr_symbolic_bus_v2/models'/f'seed{{seed}}'
 dst=r/'inputs'/f'seed{{seed}}';dst.mkdir(exist_ok=True)
 for name in ['reference.npz','ready.json']: shutil.copy2(src/name,dst/name)
 d=json.loads((dst/'ready.json').read_text())
 assert all(line['cost_mse'][1]<line['constant_cost_mse'][1] for line in d['diagnostics'])
 records.append(dict(seed=seed,passenger_cost_fit_beats_constant_on_all_lines=True,
                     unused_dynamics_failure_lines=[line['line'] for line in d['diagnostics'] if line['increment_mse']>line['persistence_increment_mse']]))
(r/'reference_quality.json').write_text(json.dumps(dict(records=records,usage='one-step passenger station-cost feedback only')))
print(json.dumps(records))
'''
    print(scheduler.run_on("node007", shared.python("node007") + " -c " + shlex.quote(code), timeout=45, check=True)[1])
    for node in previous.STORES[1:]:
        previous.relay("node007", node, "inputs")


def preflight():
    rows = [previous.task("preflight/identity", "preflight --arm identity --seed 64001", train=True)]
    for arm, seed in product(P["arms"], P["training_seeds"]):
        rows.append(previous.task(f"preflight/{arm}/seed{seed}", f"preflight --arm {arm} --seed {seed}", train=True))
    previous.submit_rows(rows, "preflight")


def preflight_report():
    jobs = [r for r in previous.tasks() if "/preflight/" in r["signature"]]
    assert len(jobs) == 5 and all(r["status"] == "done" for r in jobs)
    records = []
    for job in jobs:
        node = "node007" if job["node"].startswith("node") else job["node"]
        parts = job["signature"].split("/preflight/", 1)[1].split("/")
        arm, seed = parts[0], int(parts[1][4:]) if len(parts) == 2 else 64001
        code = f'''import json
from pathlib import Path
r=Path({previous.root(node)!r});f=r/'preflight'/{arm!r}/'seed{seed}'
t=json.loads((f/'train_summary.json').read_text())
assert len(t['history'])==2 and t['training_steps']==80
assert all(row['updates']==40 and row['timestamp_cost_identity_passed'] for row in t['history'])
if {arm!r}=='identity':
 checks=json.loads((f/'native_equivalence.json').read_text())
 assert checks['native_config_fields_unchanged'] and checks['final_actor_identical'] and checks['execution_identical']
 assert json.loads((r/'contracts.json').read_text())['passed']
else:
 assert all(row['update_mean']['jac_loss']>0 and row['update_mean']['sym_penalty']>0 for row in t['history'])
 a,b=t['history']
 if {arm!r}=='symbolic10':
  assert a['source_regime']=='normal' and b['source_regime']=='pulse3'
  assert a['anchor']['source_normal_observations']==b['anchor']['source_normal_observations']>0
  assert b['anchor']['anchor_updates']==80 and t['anchor']['mean_anchor_gradient_norm']>0
 checks={{'symbolic_losses_active':True,'paired_actor_hooks_passed':True}}
print(json.dumps(dict(arm={arm!r},seed={seed},checks=checks,anchor=t['anchor'],training_wall_seconds=t['training_wall_seconds'])))
'''
        records.append(json.loads(scheduler.run_on(node, shared.python(node) + " -c " + shlex.quote(code), timeout=45, check=True)[1]))
    previous.write(HERE / "preflight_summary.json", dict(passed=True, records=records))
    print(json.dumps(dict(passed=True, records=records)))


def submit():
    assert json.loads((HERE / "evaluation_preflight.json").read_text())["passed"]
    previous.submit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "preflight", "preflight-report", "submit", "status", "watch", "collect"])
    mode = parser.parse_args().mode
    if mode == "status":
        from collections import Counter
        rows = previous.tasks()
        print(json.dumps(dict(status_counts=dict(Counter(r["status"] for r in rows)),
            training=[{k:r.get(k) for k in ['id','status','node','signature']} for r in rows if '/train/' in r['signature'] or '/preflight/' in r['signature']])))
    else:
        {"stage":stage,"preflight":preflight,"preflight-report":preflight_report,
         "submit":submit,"watch":previous.watch,"collect":previous.collect}[mode]()
