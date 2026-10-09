"""Run small CPU diagnostics beside, rather than inside, completed campaigns."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

STORES = ["node007", "jtl110gpu", "jtl110gpu2", "jtl311linux"]
HPC = "/home/zhengliang01/scheduleurm_work"
NAME = "thesis-cs-reward-diagnosis-v1"


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", HPC) if node.startswith("node") else str(HERE)


def ssh(node, command):
    return shlex.split(scheduler._ssh_rsync_shell_for_node(node)) + [scheduler._ssh_target_for_node(node), command]


def relay(sender, receiver, folder):
    source = subprocess.Popen(ssh(sender, f"tar -C {shlex.quote(root(sender))} -czf - {folder}"), stdout=subprocess.PIPE)
    try:
        subprocess.run(ssh(receiver, f"tar -C {shlex.quote(root(receiver))} -xzf -"),
                       stdin=source.stdout, check=True, timeout=90)
    finally:
        source.stdout.close()
    if source.wait(timeout=30):
        raise RuntimeError("Server-to-server relay failed")


def stage():
    for node in STORES:
        scheduler.run_on(node, "mkdir -p " + shlex.quote(root(node)), timeout=45, check=True)
        key = "node007" if node.startswith("node") else node
        runtime = HERE.parent / "cs_resac_passenger_cohorts_v1/deployments" / key / "node_runtime.json"
        for path in [HERE / "diagnose.py", runtime]:
            subprocess.run(["rsync", "-az", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
                str(path), scheduler._ssh_target_for_node(node) + ":" + root(node) + "/"], check=True, timeout=60)
        python = json.loads(runtime.read_text())["python"]
        boot = HERE / f"boot_{key}.sh"
        boot.write_text("#!/bin/sh\nexec " + shlex.quote(python) +
                        ' -u "$(dirname "$0")/diagnose.py" "$@"\n')
        subprocess.run(["rsync", "-az", "-e", scheduler._ssh_rsync_shell_for_node(node), str(boot),
            scheduler._ssh_target_for_node(node) + ":" + root(node) + "/boot.sh"], check=True, timeout=60)
    code = f'''from pathlib import Path
import shutil
root=Path({root('node007')!r})
(root/'policies').mkdir(exist_ok=True)
for objective in ['headway','passenger']:
 for seed in [64001,64002]:
  source=root.parent/'reward_cross_v1/runs/cs'/objective/'normal'/f'csbapr_no_nau_seed{{seed}}'/'policy.pt'
  shutil.copy2(source,root/'policies'/f'{{objective}}_seed{{seed}}.pt')
'''
    scheduler.run_on("node007", HPC + "/conda_envs/csbapr-gpu-py310/bin/python -c " + shlex.quote(code), timeout=45, check=True)
    for node in STORES[1:]:
        relay("node007", node, "policies")


def submit():
    if (HERE / "submission.json").exists():
        raise ValueError("Diagnosis already submitted")
    rows = []
    actors = [(arm, seed) for arm in ["headway", "passenger"] for seed in [64001, 64002]]
    actors += [(f"fixed{hold}", 0) for hold in [0, 15, 30, 60]]
    for arm, seed in actors:
        for scene in ["normal", "burst50"]:
            label = f"{arm}/{seed}/{scene}"
            rows.append(dict(project="Thesis", description=NAME + " " + label,
                signature=NAME + "/" + label, cwd=str(HERE),
                cmd=f"bash {HERE}/boot.sh --arm {arm} --seed {seed} --scene {scene}",
                allowed_nodes=[f"node{i:03d}" for i in range(1, 8)] + STORES[1:],
                cpu=1, ram_mb=1500, vram=0, allow_no_ckpt=True, allow_no_resume=True,
                skip_launch_staging=True, resource_family=NAME))
    jobs = HERE / "jobs.json"
    jobs.write_text(json.dumps(rows, indent=2) + "\n")
    output = subprocess.check_output(["python3", "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py",
        "submit-jsonl", "--file", str(jobs), "--trusted", "--json"], text=True)
    result = json.loads(output)
    (HERE / "submission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(submitted=len(result["submitted"]), ids=[r["id"] for r in result["submitted"]])))


def collect():
    for node in STORES[1:]:
        if scheduler.run_on(node, "test -d " + shlex.quote(root(node) + "/results"), timeout=30, check=False)[0] == 0:
            relay(node, "node007", "results")
    code = f'''from pathlib import Path
import json
root=Path({root('node007')!r})
rows=[json.loads(p.read_text()) for p in sorted((root/'results').glob('*.json'))]
assert len(rows)==16, (len(rows),16)
for scene in ['normal','burst50']:
 cases=[r for r in rows if r['scene']==scene]
 assert len({{r['groups']['all']['passengers'] for r in cases}})==1
 assert len({{r['groups']['baseline']['passengers'] for r in cases}})==1
(root/'analysis_summary.json').write_text(json.dumps(dict(complete=True, evaluations=16,
 analysis_location='server', evaluation_days=1, rows=rows),indent=2)+'\\n')
'''
    scheduler.run_on("node007", HPC + "/conda_envs/csbapr-gpu-py310/bin/python -c " + shlex.quote(code), timeout=45, check=True)
    subprocess.run(["rsync", "-az", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        scheduler._ssh_target_for_node("node007") + ":" + root("node007") + "/analysis_summary.json", str(HERE) + "/"],
        check=True, timeout=60)
    print("Collected compact server-side summary; no weights or passenger records downloaded")


if __name__ == "__main__":
    dict(stage=stage, submit=submit, collect=collect)[sys.argv[1]]()
