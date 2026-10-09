"""Release fixed evaluation cases after replay, then collect the completed grid."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from setup_evaluation import HERE, REMOTE, ALL_NODES, scheduler
from stage_assets import stream

SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"
PREFIX = "thesis-final-controller-alignment-20261006"
PYTHON = scheduler.NODES["node001"]["resume_scan_python"]
LAB_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux"]


def read_server(code):
    _, output, _ = scheduler.run_on("node001", shlex.join([PYTHON, "-c", code]), timeout=60, check=True)
    return json.loads(output)


def wait(pattern):
    subprocess.run([sys.executable, SCHEDULER, "wait-for", "--signature", pattern,
                    "--poll", "30", "--timeout", "43200", "--verbose", "--refresh"], check=True)


def specification(method, seed, case):
    name = "%s/seed%d/case%02d" % (method, seed, case)
    args = ["bash", str(HERE / "eval_python.sh"), str(HERE / "node_case.py"),
            "--method", method, "--seed", str(seed), "--case", str(case)]
    return dict(project="Thesis", description="Common final-controller test " + name,
        signature=PREFIX + "/eval/" + name, cwd=str(HERE), cmd=shlex.join(args),
        allowed_nodes=ALL_NODES, cpu=1, ram_mb=2500, vram=0,
        skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True,
        resource_family=PREFIX + "/eval",
        env={"OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1"})


def submit(name, rows):
    record = HERE / (name + "_submission.json")
    if record.exists():
        return
    jobs = HERE / (name + "_jobs.json")
    jobs.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    output = subprocess.check_output([sys.executable, SCHEDULER, "submit-jsonl", "--file",
                                      str(jobs), "--trusted", "--json"], text=True)
    record.write_text(output)
    print("Submitted %s: %d fixed cases" % (name, len(rows)), flush=True)


def publish_mc(seed):
    code = '''
import json
from pathlib import Path
root=Path(ROOT)
folder=root/'train'/('seed%d'%SEED)
record=json.loads((folder/'results/ready.json').read_text())
target=root/'policies'/('mcwm_aligned_seed%d.pt'%SEED)
target.parent.mkdir(exist_ok=True)
if not target.exists():target.symlink_to(folder/'policy.pt')
records=root/'training_records'
records.mkdir(exist_ok=True)
(records/('seed%d.json'%SEED)).write_text(json.dumps(record,indent=2)+chr(10))
print(json.dumps({'seed':SEED,'source_events':record['source_events'],'updates':record['continuation_updates']}))
'''.replace("ROOT", repr(REMOTE)).replace("SEED", str(seed))
    record = read_server(code)
    names = ["policies/mcwm_aligned_seed%d.pt" % seed, "training_records/seed%d.json" % seed]
    source = shlex.join(["tar", "-chzf", "-", "-C", REMOTE, *names])
    destination = "tar -xzf - -C " + shlex.quote(str(HERE))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda node: stream("node001", node, source, destination), LAB_NODES))
    print("Published MC-WM actor to every evaluation pool: " + json.dumps(record), flush=True)


def main():
    deployment = json.loads((HERE / "evaluation_deployment.json").read_text())
    if not deployment["passenger_inputs_match"]:
        raise RuntimeError("Evaluation nodes do not have matched passenger inputs")
    wait(PREFIX + "/replay")
    replay = read_server("import json; from pathlib import Path; d=json.loads(Path(" +
        repr(REMOTE + "/replay/h2oplus_final_seed6201.json") +
        ").read_text()); print(json.dumps(d['replay_check']))")
    (HERE / "replay_check.json").write_text(json.dumps(replay, indent=2) + "\n")
    if not replay["passed"]:
        raise RuntimeError("Original H2O+ full-episode replay differs")
    submit("baseline_evaluation", [specification(method, seed, case)
        for method in ("h2oplus_final", "sac_guarded")
        for seed in range(6201, 6206) for case in range(15)])
    deadline = time.monotonic() + 43200
    while time.monotonic() < deadline:
        pending = [seed for seed in range(6201, 6206)
                   if not (HERE / ("mc_seed%d_submission.json" % seed)).exists()]
        if not pending:
            break
        code = '''
import json
from pathlib import Path
root=Path(ROOT)
rows=[]
for seed in SEEDS:
 folder=root/'train'/('seed%d'%seed)/'results'
 ready=folder/'ready.json'
 progress=folder/'progress.json'
 path=ready if ready.exists() else progress
 record=json.loads(path.read_text()) if path.exists() else {}
 rows.append({'seed':seed,'ready':ready.exists(),'source_events':record.get('source_events'),
              'continuation_updates':record.get('continuation_updates')})
print(json.dumps(rows))
'''.replace("ROOT", repr(REMOTE)).replace("SEEDS", repr(pending))
        progress = read_server(code)
        (HERE / "training_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        for row in progress:
            if row["ready"]:
                seed = row["seed"]
                publish_mc(seed)
                submit("mc_seed%d" % seed, [specification("mcwm_aligned", seed, case) for case in range(15)])
        if all(row["ready"] for row in progress):
            break
        print("Waiting for fixed training endpoints: " + json.dumps(progress), flush=True)
        time.sleep(60)
    else:
        raise TimeoutError("MC-WM training has not completed within the workflow window")
    wait(PREFIX + "/eval/*")
    subprocess.run([sys.executable, str(HERE / "collect.py"),
                    "--node", "node001", "--node", "jtl110gpu", "--node", "jtl110gpu2",
                    "--node", "jtl311linux", "--analyze"], check=True)
    result = json.loads((HERE / "analysis.json").read_text())
    if result["status"] != "complete":
        raise RuntimeError("The fixed 225-case grid is incomplete; no final comparison is reported")
    print("Final-controller comparison complete: " + str(HERE / "结果.md"), flush=True)


if __name__ == "__main__":
    main()
