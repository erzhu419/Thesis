"""Independent CPU shards and per-family smoke gates on shared server storage."""

import argparse
import json
import shlex
import subprocess
import sys
import time

from common import HERE, EXPERIMENTS, P, label, module, placements, write

infra = module("legacy_dual_infra", EXPERIMENTS / "enhancement_v1/deploy.py")


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def marker(name):
    return HERE / "gates" / (name.replace("/", "_")+".json")


def task(item, phase, case=0):
    name = label(item)+("/train" if phase == "train" else f"/eval/{case}")
    command = f"{infra.PYTHON} -u {HERE}/node_run.py --kind {item['kind']} --phase {phase} --mode {item['mode']} --objective {item['objective']} --arm {item['arm']} --seed {item['seed']}"
    if phase == "evaluate":
        command += f" --case {case}"
    if item.get("smoke"):
        command += " --smoke"
    threads = P["threads"][item["kind"]] if phase == "train" else 1
    waits = [] if item.get("smoke") else [marker("smoke-pass/"+item["kind"])]
    if phase == "evaluate" and item["arm"] != "fixed_speed":
        waits.append(marker(label(item)))
    return dict(project="Thesis", description="RE/CS dual "+name, signature=P["name"]+"/"+name,
        cwd=str(HERE), cmd=command, allowed_nodes=P["nodes"], cpu=threads,
        ram_mb=3000 if phase == "train" else 1100, vram=0,
        resource_family=P["name"]+"/"+item["kind"]+"/"+phase,
        env=dict(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1"),
        wait_for_files=list(map(str, waits)), skip_launch_staging=True,
        allow_no_ckpt=True, allow_no_resume=True, allow_cpu_training=True,
        cpu_training_justification="Existing compact bus learners; independently parallel paired seeds. Four CPU threads for the ten-member RE-SAC update and one for each CS-BAPR learner.")


def jobs():
    rows = []
    for kind in P["arms"]:
        rows.append(dict(project="Thesis", description="Dual execution contracts "+kind,
            signature=P["name"]+"/contracts/"+kind, cwd=str(HERE),
            cmd=f"{infra.PYTHON} -u {HERE}/node_run.py --contracts --kind {kind}",
            allowed_nodes=P["nodes"], cpu=1, ram_mb=2600, vram=0,
            env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True, allow_cpu_training=True,
            cpu_training_justification="Finite CPU action and learner contracts"))
    for smoke in (True, False):
        for item in placements(smoke):
            rows.append(task(item, "train"))
            rows.extend(task(item, "evaluate", case) for case in ([0] if smoke else range(9)))
    for kind in P["arms"]:
        item = dict(kind=kind, mode="dual", objective="headway", arm="fixed_speed", seed=0, smoke=False)
        rows.extend(task(item, "evaluate", case) for case in range(9))
    rows.append(dict(project="Thesis", description="RE/CS dual compact dependency handoff",
        signature=P["name"]+"/watch", cwd=str(HERE),
        cmd=f"{sys.executable} -u {HERE}/launch.py watch", require_node="local",cpu=1,ram_mb=256,vram=0,
        skip_launch_staging=True,allow_no_ckpt=True,allow_no_resume=True))
    return rows


def stage():
    infra.scheduler.run_on("node007", "mkdir -p "+shlex.quote(remote()), timeout=40, check=True)
    files = sorted(HERE.glob("*.py"))+[HERE / "protocol.json"]
    subprocess.run(["rsync","-az","--timeout=35","-e",infra.scheduler._ssh_rsync_shell_for_node("node007"),
        *map(str,files),infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/"],check=True,timeout=90)


def collect():
    code = infra.PYTHON+" -u "+shlex.quote(remote()+"/collect_results.py")
    rc,out,err = infra.scheduler.run_on("node007",code,timeout=90,check=False)
    if rc:
        raise RuntimeError((out+err)[-2000:])
    subprocess.run(["rsync","-az","--timeout=35","-e",infra.scheduler._ssh_rsync_shell_for_node("node007"),
        infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/summary.json",str(HERE)+"/"],check=True,timeout=60)
    report = json.loads((HERE / "summary.json").read_text())
    for name in report["ready"]:
        if not marker(name).exists():
            write(marker(name),dict(ready=name))
    complete = {P["name"]+"/"+name for name in report["completed_labels"]}
    corrected = []
    with infra.scheduler.state_lock(purpose="reconcile verified completed dual bus shards"):
        state = infra.scheduler.load_state()
        for row in state["tasks"]:
            reason = row.get("last_block_reason") or ""
            quiet = row.get("status")=="failed" and ("log only" in reason or "no success marker" in reason)
            pending = row.get("status")=="running" and "process exited; terminal log/result diagnosis pending" in reason
            if row.get("signature") in complete and row.get("exit_code")==0 and (quiet or pending):
                row.update(status="done",last_block_reason="",failure_category=None,
                    completion_verified_from="Validated resident dual result and zero process exit")
                if not row.get("finished_at"):
                    row["finished_at"] = time.time()
                corrected.append(row["id"])
        if corrected:
            infra.scheduler.save_state(state)
    if corrected:
        print(dict(completion_statuses_corrected=corrected),flush=True)
    return report


def watch():
    deadline = time.monotonic()+48*3600
    while time.monotonic()<deadline:
        try:
            report = collect()
            print({k:report[k] for k in ("training_complete","dual_evaluations","fixed_speed_evaluations","complete")},flush=True)
            if report["complete"]:
                print("DONE RE/CS dual campaign",flush=True)
                return
        except (RuntimeError,subprocess.SubprocessError) as error:
            print(str(error)[-600:],flush=True)
        time.sleep(40)
    raise TimeoutError("Inspect incomplete resident RE/CS shards")


def submit():
    output = HERE / "submission.json"
    if output.exists():
        raise ValueError("This campaign was already submitted")
    rows = jobs()
    path = HERE / "jobs.json"
    write(path,rows)
    result = json.loads(subprocess.check_output(["python3",infra.SCHEDULER,"submit-jsonl","--file",str(path),"--trusted","--json"],text=True))
    write(output,result)
    print(dict(submitted=len(result["submitted"]),first=result["submitted"][0]["id"],last=result["submitted"][-1]["id"]),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("stage","submit","collect","watch"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "submit":
        submit()
    elif mode == "collect":
        report = collect()
        print({k:report[k] for k in ("training_complete","dual_evaluations","fixed_speed_evaluations","reused_holding_evaluations","complete")})
    else:
        watch()
