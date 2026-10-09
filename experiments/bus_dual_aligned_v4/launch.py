"""Independent speed attribution and dependency-gated paired continuations."""

import argparse
from itertools import product
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from aligned_common import HERE, EXPERIMENTS, P, write
sys.path.insert(0, str(EXPERIMENTS / "bus_control_repair_v3"))
sys.path.insert(0, str(EXPERIMENTS / "enhancement_v1"))
from repair_common import module
infra = module("dual_aligned_infra", EXPERIMENTS / "enhancement_v1/deploy.py")


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def task(label, command, training=False, waits=()):
    threads = P["threads"] if training else 1
    row = dict(project="Thesis", description="Bus dual aligned v4 " + label,
        signature=P["name"] + "/" + label, cwd=str(HERE),
        cmd=f"{infra.PYTHON} -u {HERE}/node_run.py {command}",
        allowed_nodes=P["nodes"], cpu=threads, ram_mb=2600 if training else 1200, vram=0,
        resource_family=P["name"] + ("/train" if training else "/eval"),
        env=dict(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS=str(threads)),
        skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True,
        allow_cpu_training=True, cpu_training_justification="Paired global bus networks; shared compact replay and four CPU threads per independent training shard.")
    if waits:
        row["wait_for_files"] = [str(HERE / "gates" / (x.replace("/", "_")+".json")) for x in waits]
    return row


def diagnosis_jobs():
    rows = []
    for scene, seed, arm, objective, case in product(P["scenes"], [65001,65002], ["h2oplus","mcwm"], P["objectives"], range(10)):
        fields = f"--branch new --objective {objective} --arm {arm} --scene {scene} --seed {seed} --case {case} --speed 1.0"
        rows.append(task(f"diagnosis/new/{objective}/{scene}/{arm}/{seed}/{case}", "diagnose_speed "+fields))
    for scene, seed, arm, speed, case in product(P["scenes"], [63001,63002], ["h2o_matched","fixed0250"], [1.1,1.2], range(10)):
        fields = f"--branch old --mode holding --objective passenger --arm {arm} --scene {scene} --seed {seed} --case {case} --speed {speed}"
        rows.append(task(f"diagnosis/old/{scene}/{arm}/{seed}/{speed}/{case}", "diagnose_speed "+fields))
    return rows


def executed_history_jobs():
    rows = []
    for scene,seed,arm,objective,case in product(P["scenes"],[65001,65002],["h2oplus","mcwm"],P["objectives"],range(10)):
        fields = f"--branch new_executed --objective {objective} --arm {arm} --scene {scene} --seed {seed} --case {case} --speed 1.0"
        rows.append(task(f"diagnosis/new_executed/{objective}/{scene}/{arm}/{seed}/{case}","diagnose_speed "+fields))
    return rows


def aligned_jobs(smoke):
    prefix = "smoke/" if smoke else "formal/"
    outer = [] if smoke else ["smoke-pass"]
    shards = 2 if smoke else P["offline_shards"]
    extra = " --smoke" if smoke else ""
    rows = [task(prefix+f"data/shard{i}",f"run --phase collect --shard {i}"+extra,waits=outer) for i in range(shards)]
    rows.append(task(prefix+"data/merge","run --phase merge"+extra,waits=[prefix+f"data/shard{i}" for i in range(shards)]+outer))
    seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
    scenes = P["scenes"][:1] if smoke else P["scenes"]
    for objective,seed in product(P["objectives"],seeds):
        init = prefix+f"init/{objective}/{seed}"
        rows.append(task(init,f"run --phase initialize --objective {objective} --seed {seed}"+extra,
                         training=True,waits=[prefix+"data/merge"]+outer))
        for scene,arm in product(scenes,P["arms"]):
            label = prefix+f"policy/{objective}/{scene}/{arm}/{seed}"
            fields = f"--objective {objective} --scene {scene} --arm {arm} --seed {seed}"+extra
            rows.append(task(label,"run --phase train "+fields,training=True,waits=[init]+outer))
            for layer,case in product(P["executors"],[0] if smoke else range(10)):
                rows.append(task(label+f"/eval/{layer}/{case}",f"run --phase evaluate --layer {layer} --case {case} "+fields,waits=[label]+outer))
    for arm,layer,case in product(["daganzo_duet","zero"],P["executors"],[0] if smoke else range(10)):
        rows.append(task(prefix+f"rule/{arm}/{layer}/{case}",f"run --phase evaluate --arm {arm} --layer {layer} --case {case}"+extra,waits=outer))
    return rows


def submit(rows, name):
    path = HERE / (name+"_submission.json")
    if path.exists():
        raise ValueError("Campaign already submitted: "+name)
    jobs = HERE / (name+"_jobs.json")
    write(jobs, rows)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(jobs), "--trusted", "--json"], text=True))
    write(path, result)
    print(dict(campaign=name, submitted=len(result["submitted"]), first=result["submitted"][0]["id"], last=result["submitted"][-1]["id"]))


def stage():
    infra.scheduler.run_on("node007", "mkdir -p "+shlex.quote(remote()), timeout=40, check=True)
    files = sorted(HERE.glob("*.py")) + [HERE / "protocol.json"]
    subprocess.run(["rsync", "-az", "--timeout=35", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        *map(str, files), infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/"], check=True, timeout=90)


def collect():
    rc,out,err = infra.scheduler.run_on("node007",infra.PYTHON+" -u "+shlex.quote(remote()+"/summarize.py"),timeout=90,check=False)
    if rc:
        raise RuntimeError((out+err)[-1500:])
    subprocess.run(["rsync","-az","--timeout=30","-e",infra.scheduler._ssh_rsync_shell_for_node("node007"),
        infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/summary.json",str(HERE)+"/"],check=True,timeout=60)
    report = json.loads((HERE / "summary.json").read_text())
    for label in report["ready"]:
        marker = HERE / "gates" / (label.replace("/","_")+".json")
        if not marker.exists():
            write(marker,dict(ready=label))
    completed = {P["name"]+"/"+label for label in report["completed_labels"]}
    corrected = []
    with infra.scheduler.state_lock(purpose="reconcile verified completed bus tasks"):
        state = infra.scheduler.load_state()
        for t in state["tasks"]:
            reason = t.get("last_block_reason") or ""
            quiet_failure = t.get("status")=="failed" and ("log only" in reason or "no success marker" in reason)
            pending_diagnosis = t.get("status")=="running" and "process exited; terminal log/result diagnosis pending" in reason
            if (t.get("signature") in completed and t.get("exit_code")==0
                    and (quiet_failure or pending_diagnosis)):
                t["status"] = "done"
                t["last_block_reason"] = ""
                t["failure_category"] = None
                if not t.get("finished_at"):
                    t["finished_at"] = time.time()
                t["completion_verified_from"] = "Validated resident result and zero process exit; reconciled terminal diagnosis"
                corrected.append(t["id"])
        if corrected:
            infra.scheduler.save_state(state)
    if corrected:
        print(dict(completion_statuses_corrected=corrected),flush=True)
    return report


def watch():
    deadline = time.monotonic()+72*3600
    while time.monotonic()<deadline:
        try:
            report = collect()
            print({k:report[k] for k in ("smoke_complete","formal_initializers","formal_training","formal_evaluations","complete")},flush=True)
            if report["complete"]:
                return
        except (RuntimeError,subprocess.SubprocessError) as error:
            print(str(error)[-400:],flush=True)
        time.sleep(40)
    raise TimeoutError("Inspect incomplete aligned shards; no data or weights need local download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "diagnose", "executed-history", "submit", "collect", "watch"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "diagnose":
        submit(diagnosis_jobs(), "diagnosis")
    elif mode == "executed-history":
        submit(executed_history_jobs(),"executed_history")
    elif mode == "submit":
        rows = aligned_jobs(True)+aligned_jobs(False)
        rows.append(dict(project="Thesis",description="Bus dual aligned v4 compact gate handoff",
            signature=P["name"]+"/watch",cwd=str(HERE),cmd=f"{sys.executable} -u {HERE}/launch.py watch",
            require_node="local",cpu=1,ram_mb=256,vram=0,skip_launch_staging=True,allow_no_ckpt=True,allow_no_resume=True))
        submit(rows,"aligned")
    elif mode == "collect":
        print(json.dumps(collect(),ensure_ascii=False))
    else:
        watch()
