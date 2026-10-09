"""Unpinned shared-storage jobs and compact local readiness gates."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from cross_common import HERE, EXPERIMENTS, P, module, write
from collect_results import placements

infra = module("reward_cross_infra", EXPERIMENTS / "enhancement_v1/deploy.py")


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def key(item):
    return f"{item['kind']}_{item['objective']}_{item['scene']}_{item['arm']}_{item['seed']}"


def gate(item, initialize=False):
    prefix = "smoke_" if item["smoke"] else ""
    name = f"init_{item['objective']}_{item['seed']}" if initialize else key(item)
    return HERE / "gates" / (prefix+name+".json")


def task(item, *, case=None, initialize=False):
    phase = "init" if initialize else f"eval{case}" if case is not None else "train"
    label = ("smoke/" if item["smoke"] else "runs/")+key(item)+"/"+phase
    cmd = f"{HERE}/node_python -u {HERE}/node_run.py --kind {item['kind']} --objective {item['objective']} --seed {item['seed']}"
    if initialize:
        cmd += " --initialize"
    else:
        cmd += f" --arm {item['arm']} --scenario {item['scene']}"
    if case is not None:
        cmd += f" --evaluate --case {case}"
    if item["smoke"]:
        cmd += " --smoke"
    family = P["name"]+f"/{item['kind']}/"+("eval" if case is not None else "smoke" if item["smoke"] else "train")
    row = dict(project="Thesis", description="Reward cross "+label, signature=P["name"]+"/"+label,
        cwd=str(HERE), cmd=cmd, allowed_nodes=P["training_nodes"], cpu=1,
        ram_mb=2500 if case is not None else 4500 if item["smoke"] and item["kind"] == "mc" else 11000 if item["kind"] == "mc" else 2500,
        vram=0, env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        resource_family=family, ram_resource_family=family, vram_resource_family=family,
        skip_launch_staging=True, stage_excludes=["runs", "smoke", "inputs", "smoke_inputs", "gates"],
        allow_no_ckpt=True, allow_no_resume=True, allow_cpu_training=True,
        cpu_training_justification="Explicit single-thread CPU training or policy evaluation; method names in evaluation signatures can match the scheduler's training heuristic.")
    if case is not None:
        row["wait_for_files"] = [str(gate(item))]
    elif item["kind"] == "mc" and not initialize:
        row["wait_for_files"] = [str(gate(item, True))]
    return row


def jobs(smoke=False):
    rows = []
    seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
    for objective in P["objectives"]:
        for seed in seeds:
            rows.append(task(dict(kind="mc", objective=objective, seed=seed, scene="normal", arm="initializer", smoke=smoke), initialize=True))
    for item in placements(smoke):
        rows.append(task(item))
        rows.extend(task(item, case=c) for c in ([0] if smoke else range(10 if item["kind"] == "mc" else 9)))
    return rows


def stage():
    infra.sync("node007", str(HERE)+"/", remote(), ["--delete", "--include=*.py", "--include=protocol.json", "--exclude=*"])
    runtime = dict(train=dict(python=infra.executable("node007"), env=infra.environment("node007")),
        evaluate=dict(python=infra.executable("node001"), env=infra.environment("node001", True)))
    path = HERE / "deployment/node_runtime.json"
    write(path, runtime)
    infra.sync("node007", path, remote())
    infra.scheduler.run_on("node007", "ln -sfn "+shlex.quote(infra.executable("node007"))+" "+shlex.quote(remote()+"/node_python"), timeout=40, check=True)
    required = [EXPERIMENTS / "mcwm_alignment_v2/inputs/h2oplus_seed6201.pt",
        EXPERIMENTS / "mcwm_seed_confirmation_v13/run.py", EXPERIMENTS / "legacy_bus/train_representation.py",
        EXPERIMENTS / "enhancement_v1/runtime/cs/CS-BAPR/csbapr/agent.py"]
    infra.scheduler.run_on("node007", "test "+" -a ".join("-s "+shlex.quote(remote(p)) for p in required), timeout=40, check=True)
    print("Staged reward selector and protocol only; existing data/runtimes stay on the server", flush=True)


def collect(smoke=False):
    command = f"{infra.executable('node007')} -u {remote()}/collect_results.py"+(" --smoke" if smoke else "")
    infra.scheduler.run_on("node007", command, timeout=120, check=True)
    name = "smoke_summary.json" if smoke else "summary.json"
    subprocess.run(["rsync", "-az", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/"+name, str(HERE)+"/"], check=True, timeout=60)
    report = json.loads((HERE / name).read_text())
    assert report["experiment"] == P["name"] and report["analysis_location"] == "server"
    for item in report["initializers"]:
        path = gate(dict(item, smoke=smoke), True)
        if not path.exists():
            write(path, item)
    for item in report["policies"]:
        path = gate(item)
        if not path.exists():
            write(path, item)
    return report


def watch(smoke=False):
    deadline = time.monotonic()+48*3600
    while time.monotonic() < deadline:
        try:
            report = collect(smoke)
            print(json.dumps({k:report[k] for k in ("initialization_ready", "training_ready", "evaluation_ready", "complete")}), flush=True)
            if report["complete"]:
                print("DONE reward cross handoff", flush=True)
                return
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-400:], flush=True)
        time.sleep(45)
    raise TimeoutError("Reward crossover campaign incomplete; inspect failed training/evaluation logs")


def submit(smoke=False):
    name = "smoke" if smoke else "campaign"
    output = HERE / (name+"_submission.json")
    if output.exists():
        raise ValueError("Submission already recorded: "+str(output))
    rows = jobs(smoke)
    rows.append(dict(project="Thesis", description="Reward cross compact JSON handoff "+name,
        signature=P["name"]+"/"+name+"/watch", cwd=str(HERE),
        cmd=f"python3 -u {HERE}/launch.py {'watch-smoke' if smoke else 'watch'}", require_node="local",
        cpu=1, ram_mb=256, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True))
    path = HERE / (name+"_jobs.json")
    write(path, rows)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(path), "--trusted", "--json"], text=True))
    write(output, result)
    print(json.dumps(dict(batch=name, submitted=len(result["submitted"]), first=result["submitted"][0]["id"], last=result["submitted"][-1]["id"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "smoke", "submit", "collect", "collect-smoke", "watch", "watch-smoke"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode.startswith("collect"):
        print(json.dumps({k:v for k,v in collect(mode == "collect-smoke").items() if k.endswith("ready") or k == "complete"}))
    elif mode.startswith("watch"):
        watch(mode == "watch-smoke")
    else:
        submit(mode == "smoke")
