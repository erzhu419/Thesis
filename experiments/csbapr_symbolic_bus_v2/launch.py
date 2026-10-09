"""Freeze sources, stage small code and launch independent CPU shards."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from settings import HERE, EXPERIMENTS, FROZEN, P, module, write

sys.path.insert(0, str(EXPERIMENTS / "enhancement_v1"))
infra = module("symbolic_bus_infra", EXPERIMENTS / "enhancement_v1/deploy.py")


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def gate(label):
    return HERE / "gates" / (label.replace("/", "_")+".json")


def stage():
    source = EXPERIMENTS.parents[1] / "CS-BAPR/experiments/random_load_validation_v8_20261001"
    if not FROZEN.exists():
        shutil.copytree(source / "runtime/csbapr", FROZEN / "csbapr",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runtime = json.loads((EXPERIMENTS / "reward_cross_v1/deployment/node_runtime.json").read_text())
    write(HERE / "node_runtime.json", runtime)
    target = remote()
    infra.scheduler.run_on("node007", "mkdir -p "+shlex.quote(target), timeout=40, check=True)
    files = sorted(HERE.glob("*.py"))+[HERE / "protocol.json", HERE / "node_runtime.json"]
    connection = ["-e", infra.scheduler._ssh_rsync_shell_for_node("node007")]
    destination = infra.scheduler._ssh_target_for_node("node007")+":"+target+"/"
    subprocess.run(["rsync", "-az", "--timeout=40", *connection, *map(str, files), destination], check=True, timeout=90)
    subprocess.run(["rsync", "-az", "--timeout=40", "--exclude=__pycache__", *connection,
                    str(FROZEN), destination], check=True, timeout=90)
    # Only the sparse fitting subprocess needs these wheels. Simulator runtime stays intact.
    wheels = [p for p in (source / "wheelhouse").glob("*.whl") if p.name.startswith(("pysindy-", "derivative-"))]
    infra.scheduler.run_on("node007", "mkdir -p "+shlex.quote(target+"/wheelhouse"), timeout=35, check=True)
    subprocess.run(["rsync", "-az", "--timeout=40", *connection, *map(str, wheels), destination+"wheelhouse/"], check=True, timeout=120)
    code = (f"{infra.executable('node007')} -m venv --system-site-packages {shlex.quote(target+'/fit_env')} && "
            f"{shlex.quote(target+'/fit_env/bin/python')} -m pip install --no-deps --no-index --find-links {shlex.quote(target+'/wheelhouse')} "
            "pysindy==2.1.0 derivative==0.6.3 && "
            f"ln -sfn {shlex.quote(infra.executable('node007'))} {shlex.quote(target+'/node_python')}")
    rc, out, err = infra.scheduler.run_on("node007", code, timeout=180, check=False)
    print(out[-1600:], err[-400:])
    if rc != 0:
        raise RuntimeError("Isolated fitting runtime setup failed")


def task(phase, *, seed=64001, objective="headway", arm="centered_symbolic", day=0, case=0, smoke=False):
    prefix = "smoke/" if smoke else ""
    if phase == "collect":
        label = f"day/{seed}/{day}"
    elif phase == "fit":
        label = prefix+f"model/{seed}"
    else:
        label = prefix+f"{objective}/{arm}/{seed}/{phase}/{case}"
    cmd = (f"{HERE}/node_python -u {HERE}/node_run.py --phase {phase} --seed {seed} --objective {objective} "
           f"--arm {arm} --day {day} --case {case}"+(" --smoke" if smoke else ""))
    waits = []
    if phase == "fit":
        waits += [f"day/{seed}/{d}" for d in ([0, 3] if smoke else range(4))]
    if phase in ("train", "evaluate") and arm in ("centered_symbolic", "uncentered_symbolic", "fitted_reference"):
        waits.append(prefix+f"model/{seed}")
    if phase == "evaluate" and arm in P["arms"]:
        waits.append(prefix+f"policy/{objective}/{arm}/{seed}")
    if not smoke and phase != "collect":
        waits.append("smoke-pass")
    spec = dict(project="Thesis", description="CS symbolic bus "+label,
        signature=P["name"]+"/"+label, cwd=str(HERE), cmd=cmd,
        allowed_nodes=P["nodes"], cpu=1, ram_mb=2500, vram=0,
        env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        resource_family=P["name"]+"/"+phase, ram_resource_family=P["name"]+"/"+phase,
        vram_resource_family=P["name"]+"/"+phase, skip_launch_staging=True,
        stage_excludes=["identification", "models", "smoke_models", "runs", "smoke", "fit_env", "wheelhouse", "gates"],
        allow_no_ckpt=True, allow_no_resume=True, allow_cpu_training=True,
        cpu_training_justification="Single-thread 64-unit bus networks and independent simulator days; shared artifacts work on any allowed CPU node.")
    if waits:
        spec["wait_for_files"] = [str(gate(x)) for x in waits]
    return spec


def jobs():
    rows = [task("collect", seed=seed, day=day) for seed in P["training_seeds"] for day in range(4)]
    for smoke in (True, False):
        seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
        for seed in seeds:
            rows.append(task("fit", seed=seed, smoke=smoke))
            for objective in P["objectives"]:
                for arm in P["arms"]:
                    fields = dict(objective=objective, arm=arm, seed=seed, smoke=smoke)
                    rows.append(task("train", **fields))
                    rows += [task("evaluate", case=c, **fields) for c in ([0] if smoke else range(9))]
                rows += [task("evaluate", seed=seed, objective=objective, arm="fitted_reference", case=c, smoke=smoke)
                         for c in ([0] if smoke else range(9))]
        for arm in ("daganzo", "zero"):
            rows += [task("evaluate", arm=arm, case=c, smoke=smoke) for c in ([0] if smoke else range(9))]
    return rows


def collect():
    infra.scheduler.run_on("node007", infra.executable("node007")+" -u "+shlex.quote(remote()+"/collect_results.py"), timeout=90, check=True)
    subprocess.run(["rsync", "-az", "--timeout=30", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/summary.json", str(HERE)+"/"], check=True, timeout=60)
    result = json.loads((HERE / "summary.json").read_text())
    for label in result["ready"]:
        if not gate(label).exists():
            write(gate(label), dict(ready=label, analysis_location="server"))
    return result


def watch():
    deadline = time.monotonic()+72*3600
    while time.monotonic() < deadline:
        try:
            report = collect()
            print({k:report[k] for k in ("smoke_evaluations", "formal_evaluations", "complete")}, flush=True)
            if report["complete"]:
                return
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-500:], flush=True)
        time.sleep(40)
    raise TimeoutError("Inspect failed tasks; checkpoints stay on the server")


def submit():
    if (HERE / "submission.json").exists():
        raise ValueError("Submission already exists")
    rows = jobs()
    rows.append(dict(project="Thesis", description="CS symbolic bus compact gate handoff",
        signature=P["name"]+"/watch", cwd=str(HERE), cmd=f"{sys.executable} -u {HERE}/launch.py watch",
        require_node="local", cpu=1, ram_mb=256, vram=0,
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True))
    write(HERE / "jobs.json", rows)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(HERE / "jobs.json"), "--trusted", "--json"], text=True))
    write(HERE / "submission.json", result)
    print(dict(submitted=len(result["submitted"]), first=result["submitted"][0]["id"], last=result["submitted"][-1]["id"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "submit", "watch", "collect"))
    globals()[parser.parse_args().mode]()
