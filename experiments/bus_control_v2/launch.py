"""Shared-storage CPU jobs, smoke gates, and parallel independent experiment cells."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from station_common import HERE, EXPERIMENTS, P, write

sys.path.insert(0, str(EXPERIMENTS / "reward_cross_v1"))
sys.path.insert(0, str(EXPERIMENTS / "enhancement_v1"))
spec = importlib.util.spec_from_file_location("bus_control_infra", EXPERIMENTS / "enhancement_v1/deploy.py")
infra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infra)
sys.path.insert(0, str(HERE))


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def gate(label):
    return HERE / "gates" / (label.replace("/", "_")+".json")


def task(phase, *, mode="dual", objective="headway", seed=None, scene="calibrated", arm="h2oplus", shard=0, case=0, smoke=False):
    seed = seed or P["training_seeds"][0]
    prefix = "smoke/" if smoke else ""
    cell = f"{mode}/{objective}/{scene}/{arm}_seed{seed}"
    cmd = f"{HERE}/node_python -u {HERE}/node_run.py --phase {phase} --mode {mode} --objective {objective} --seed {seed} --scenario {scene} --arm {arm} --shard {shard} --case {case}"
    if smoke:
        cmd += " --smoke"
    label = prefix + (f"shard/{shard}" if phase == "dual-data" else f"data/{phase}" if phase in ("holding-data", "merge") else
                      f"init/{mode}/{objective}/{seed}" if phase == "initialize" else cell+f"/{phase}/{case}")
    waits = []
    if phase == "merge":
        waits = [prefix+f"shard/{i}" for i in range(2 if smoke else P["offline_dual_shards"])]
    elif phase == "initialize":
        waits = [prefix+"data/"+mode]
    elif phase == "train":
        waits = [prefix+f"init/{mode}/{objective}/{seed}"]
    elif phase == "evaluate":
        waits = [prefix+"policy/"+cell] if arm in P["arms"] else [prefix+"data/holding"]
    if not smoke:
        waits.append("smoke-pass")
    row = dict(project="Thesis", description="Bus control v2 "+label, signature=P["name"]+"/"+label,
        cwd=str(HERE), cmd=cmd, allowed_nodes=P["nodes"], cpu=1, ram_mb=2500 if phase in ("dual-data", "evaluate") else 2200,
        vram=0, resource_family=P["name"]+"/"+phase, ram_resource_family=P["name"]+"/"+phase,
        vram_resource_family=P["name"]+"/"+phase,
        env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        allow_no_ckpt=True, allow_no_resume=True, allow_cpu_training=True,
        cpu_training_justification="Small station-level networks use measured single-thread CPU kernels; all seeds/cells and SUMO days are independent shared-storage shards.",
        skip_launch_staging=True, stage_excludes=["data", "smoke_data", "inputs", "smoke_inputs", "runs", "smoke", "gates"])
    if waits:
        row["wait_for_files"] = [str(gate(x)) for x in waits]
    return row


def jobs(smoke):
    rows = [task("holding-data", mode="holding", smoke=smoke)]
    rows += [task("dual-data", shard=i, smoke=smoke) for i in range(2 if smoke else P["offline_dual_shards"])]
    rows.append(task("merge", smoke=smoke))
    for mode, objective in P["cells"]:
        for seed in P["training_seeds"][:1] if smoke else P["training_seeds"]:
            rows.append(task("initialize", mode=mode, objective=objective, seed=seed, smoke=smoke))
            for scene in ["calibrated"] if smoke else [s["name"] for s in P["scenarios"]]:
                for arm in P["arms"]:
                    fields = dict(mode=mode, objective=objective, seed=seed, scene=scene, arm=arm, smoke=smoke)
                    rows.append(task("train", **fields))
                    rows += [task("evaluate", case=i, **fields) for i in ([0] if smoke else range(10))]
    for rule in ("zero", "daganzo_duet"):
        rows += [task("evaluate", mode="holding", arm=rule, case=i, smoke=smoke) for i in ([0] if smoke else range(10))]
    return rows


def stage():
    infra.scheduler.run_on("node007", "mkdir -p "+shlex.quote(remote()), timeout=40, check=True)
    runtime = json.loads((EXPERIMENTS / "reward_cross_v1/deployment/node_runtime.json").read_text())
    write(HERE / "node_runtime.json", runtime)
    files = sorted(HERE.glob("*.py")) + [HERE / "protocol.json", HERE / "node_runtime.json"]
    subprocess.run(["rsync", "-az", "--timeout=35", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        *map(str, files), infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/"], check=True, timeout=75)
    infra.scheduler.run_on("node007", "ln -sfn "+shlex.quote(infra.executable("node007"))+" "+shlex.quote(remote()+"/node_python"), timeout=40, check=True)


def collect():
    infra.scheduler.run_on("node007", infra.executable("node007")+" -u "+shlex.quote(remote()+"/collect_results.py"), timeout=90, check=True)
    subprocess.run(["rsync", "-az", "--timeout=30", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/summary.json", str(HERE)+"/"], check=True, timeout=60)
    summary = json.loads((HERE / "summary.json").read_text())
    for label in summary["ready"]:
        if not gate(label).exists():
            write(gate(label), dict(ready=label, analysis_location="server"))
    return summary


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
    raise TimeoutError("Inspect experiment failures; no dataset or weights need downloading")


def submit():
    path = HERE / "submission.json"
    if path.exists():
        raise ValueError("Submission already exists; do not duplicate the campaign")
    rows = jobs(True)+jobs(False)
    rows.append(dict(project="Thesis", description="Bus control v2 compact gate handoff",
        signature=P["name"]+"/watch", cwd=str(HERE), cmd=f"{sys.executable} -u {HERE}/launch.py watch",
        require_node="local", cpu=1, ram_mb=256, vram=0, skip_launch_staging=True,
        allow_no_ckpt=True, allow_no_resume=True))
    write(HERE / "jobs.json", rows)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(HERE / "jobs.json"), "--trusted", "--json"], text=True))
    write(path, result)
    print(dict(submitted=len(result["submitted"]), first=result["submitted"][0]["id"], last=result["submitted"][-1]["id"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "submit", "collect", "watch"))
    args = parser.parse_args()
    globals()[args.mode]()
