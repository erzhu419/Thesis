"""Two independent smoke gates and freely placed CPU experiment shards."""

import argparse
from itertools import product
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from repair_common import HERE, EXPERIMENTS, P, module, write

sys.path.insert(0, str(EXPERIMENTS / "enhancement_v1"))
infra = module("repair_infra", EXPERIMENTS / "enhancement_v1/deploy.py")


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def gate(label):
    return HERE / "gates" / (label.replace("/", "_") + ".json")


def task(kind, arguments, label, waits=(), training=False):
    spec = dict(project="Thesis", description="Bus repair v3 " + label,
        signature=P["name"] + "/" + label, cwd=str(HERE),
        cmd=f"{infra.PYTHON} -u {HERE}/node_run.py {kind} {arguments}",
        allowed_nodes=P["nodes"], cpu=1, ram_mb=1800 if training else 1200, vram=0,
        resource_family=P["name"] + "/" + kind + ("/train" if training else "/eval"),
        env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True,
        allow_cpu_training=True, cpu_training_justification="Independent small bus networks and SUMO days; each shard uses one CPU thread and shared existing inputs.")
    if waits:
        spec["wait_for_files"] = [str(gate(x)) for x in waits]
    return spec


def jobs():
    rows = []
    for smoke in (True, False):
        prefix = "smoke/" if smoke else ""
        seeds = P["cs_seeds"][:1] if smoke else P["cs_seeds"]
        for objective, seed in product(P["cs_objectives"], seeds):
            for arm in P["cs_arms"]:
                label = prefix + f"cs/{objective}/{arm}/{seed}"
                fields = f"--objective {objective} --arm {arm} --seed {seed}" + (" --smoke" if smoke else "")
                rows.append(task("cs", "--phase train " + fields, label + "/train",
                                 [] if smoke else ["cs-smoke-pass"], True))
                for case in ([0] if smoke else range(9)):
                    rows.append(task("cs", f"--phase evaluate --case {case} " + fields,
                        label + f"/eval/{case}", [label + "/policy"]))
            for case in ([0] if smoke else range(9)):
                label = prefix + f"cs/{objective}/fitted_reference/{seed}/eval/{case}"
                rows.append(task("cs", f"--phase evaluate --objective {objective} --arm fitted_reference --seed {seed} --case {case}" + (" --smoke" if smoke else ""), label,
                    [] if smoke else ["cs-smoke-pass"]))
    for branch in ("old", "new"):
        seeds = P["mc_old_seeds"] if branch == "old" else P["mc_new_seeds"]
        arms = P["mc_old_arms"] if branch == "old" else P["mc_new_arms"]
        cells = [("holding", "passenger")] if branch == "old" else P["mc_cells"]
        layer = "direct" if branch == "old" else "guarded"
        for scene, seed, arm, (mode, objective) in product(P["mc_scenes"], seeds, arms, cells):
            fields = f"--branch {branch} --mode {mode} --objective {objective} --arm {arm} --scene {scene} --seed {seed} --layer {layer}"
            for case in range(10):
                label = f"mc/{branch}/{mode}/{objective}/{scene}/{arm}/{seed}/{layer}/{case}"
                rows.append(task("mc", fields + f" --case {case}", label, ["mc-smoke-pass"]))
    for arm, layer in [("daganzo_duet", "guarded"), ("speed1.1", "direct"), ("speed1.2", "direct")]:
        for case in range(10):
            rows.append(task("mc", f"--branch rule --arm {arm} --layer {layer} --case {case}",
                f"mc/rule/{arm}/{layer}/{case}", ["mc-smoke-pass"]))
    for branch, arm, mode, objective, seed, layer in [("old", "fixed0250", "holding", "passenger", 63001, "guarded"),
            ("new", "mcwm", "dual", "headway", 65001, "guarded"),
            ("new", "mcwm", "dual", "passenger", 65001, "guarded"),
            ("rule", "daganzo_duet", "holding", "passenger", 0, "guarded")]:
        rows.append(task("mc", f"--branch {branch} --arm {arm} --mode {mode} --objective {objective} --seed {seed} --layer {layer} --case 0 --smoke",
            f"smoke/mc/{branch}/{objective}/{arm}"))
    return rows


def stage():
    infra.scheduler.run_on("node007", "mkdir -p " + shlex.quote(remote()), timeout=40, check=True)
    files = sorted(HERE.glob("*.py")) + [HERE / "protocol.json"]
    subprocess.run(["rsync", "-az", "--timeout=35", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        *map(str, files), infra.scheduler._ssh_target_for_node("node007") + ":" + remote() + "/"], check=True, timeout=90)


def collect():
    rc, out, err = infra.scheduler.run_on("node007", infra.PYTHON + " -u " + shlex.quote(remote() + "/collect_results.py"), timeout=90, check=False)
    if rc != 0:
        raise RuntimeError((out + err)[-1500:])
    subprocess.run(["rsync", "-az", "--timeout=30", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        infra.scheduler._ssh_target_for_node("node007") + ":" + remote() + "/summary.json", str(HERE) + "/"], check=True, timeout=60)
    summary = json.loads((HERE / "summary.json").read_text())
    for label in summary["ready"]:
        if not gate(label).exists():
            write(gate(label), dict(ready=label))
    return summary


def watch():
    deadline = time.monotonic() + 72*3600
    while time.monotonic() < deadline:
        try:
            report = collect()
            print({k: report[k] for k in ("cs_training", "cs_evaluations", "mc_evaluations", "complete")}, flush=True)
            if report["complete"]:
                return
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-500:], flush=True)
        time.sleep(40)
    raise TimeoutError("Inspect failed repair tasks; original artifacts are unchanged")


def submit():
    if (HERE / "submission.json").exists():
        raise ValueError("Repair campaign is already submitted")
    rows = jobs()
    rows.append(dict(project="Thesis", description="Bus repair v3 compact gate handoff",
        signature=P["name"] + "/watch", cwd=str(HERE), cmd=f"{sys.executable} -u {HERE}/launch.py watch",
        require_node="local", cpu=1, ram_mb=256, vram=0, skip_launch_staging=True,
        allow_no_ckpt=True, allow_no_resume=True))
    write(HERE / "jobs.json", rows)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(HERE / "jobs.json"), "--trusted", "--json"], text=True))
    write(HERE / "submission.json", result)
    print(dict(submitted=len(result["submitted"]), first=result["submitted"][0]["id"], last=result["submitted"][-1]["id"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "submit", "collect", "watch"))
    globals()[parser.parse_args().mode]()
