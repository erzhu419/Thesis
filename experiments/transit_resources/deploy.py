"""Stage a read-only evaluation copy and prepare Linux-only CPU shards."""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "TransitDuet" / "transit_duet"
REMOTE = str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work")
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
NODES = [f"node{i:03d}" for i in range(1, 7)]
ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MPLBACKEND="Agg", SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    runtime = HERE / "runtime"
    runtime.mkdir(exist_ok=True)
    for name in ("env", "upper", "lower", "coupling", "configs_ablation"):
        shutil.copytree(PROJECT / name, runtime / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.png", "*.csv"))
    for path in PROJECT.glob("*.py"):
        shutil.copy2(path, runtime / path.name)
    for path in PROJECT.glob("*.yaml"):
        shutil.copy2(path, runtime / path.name)
    (runtime / "scripts").mkdir(exist_ok=True)
    shutil.copy2(PROJECT / "scripts/heldout_final_eval.py", runtime / "scripts/heldout_final_eval.py")
    specs = []
    for shard in range(36):
        output = HERE / "runs" / f"shard{shard:02d}" / "results"
        specs.append(dict(project="Thesis", description=f"TransitDuet resource evaluation {shard:02d}/36",
                          signature=f"thesis-transit-resources-v1/shard{shard:02d}",
                          allowed_nodes=NODES, preferred_node=NODES[shard % 6],
                          cmd=f"{PYTHON} -u {REMOTE}/run.py --shard {shard} --workers 16",
                          cwd=str(HERE), cpu=16, ram_mb=24000, vram=0,
                          env=ENV, skip_launch_staging=True,
                          allow_cpu_training=True,
                          cpu_training_justification="Frozen bus-policy evaluation; independent CPU simulations, not training.",
                          result_dir=str(output), local_result_dir=str(output)))
    (HERE / "jobs.json").write_text(json.dumps(specs, indent=2) + "\n")
    print(json.dumps({"shards": 36, "workers_per_shard": 16, "episodes": 1500}), flush=True)
    if not (args.stage or args.smoke):
        return
    sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
    import scheduler
    env_command = "env " + " ".join(shlex.quote(k + "=" + v) for k, v in ENV.items())
    if args.stage:
        methods = ["H_timetable_v5_stable_300", "H_fixed_timetable_300_final", "H_hiro_final"]
        seeds = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
        check = ("from pathlib import Path; "
                 "root=Path('/home/zhengliang01/scheduleurm_work/TransitDuet/transit_duet/logs'); "
                 f"paths=[root/f'{{m}}_seed{{s}}/checkpoints/{{layer}}_ep299.pt' for m in {methods!r} "
                 f"for s in {seeds!r} for layer in ['lower','upper']]; "
                 "missing=[str(p) for p in paths if not p.is_file()]; assert not missing,missing; print('60 checkpoints available')")
        for node in NODES:
            scheduler.run_on(node, f"mkdir -p {shlex.quote(REMOTE)}", timeout=45, check=True)
            subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node(node),
                            "--exclude=runs/", "--exclude=__pycache__/", str(HERE) + "/",
                            f"{scheduler._ssh_target_for_node(node)}:{REMOTE}/"], check=True)
            code, stdout, stderr = scheduler.run_on(node, f"{PYTHON} -c {shlex.quote(check)}", timeout=60, check=False)
            print(node, code, stdout.strip(), flush=True)
            if code:
                raise RuntimeError(stderr)
    if args.smoke:
        command = f"{env_command} {PYTHON} -u {REMOTE}/run.py --smoke"
        code, stdout, stderr = scheduler.run_on(NODES[0], command, timeout=600, check=False)
        print(stdout, flush=True)
        print(stderr[-2500:], flush=True)
        if code:
            raise RuntimeError(f"Observer compatibility failed: exit {code}")
        (HERE / "smoke.log").write_text(stdout + stderr)


if __name__ == "__main__":
    main()
