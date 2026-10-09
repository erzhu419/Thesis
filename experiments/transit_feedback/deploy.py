"""Stage the isolated study and submit its matched Linux-only jobs."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "TransitDuet/transit_duet"
REMOTE = str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work")
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"
NODES = [f"node{i:03d}" for i in range(1, 7)]
ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MPLBACKEND="Agg", SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")


def stage():
    runtime = HERE / "runtime"
    runtime.mkdir(exist_ok=True)
    for name in ("env", "upper", "lower", "coupling", "configs_ablation"):
        shutil.copytree(SOURCE / name, runtime / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.png", "*.csv"))
    for pattern in ("*.py", "*.yaml"):
        for path in SOURCE.glob(pattern):
            shutil.copy2(path, runtime / path.name)
    (runtime / "scripts").mkdir(exist_ok=True)
    shutil.copy2(SOURCE / "scripts/heldout_final_eval.py", runtime / "scripts/heldout_final_eval.py")
    shutil.copy2(HERE.parent / "transit_resources/measurement.py", HERE / "measurement.py")
    sys.path.insert(0, str(Path(SCHEDULER).parent))
    import scheduler
    for node in NODES + ["node007"]:
        scheduler.run_on(node, f"mkdir -p {shlex.quote(REMOTE)}", timeout=45, check=True)
        subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node(node),
                        "--exclude=runs/", "--exclude=benchmarks/", "--exclude=__pycache__/",
                        str(HERE) + "/", f"{scheduler._ssh_target_for_node(node)}:{REMOTE}/"], check=True)
        print(f"staged {node}", flush=True)


def spec(label, command, node, benchmark=False):
    result = HERE / "benchmarks" if benchmark else HERE / "runs" / label / "results"
    gpu = "--device cuda" in command
    task = dict(project="Thesis", description=f"TransitDuet feedback {label}",
                signature=f"thesis-feedback-v1/{label}", cmd=f"{PYTHON} -u {REMOTE}/run.py {command}",
                cwd=str(HERE), cpu=1 if benchmark else 33, ram_mb=2500 if benchmark else 24000,
                vram=1600 if gpu else 0, env=ENV, skip_launch_staging=True,
                require_node=node, result_dir=str(result), local_result_dir=str(result),
                allow_cpu_training=not gpu,
                cpu_training_justification="Matched bus training uses one thread; independent seeds and post-training test episodes are parallelized.",
                allow_no_resume=True)
    if benchmark:
        task["allow_no_ckpt"] = True
    else:
        task["ckpt_dir"] = str(HERE / "runs" / label / "checkpoints")
    return task


def submit(jobs, name, dispatch=True):
    destination = HERE / f"{name}_jobs.json"
    destination.write_text(json.dumps(jobs, indent=2) + "\n")
    result = json.loads(subprocess.check_output(["python3", SCHEDULER, "submit-jsonl",
        "--file", str(destination), "--trusted", "--json"], text=True))
    (HERE / f"{name}_submission.json").write_text(json.dumps(result, indent=2) + "\n")
    identifiers = [task["id"] for task in result["submitted"]]
    print(json.dumps({"submitted": identifiers}), flush=True)
    if dispatch:
        subprocess.run(["python3", SCHEDULER, "dispatch"] +
                       [part for task in identifiers for part in ("--task-id", task)], check=True)
    return result["submitted"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--benchmarks", action="store_true")
    parser.add_argument("--eval-smoke", action="store_true")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    if args.stage:
        stage()
    if args.benchmarks:
        cases = [("cpu", "full", "node001"), ("cpu", "masked", "node002"),
                 ("cuda", "full", "node007")]
        submit([spec(f"benchmark/{device}_{arm}", f"--benchmark --device {device} --arm {arm}",
                     node, benchmark=True) for device, arm, node in cases], "benchmark")
    if args.production:
        seeds = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
        submit([spec(f"{arm}_seed{seed}", f"--arm {arm} --seed {seed} --device cpu --workers 32",
                     NODES[i % len(NODES)]) for i, seed in enumerate(seeds)
                for arm in ("full", "masked")], "production")
    if args.eval_smoke:
        job = spec("benchmark/evaluation_smoke", "--eval-smoke --arm masked", "node003", benchmark=True)
        job["cpu"] = 2
        submit([job], "evaluation_smoke")
