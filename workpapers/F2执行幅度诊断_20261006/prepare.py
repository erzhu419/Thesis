"""Prepare the original-scale replay and all 32 frozen-policy comparisons."""
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
HPC = "/home/zhengliang01/scheduleurm_work/Thesis"
REMOTE = HPC + "/workpapers/" + HERE.name
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONPATH": "/home/zhengliang01/scheduleurm_work/runtimes/h2oplus-eval-v7-sumo125-overlay-v1/site-packages"}


def job(case, scale):
    label = f"scale{scale:g}-case{case:02d}"
    args = [PYTHON, "-u", REMOTE + "/run.py", "--eval-root", HPC + "/experiments/sumo_eval",
            "--out-root", REMOTE + "/runs", "--case", str(case), "--scale", str(scale)]
    spec = dict(project="Thesis", description="F-2 frozen policy execution-scale diagnosis: " + label,
                signature="thesis-f2-execution-scale-20261006/" + label,
                cwd=str(HERE), cmd=shlex.join(args), cpu=1, ram_mb=2000, vram=0,
                skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True, env=ENV,
                resource_family="thesis-f2-execution-scale-20261006")
    if scale == 0.25:
        spec["require_node"] = "node001"
    else:
        spec["allowed_nodes"] = ["node001", "node004", "node005", "node006"]
    return spec


if __name__ == "__main__":
    for name, jobs in (("replay_jobs.json", [job(0, 0.25)]),
                       ("scale1_jobs.json", [job(i, 1.0) for i in range(32)])):
        (HERE / name).write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
    print("Prepared one replay and 32 evaluation cases; no jobs submitted.")
