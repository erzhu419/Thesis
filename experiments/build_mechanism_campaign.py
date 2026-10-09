"""Freeze and submit independent reward-training and repeated-shock cells."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           MPLBACKEND="Agg", SDL_VIDEODRIVER="dummy")
nodes = ["node007", "jtl110gpu", "jtl110gpu2", "jtl311linux"]
specs = []
for arm in ["ridge", "exp60", "exp60_hold", "exp1_hold"]:
    for seed in [17001, 17002, 17003]:
        node = nodes[len(specs) % len(nodes)]
        python = {
            "node007": "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python",
            "jtl110gpu": "/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python",
            "jtl110gpu2": "/home/erzhu419/miniconda3/envs/csbapr/bin/python",
            "jtl311linux": "/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python",
        }[node]
        root = HERE / "legacy_bus"
        tag = f"{arm}_seed{seed}"
        out = root / "runs/reward_v1" / tag
        specs.append(dict(project="Thesis", cwd=str(root),
            description=f"Thesis reward ablation {tag} 100 episodes",
            signature=f"thesis-reward-v1/{tag}", require_node=node,
            cmd=f"{python} -u train_reward.py --arm {arm} --seed {seed} --out {out} --episodes 100 --resume",
            cpu=1, ram_mb=2500, vram=2400,
            env=ENV, stage_excludes=["runs", "__pycache__"],
            ckpt_dir=str(out / "checkpoints"), ckpt_glob="latest.json",
            resume_flag="--resume", resume_managed_by_cmd=True,
            result_dir=str(out / "results"), local_result_dir=str(out / "results")))
remote = "/home/erzhu419/csbapr_exp/thesis_repeated_v1"
for method in ["bapr-pristine", "csbapr-no-nau", "csbapr"]:
    for seed in [0, 1, 2]:
        tag = f"{method}_seed{seed}"
        specs.append(dict(project="Thesis", cwd=remote,
            description=f"Thesis repeated demand shock evaluation {tag}",
            signature=f"thesis-repeated-v1/{tag}", require_node="jtl110gpu2",
            cmd=f"/home/erzhu419/miniconda3/envs/csbapr/bin/python -u evaluate_repeated.py --method {method} --seed {seed} --out {remote}/runs/{tag}",
            cpu=1, ram_mb=2500, vram=0, env=ENV, skip_launch_staging=True,
            result_dir=f"{remote}/runs/{tag}",
            local_result_dir=str(HERE / "csbapr_bus/runs/repeated_v1" / tag)))
(HERE / "mechanism_jobs.json").write_text(json.dumps(specs, indent=2) + "\n")
print(f"{len(specs)} jobs: 12 reward-training cells and 9 frozen-policy evaluation cells")
