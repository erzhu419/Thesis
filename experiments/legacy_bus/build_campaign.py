"""Batch legacy comparisons and categorical ablations without outcome gates."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
python = "/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python"
common = dict(project="Thesis", cwd=str(HERE),
              cpu=1, ram_mb=2500, stage_excludes=["runs", "__pycache__"],
              env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MPLBACKEND="Agg", SDL_VIDEODRIVER="dummy"))
specs = []
cpu_nodes = [f"node{i:03d}" for i in range(1, 7)]
hpc_python = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
for method in ["resac", "bapr", "no_hold", "headway"]:
    for scene in ["normal", "congestion_severe", "demand_surge", "slow_switch", "fast_switch"]:
        tag = f"{method}_{scene}"
        out = HERE / "runs/paired_v1" / tag
        node = cpu_nodes[len(specs) % len(cpu_nodes)]
        executable = hpc_python if node.startswith("node") else "python"
        specs.append(dict(common, description=f"Thesis legacy bus frozen-policy evaluation {tag}",
                          signature=f"thesis-legacy-bus-v1/{tag}", vram=0, allow_cpu_training=True,
                          require_node=node,
                          cmd=f"{executable} -u evaluate_pair.py --method {method} --scene {scene} --out {out}",
                          result_dir=str(out), local_result_dir=str(out)))
for representation in ["learned", "masked", "onehot"]:
    for seed in [16001, 16002]:
        tag = f"{representation}_seed{seed}"
        out = HERE / "runs/representation_v1" / tag
        node = ["jtl110gpu", "jtl311linux", "node007"][(len(specs)-20) % 3]
        executable = hpc_python if node == "node007" else python
        specs.append(dict(common, description=f"Thesis categorical representation training {tag}",
                          signature=f"thesis-category-v1/{tag}", vram=2400,
                          allow_gpu_over_one_third=True,
                          require_node=node,
                          cmd=f"{executable} -u train_representation.py --representation {representation} "
                              f"--seed {seed} --out {out} --episodes 100 --resume",
                          ckpt_dir=str(out / "checkpoints"), ckpt_glob="latest.json", resume_flag="--resume",
                          resume_managed_by_cmd=True, result_dir=str(out / "results"),
                          local_result_dir=str(out / "results")))
(HERE / "campaign_jobs.json").write_text(json.dumps(specs, indent=2)+"\n")
print(f"{len(specs)} jobs: 20 frozen-policy cells and 6 representation training cells")
