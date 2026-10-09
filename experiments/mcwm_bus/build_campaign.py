"""Emit all independent cells for scheduler bulk submission."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = json.loads((HERE / "budget_protocol_v2.json").read_text())
cells = [(count, method, "calibrated") for count in [2, 6] for method in ["offline", "h2o", "mcwm"]]
cells += [(0, "online", "calibrated")]
cells += [(6, method, scenario["name"]) for scenario in p["mismatch_scenarios"][1:]
          for method in ["h2o", "mcwm"]]
runtime = HERE / "runtime"
source = Path("/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/v7_formal_v2_training_v7r1/workspace")
env = dict(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
           THESIS_H2O_ROOT=str(runtime / "H2Oplus"),
           THESIS_MCWM_ROOT=str(runtime / "MC-WM"),
           THESIS_DATASET=str(source / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"),
           THESIS_SOURCE_ENV=str(source / "bus_h2o/calibrated_env"),
           THESIS_SUMO_ROOT="/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang",
           MPLBACKEND="Agg")
specs = []
for count, method, scenario in cells:
    name = f"blocks{count}_{scenario}_{method}"
    out = HERE / "runs/campaign_v3" / name
    cmd = ("/home/erzhu419/miniconda3/envs/csbapr/bin/python -u run_campaign_cell.py "
           f"--out {out} --initializer {HERE}/prefixes/blocks{count or 2}/offline_initializer.pt "
           f"--blocks {count} --method {method} --scenario {scenario} --threads 2 --resume")
    specs.append(dict(description=f"Thesis bus training and SUMO evaluation: {name}",
                      project="Thesis", signature=f"thesis-bus-campaign-v3/{name}",
                      cmd=cmd, cwd=str(HERE), env=env,
                      require_node="jtl110gpu2",
                      vram=5200 if count == 6 else 2400, ram_mb=10000 if count == 6 else 5000,
                      cpu=2, ckpt_dir=str(out / "checkpoints"), ckpt_glob="latest.pt",
                      resume_flag="--resume", resume_managed_by_cmd=True,
                      result_dir=str(out / "results"), local_result_dir=str(out / "results"),
                      skip_launch_staging=True))
(HERE / "campaign_v3_jobs.json").write_text(json.dumps(specs, indent=2) + "\n")
print(f"{len(specs)} independent jobs; results-only synchronization")
