"""Queue CPU evaluation when the corresponding small policy export arrives."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
placements = json.loads((HERE.parent / "mcwm_bus/campaign_v3_placement.json").read_text())
original = json.loads((HERE.parent / "mcwm_bus/campaign_v3_jobs.json").read_text())
settings = json.loads((HERE.parent / "mcwm_bus/budget_protocol_v2.json").read_text())
already_split = {cell["signature"] for cell in placements}
placements += [cell for cell in original if cell["signature"] not in already_split]
nodes = [f"node{i:03d}" for i in range(1, 7)]
scenario_nodes = dict(speed25="node003", od25="node004", signals25="node005",
                      dwell25="node006", joint25="node006", joint50="node001",
                      mechanistic="node003")
specs = []
for index, cell in enumerate(placements):
    name = cell["signature"].split("/")[-1]
    inputs = Path(cell["local_result_dir"])
    out = HERE / "runs/sumo125_v3" / name
    scenario = name.split("_")[1]
    policies = len(settings["source_event_budgets"]) if scenario == "calibrated" else 1
    policies += int(name.endswith("_offline"))
    cases = policies * len(settings["evaluation_blocks"]) * len(settings["evaluation_demand_scales"])
    workers = min(cases, 96)
    node = scenario_nodes.get(scenario, "node001" if name.startswith("blocks2_") else "node002")
    python = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
    specs.append(dict(project="Thesis", description=f"Thesis SUMO CPU evaluation: {name}",
                      signature=f"thesis-sumo-cpu-v3/{name}", allowed_nodes=nodes,
                      preferred_node=node,
                      cmd=f"{python} -u run.py --input {inputs} --out {out} --workers {workers}",
                      cwd=str(HERE), vram=0, ram_mb=1500 + 1000 * workers, cpu=workers,
                      allow_cpu_training=True,
                      env=dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                               MPLBACKEND="Agg",
                               PYTHONPATH=("/home/zhengliang01/scheduleurm_work/runtimes/"
                                           "h2oplus-eval-v7-sumo125-overlay-v1/site-packages")),
                      stage_excludes=["runs", "__pycache__"], stage_input_paths=[str(inputs)],
                      wait_for_files=[str(inputs / "training_summary.json")],
                      result_dir=str(out / "results"), local_result_dir=str(out / "results")))
(HERE / "campaign_jobs.json").write_text(json.dumps(specs, indent=2)+"\n")
(HERE / "handoff_jobs.json").write_text(json.dumps(specs[17:], indent=2)+"\n")
print(f"{len(specs)} CPU evaluation jobs across {len(nodes)} nodes")
