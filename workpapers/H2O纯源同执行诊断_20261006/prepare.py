"""Prepare the isolated replay and the fixed 150-case execution comparison."""
import argparse
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
REMOTE = str(HERE)
PYTHON = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
DEPLOYMENT = Path("/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/h2oplus_v20_formal_confirmation_evaluation_v4")
ROOT = DEPLOYMENT / "workspace/H2Oplus"
SUMO = DEPLOYMENT / "workspace/SUMO_ruiguang/online_control"
OLD_LOCAL = Path("/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/h2oplus_v20_formal_confirmation_results/evaluation_v4/single_jtl110gpu")
NODE = "jtl110gpu"


def command(seed, block, demand, mode, dataset_stat=None):
    scenario = f"demand_{str(demand).replace('.', 'p')}_jitter120s_seed{block+100}"
    method = f"sac-online-seed{seed}-events000025000"
    relative = Path(f"v20-formal-confirmation-{block-8100:02d}") / method / scenario / f"sumo_seed{block}.json"
    # Read the reference now so a missing or misidentified input prevents submission.
    reference = json.loads((OLD_LOCAL / relative).read_text())
    assert reference["sumo_seed"] == block
    assert reference["policy"]["training_metadata"]["training_seed"] == seed
    name = ("replay_raw_seed6201_sumo8101_demand0p75.json" if mode == "source-raw"
            else f"sac_guarded_seed{seed}_sumo{block}_demand{str(demand).replace('.', 'p')}.json")
    values = {
        "h2o-root": ROOT, "reference-result": DEPLOYMENT / "remote_results_v4/single_jtl110gpu" / relative,
        "mode": mode,
        "checkpoint": ROOT / f"reproducibility/h2oplus_v8_fewshot_results/training/sac-online_seed{seed}/fewshot_policies/online_events_000025000.pt",
        "scenario-manifest": ROOT / "bus_h2o/formal_confirmation_scenarios_v20/v1" / scenario / "manifest_portable_v8.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "sumo-dir": SUMO, "edge-xml": SUMO / "intersection_delay/a_sorted_busline_edge.xml",
        "sumo-runtime-lock": ROOT / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "output": Path(REMOTE) / ("replay" if mode == "source-raw" else "results") / name,
    }
    if dataset_stat:
        values.update({"verified-dataset-size-bytes": dataset_stat["size_bytes"],
                       "verified-dataset-mtime-ns": dataset_stat["mtime_ns"]})
    argv = [PYTHON, "-u", REMOTE + "/run.py"]
    for key, value in values.items():
        argv += ["--" + key, str(value)]
    return argv + ["--quiet"]


def job(command_args, label, cpu):
    family = "thesis-h2o-source-guard-20261006/" + NODE + "/" + label
    return dict(project="Thesis", description="Frozen H2O/SAC execution diagnosis: " + label,
                signature=family, cwd=str(HERE), cmd=shlex.join(command_args),
                require_node=NODE, skip_launch_staging=True,
                cpu=cpu, ram_mb=2500 * cpu, vram=0,
                resource_family=family, ram_resource_family=family,
                allow_no_ckpt=True, allow_no_resume=True,
                env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
                     "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
                     "PYTHONPATH": str(ROOT),
                     "TMPDIR": "/home/erzhu419/.cache/h2oplus-v7/tmp",
                     "PYTHONDONTWRITEBYTECODE": "1", "NUMEXPR_NUM_THREADS": "1",
                     "MPLBACKEND": "Agg", "SDL_VIDEODRIVER": "dummy"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-stat", type=Path)
    parser.add_argument("--node", choices=("jtl110gpu", "jtl110gpu2"), default="jtl110gpu")
    parser.add_argument("--first-block", type=int, default=8101)
    parser.add_argument("--last-block", type=int, default=8110)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    NODE = args.node
    if NODE == "jtl110gpu2":
        DEPLOYMENT = HERE / "runtime"
        ROOT = DEPLOYMENT / "workspace/H2Oplus"
        SUMO = DEPLOYMENT / "workspace/SUMO_ruiguang/online_control"
    stat = json.loads(args.dataset_stat.read_text()) if args.dataset_stat else None
    cases = [command(seed, block, demand, "source-guarded", stat)
             for block in range(args.first_block, args.last_block + 1) for demand in (0.75, 1, 1.25)
             for seed in range(6201, 6206)]
    prefix = NODE + f"_{args.first_block}_{args.last_block}"
    files = {
        prefix + "_cases.json": cases,
        NODE + "_replay_jobs.json": [job(command(6201, 8101, 0.75, "source-raw"), "raw-replay", 1)],
        prefix + "_batch_jobs.json": [job([PYTHON, "-u", REMOTE + "/batch.py", "--cases", REMOTE + "/" + prefix + "_cases.json",
                                 "--workers", str(args.workers)], f"guarded-{args.first_block}-{args.last_block}", args.workers)],
        prefix + "_individual_jobs.json": [job(argv, Path(argv[argv.index("--output") + 1]).stem, 1) for argv in cases],
    }
    for name, data in files.items():
        (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared one raw replay and {len(cases)} guarded evaluations for {NODE}; no tasks submitted.")
