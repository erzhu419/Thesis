"""Deploy four independent CPU tasks; transfer only small JSON inputs/results."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
sys.path.insert(0, str(HERE.parent / "mcwm_usage_v3"))
from usage_common import module, write
infra = module("context_filter_infra", HERE.parent / "enhancement_v1/deploy.py")


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def stage():
    for seed in P["seeds"]:
        for scene in P["contexts"]:
            for arm in P["fit_arms"]:
                source = HERE.parent / "mcwm_context_diagnosis_v8/inputs" / scene / f"{arm}_seed{seed}.json"
                write(HERE / "inputs" / scene / source.name, json.loads(source.read_text()))
            folder = HERE.parent / "mcwm_context_v7/runs" / scene / f"recognized_context_seed{seed}" / "results"
            blocks = json.loads((folder / "context_blocks.json").read_text())
            trace = json.loads((folder / "context_trace.json").read_text())
            selection = trace["selection"]
            keep = {key:selection[key] for key in ("provider_seed", "consumer_seed", "coefficients", "recognizer")}
            global_coefficient = {"fixed_zero":0., "fixed_mid":.125, "fixed_original":.25}[selection["best_global_fixed_arm"]]
            write(HERE / "inputs" / scene / f"recognized_context_seed{seed}.json", dict(blocks,
                trace=trace["trace"], selection=keep, global_coefficient=global_coefficient,
                seed=seed, scenario=scene, arm="recognized_context"))
    infra.sync("node001", str(HERE)+"/", remote(), ["--include=*/", "--include=*.py",
        "--include=protocol.json", "--include=inputs/***", "--exclude=*"])
    infra.sync("node001", HERE.parent / "mcwm_context_v7/context.py", remote(HERE.parent / "mcwm_context_v7"))


def tasks():
    return [dict(project="Thesis", description=f"Matched context filter diagnosis {variant} seed{seed}",
        signature=f"{P['name']}/{variant}/seed{seed}", cwd=str(HERE),
        cmd=f"{infra.executable('node001')} -u {remote()}/diagnose.py --seed {seed} --variant {variant}",
        allowed_nodes=infra.CPUS, cpu=1, ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True,
        skip_launch_staging=True, env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        for seed in P["seeds"] for variant in P["variants"]]


def collect():
    (HERE / "results").mkdir(exist_ok=True)
    subprocess.run(["rsync", "-az", "--include=*.json", "--exclude=*", "-e", infra.scheduler._ssh_rsync_shell_for_node("node001"),
        infra.scheduler._ssh_target_for_node("node001")+":"+remote()+"/results/", str(HERE / "results")+"/"], check=True)
    sys.path.insert(0, str(HERE))
    diagnostic = module("matched_filter_collector", HERE / "diagnose.py")
    result = diagnostic.collect()
    print(json.dumps(dict(complete=result["complete"], ready=result["ready"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "submit", "collect"])
    args = parser.parse_args()
    if args.mode == "stage":
        stage()
    elif args.mode == "collect":
        collect()
    else:
        assert not (HERE / "submission.json").exists()
        write(HERE / "jobs.json", tasks())
        result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file",
            str(HERE / "jobs.json"), "--trusted", "--json"], text=True))
        write(HERE / "submission.json", result)
        print(json.dumps(dict(ids=[r["id"] for r in result["submitted"]])))
