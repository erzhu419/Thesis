"""Stage small frozen actors once per Linux CPU node and prepare batch jobs."""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
NODES = [f"node{i:03d}" for i in range(1, 7)]
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MPLBACKEND="Agg", PYTHONPATH=("/home/zhengliang01/scheduleurm_work/runtimes/"
           "h2oplus-eval-v7-sumo125-overlay-v1/site-packages"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    p = json.loads((HERE / "protocol.json").read_text())
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)
    budgets = {}
    for method, source in p["policy_exports"].items():
        directory = BASE.parent / "mcwm_bus/runs/campaign_v3" / source["cell"] / "results"
        summary = json.loads((directory / "training_summary.json").read_text())
        budgets[method] = next(row for row in summary["snapshots"] if row["tag"] == source["tag"])
        shutil.copy2(directory / "policies" / f"{source['tag']}.pt", inputs / f"{method}.pt")
    (inputs / "training_budgets.json").write_text(json.dumps(budgets, indent=2) + "\n")
    specs = []
    for shard in range(p["shards"]):
        out = HERE / "runs" / f"shard{shard:02d}"
        specs.append(dict(project="Thesis", description=f"Thesis execution attribution shard {shard:02d}/36",
                          signature=f"thesis-attribution-v1/shard{shard:02d}",
                          allowed_nodes=NODES, preferred_node=NODES[shard % len(NODES)],
                          cmd=f"{PYTHON} -u attribution_v1/run.py --shard {shard}", cwd=str(BASE),
                          vram=0, cpu=p["workers_per_shard"], ram_mb=14000,
                          allow_cpu_training=True, env=ENV, skip_launch_staging=True,
                          result_dir=str(out / "results"), local_result_dir=str(out / "results")))
    (HERE / "jobs.json").write_text(json.dumps(specs, indent=2) + "\n")
    print(json.dumps(dict(jobs=len(specs), cases=p["expected_cases"],
                          input_bytes=sum(f.stat().st_size for f in inputs.iterdir()))), flush=True)
    if not args.stage and not args.smoke:
        return
    sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
    import scheduler
    remote = str(BASE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work")
    env_command = "env " + " ".join(shlex.quote(k + "=" + v) for k, v in ENV.items())
    deployment = []
    if args.stage:
        for node in NODES:
            scheduler.run_on(node, f"mkdir -p {shlex.quote(remote)}", timeout=45, check=True)
            subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node(node),
                            "--exclude=runs/", "--exclude=__pycache__/", str(BASE) + "/",
                            f"{scheduler._ssh_target_for_node(node)}:{remote}/"], check=True)
            command = (f"cd {shlex.quote(remote)}/attribution_v1 && {env_command} "
                       f"{PYTHON} test_actions.py")
            code, stdout, stderr = scheduler.run_on(node, command, timeout=90, check=False)
            print(json.dumps(dict(node=node, test_exit_code=code, output=stdout + stderr)), flush=True)
            if code:
                raise RuntimeError(f"Node preflight failed: {node}")
            deployment.append(dict(node=node, ready_at=datetime.now(timezone.utc).isoformat()))
        (HERE / "deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")
    if args.smoke:
        command = (f"cd {shlex.quote(remote)} && {env_command} "
                   f"{PYTHON} -u attribution_v1/run.py --shard 0 --smoke")
        code, stdout, stderr = scheduler.run_on(NODES[0], command, timeout=180, check=False)
        print(json.dumps(dict(smoke_exit_code=code, output=stdout + stderr)), flush=True)
        if code:
            raise RuntimeError("Short remote SUMO smoke failed")


if __name__ == "__main__":
    main()
