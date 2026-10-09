"""Continuous 300-episode paired runs, with separately scheduled endpoint tests."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from deploy import HERE, REMOTE, SCHEDULER, NODES, PYTHON, spec, submit

sys.path.insert(0, str(Path(SCHEDULER).parent))
import scheduler


def make_jobs():
    jobs = []
    seeds = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
    for i, seed in enumerate(seeds):
        for arm in ("full", "masked"):
            label = f"{arm}_seed{seed}"
            root = HERE / "runs300" / label
            node = NODES[i % len(NODES)]
            common = f"--arm {arm} --seed {seed} --training-episodes 300 --test-plan extended"
            train = spec(label, common + " --train-only --device cpu", node)
            train.update(signature=f"thesis-feedback-300-continuous/train/{label}",
                         description=f"TransitDuet continuous 300 from scratch {label}",
                         cpu=1, ram_mb=4096, ckpt_dir=str(root / "checkpoints"))
            train.pop("result_dir")
            train.pop("local_result_dir")
            jobs.append(train)
            for endpoint in (99, 299):
                result = root / f"results_ep{endpoint}_extended"
                task = spec(label, common + f" --evaluate-only --checkpoint-episode {endpoint} --workers 32", node)
                task.update(signature=f"thesis-feedback-300-continuous/eval{endpoint}/{label}",
                            description=f"TransitDuet fixed endpoint {endpoint+1} evaluation {label}",
                            result_dir=str(result), local_result_dir=str(result),
                            ckpt_dir=str(root / "checkpoints"),
                            wait_for_files=[str(HERE / "gates300" / f"{label}_ep{endpoint}.json")],
                            stage_excludes=["runs", "runs300", "runtime"])
                jobs.append(task)
    return jobs


def watch_ready():
    destination = HERE / "runs300"
    destination.mkdir(exist_ok=True)
    deadline = time.monotonic() + 8 * 3600
    while time.monotonic() < deadline:
        # This server directory is shared by node001-node006. Only tiny ready markers are pulled.
        result = subprocess.run([
            "rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node001"),
            "--include=*/", "--include=ready_ep*.json", "--exclude=*",
            f"{scheduler._ssh_target_for_node('node001')}:{REMOTE}/runs300/", str(destination) + "/"])
        if result.returncode:
            print("Ready-marker pull failed; retrying next pass", flush=True)
        else:
            gates = HERE / "gates300"
            gates.mkdir(exist_ok=True)
            code = (
                "import json; from pathlib import Path; root=Path("+repr(REMOTE+"/runs300")+"); "
                "print(json.dumps({p.parent.parent.name+'_'+p.stem.replace('ready_', ''): "
                "all((p.parent/(layer+'_'+p.stem.replace('ready_', '')+'.pt')).is_file() "
                "for layer in ('lower','upper')) for p in root.glob('*/checkpoints/ready_ep*.json')}))"
            )
            _, availability, _ = scheduler.run_on(
                "node001", PYTHON+" -c "+shlex.quote(code), timeout=45, check=True)
            available = json.loads(availability)
            (HERE / "continuous300_checkpoint_availability.json").write_text(
                json.dumps(available, indent=2)+"\n")
            for marker in destination.glob("*/checkpoints/ready_ep*.json"):
                label = marker.parent.parent.name
                endpoint = marker.stem.removeprefix("ready_ep")
                gate = gates / f"{label}_ep{endpoint}.json"
                if not available.get(f"{label}_ep{endpoint}", False):
                    if gate.exists():
                        gate.replace(gate.with_suffix(".missing"))
                    continue
                if not gate.exists():
                    temporary = gate.with_suffix(".tmp")
                    temporary.write_text(marker.read_text())
                    temporary.replace(gate)
            count100 = sum(ok for key, ok in available.items() if key.endswith("_ep99"))
            count300 = sum(ok for key, ok in available.items() if key.endswith("_ep299"))
            print(f"available100={count100}/20 available300={count300}/20", flush=True)
            if count300 == 20:
                print("All continuous training endpoints ready", flush=True)
                return
        time.sleep(30)
    raise TimeoutError("Continuous training not all complete after eight hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch:
        watch_ready()
    else:
        for node in NODES:
            subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node(node),
                            str(HERE / "run.py"), f"{scheduler._ssh_target_for_node(node)}:{REMOTE}/"], check=True)
        submitted = submit(make_jobs(), "continuous300", dispatch=False)
        watcher = dict(project="Thesis", description="Pull continuous-study endpoint ready markers",
                       signature="thesis-feedback-300-continuous/ready-markers",
                       cmd=f"python3 -u {HERE}/deploy300.py --watch", cwd=str(HERE),
                       cpu=1, ram_mb=512, vram=0, require_node="local", skip_launch_staging=True,
                       allow_no_ckpt=True, allow_no_resume=True)
        submitted += submit([watcher], "continuous300_watcher", dispatch=False)
        subprocess.run(["python3", SCHEDULER, "dispatch"] +
                       [x for task in submitted for x in ("--task-id", task["id"])], check=True)
