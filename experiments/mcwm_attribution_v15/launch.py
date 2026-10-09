"""Code-only staging and small, server-analyzed dependency handoffs."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("attribution_infra", HERE.parent / "enhancement_v1/deploy.py")
infra = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(HERE.parent / "enhancement_v1"))
spec.loader.exec_module(infra)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def remote(path):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def gate(root, label):
    return root / "gates" / (label.replace("/", "_")+".json")


def task(root, protocol, label, command, waits=(), *, training=False, ram=2500):
    family = protocol["name"]+("/train" if training else "/eval")
    row = dict(project="Thesis", description=protocol["name"]+" "+label,
        signature=protocol["name"]+"/"+label, cwd=str(root),
        cmd=f"{infra.PYTHON} -u {root}/node_run.py {command}",
        allowed_nodes=protocol["training_nodes"], cpu=1, ram_mb=ram, vram=0,
        resource_family=family, ram_resource_family=family, vram_resource_family=family,
        env=dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)
    if waits:
        row["wait_for_files"] = [str(gate(root,x)) for x in waits]
    if training:
        row.update(allow_cpu_training=True, cpu_training_justification="Independent single-thread paired learners; shared server data, no GPU requirement.")
    return row


def stage(root, runtime):
    infra.sync("node007", str(root)+"/", remote(root),
               ["--delete", "--include=*.py", "--include=protocol.json", "--exclude=*"])
    write(root / "node_runtime.json", runtime)
    infra.sync("node007", root / "node_runtime.json", remote(root))


def submit(root, protocol, rows, name):
    output = root / (name+"_submission.json")
    if output.exists():
        raise ValueError("Already submitted: "+str(output))
    path = root / (name+"_jobs.json")
    write(path, rows)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(path), "--trusted", "--json"], text=True))
    write(output, result)
    print(json.dumps(dict(batch=name, experiment=protocol["name"], submitted=len(result["submitted"]),
        first=result["submitted"][0]["id"], last=result["submitted"][-1]["id"])), flush=True)


def collect(root):
    command = infra.PYTHON+" -u "+shlex.quote(remote(root / "collect_results.py"))
    infra.scheduler.run_on("node007", command, timeout=120, check=True)
    subprocess.run(["rsync", "-az", "--timeout=30", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        infra.scheduler._ssh_target_for_node("node007")+":"+remote(root / "analysis_summary.json"), str(root)+"/"], check=True, timeout=60)
    report = json.loads((root / "analysis_summary.json").read_text())
    for label in report["ready"]:
        if not gate(root,label).exists():
            write(gate(root,label), dict(ready=label, analysis_location="server"))
    return report


def watch(root):
    deadline = time.monotonic()+72*3600
    while time.monotonic()<deadline:
        try:
            report = collect(root)
            print(json.dumps({k:report[k] for k in ("experiment", "training_ready", "evaluation_ready", "complete")}), flush=True)
            if report["complete"]:
                return
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-600:], flush=True)
        time.sleep(40)
    raise TimeoutError("Inspect incomplete resident jobs; no checkpoint download is needed")


def watcher(root, protocol):
    return dict(project="Thesis", description=protocol["name"]+" compact handoff",
        signature=protocol["name"]+"/watch", cwd=str(root), cmd=f"{sys.executable} -u {root}/launch.py watch",
        require_node="local", cpu=1, ram_mb=256, vram=0, allow_no_ckpt=True,
        allow_no_resume=True, skip_launch_staging=True)


def jobs(smoke=False):
    seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
    scenes = P["scenes"][-1:] if smoke else P["scenes"]
    rows = []
    for seed in seeds:
        for scene in scenes:
            for arm in P["arms"]:
                label = f"{'smoke' if smoke else 'runs'}/{scene}/{arm}_seed{seed}"
                fields = f"--arm {arm} --scenario {scene} --seed {seed} --out {label}"+(" --smoke" if smoke else "")
                rows.append(task(HERE,P,label,fields,training=True,ram=9500 if not smoke else 4000))
                if not smoke:
                    for case in P["cases"]:
                        rows.append(task(HERE,P,label+f"/eval{case}",fields+f" --evaluate --case {case}",[label]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "smoke", "submit", "watch", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        runtime = json.loads((HERE.parent / "mcwm_seed_confirmation_v13/deployments/node007/node_runtime.json").read_text())
        stage(HERE,runtime)
    elif mode == "watch":
        watch(HERE)
    elif mode == "collect":
        print(json.dumps(collect(HERE)))
    else:
        smoke = mode == "smoke"
        rows = jobs(smoke)
        if not smoke:
            rows.append(watcher(HERE,P))
        submit(HERE,P,rows,"smoke" if smoke else "campaign")
