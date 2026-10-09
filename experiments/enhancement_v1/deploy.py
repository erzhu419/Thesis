"""Stage isolated runners, submit independent jobs, and transfer compact policies."""

from study_common import HERE, EXPERIMENTS, P, write
import argparse
import json
import shlex
import subprocess
import sys
import time

SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"
sys.path.insert(0, str(__import__("pathlib").Path(SCHEDULER).parent))
import scheduler

GPUS = ["node007", "jtl110gpu", "jtl110gpu2", "jtl311linux"]
CPUS = [f"node{i:03d}" for i in range(1, 7)]
HPC = "/home/zhengliang01/scheduleurm_work"
PYTHON = HPC + "/conda_envs/csbapr-gpu-py310/bin/python"


def resource_spec(kind, arm, node, *, smoke=False, evaluation=False, cpu_training=False):
    mode = "cpu" if evaluation or cpu_training else "gpu"
    phase = "eval" if evaluation else "smoke" if smoke else "train"
    family = f"thesis-enhancement-v1/{kind}/{arm}/{mode}/{phase}"
    # Full MC-WM runs stay below 0.5 GiB including the CUDA context on these nodes.
    vram = 768 if kind in ("mc", "legacy") else 512
    if node == "jtl110gpu":
        vram = 1600 if kind == "legacy" else 2200
    return dict(resource_family=family, vram_resource_family=f"{family}/{node}",
                vram=0 if mode == "cpu" else vram,
                ram_mb=9000 if kind == "mc" and not evaluation else 2500)


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", HPC) if node.startswith("node") else str(HERE)


def executable(node):
    return PYTHON if node.startswith("node") else json.loads(
        (EXPERIMENTS / f"mcwm_bus/deployment_{node}.json").read_text())["python"]


def environment(node, evaluation=False):
    key = "node007" if node.startswith("node") else node
    env = json.loads((EXPERIMENTS / f"mcwm_bus/deployment_{key}.json").read_text())["env"]
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               SDL_VIDEODRIVER="dummy", MPLBACKEND="Agg")
    if evaluation:
        base = str(__import__("pathlib").Path(root(node)).parent / "sumo_eval")
        env.update(THESIS_DATASET=base+"/contract.h5", THESIS_SUMO_ROOT=base+"/assets/SUMO_ruiguang",
                   THESIS_H2O_ROOT=base+"/runtime/H2Oplus",
                   SUMO_HOME=base+"/assets/SUMO_ruiguang",
                   PYTHONPATH=HPC+"/runtimes/h2oplus-eval-v7-sumo125-overlay-v1/site-packages")
    return env


def sync(node, source, destination, filters=()):
    scheduler.run_on(node, "mkdir -p " + shlex.quote(destination), timeout=45, check=True)
    subprocess.run(["rsync", "-aL", "--exclude=__pycache__", "--exclude=.git", *filters,
                    "-e", scheduler._ssh_rsync_shell_for_node(node), str(source),
                    scheduler._ssh_target_for_node(node)+":"+destination+"/"], check=True)


def stage():
    for node in GPUS:
        base = root(node)
        parent = str(__import__("pathlib").Path(base).parent)
        sync(node, str(HERE)+"/", base, ["--exclude=runs", "--exclude=smoke", "--exclude=runtime", "--exclude=*jobs.json", "--exclude=*submission.json"])
        scheduler.run_on(node, f"mkdir -p {shlex.quote(base+'/runs')} {shlex.quote(base+'/smoke')}", timeout=45, check=True)
        sync(node, str(EXPERIMENTS / "legacy_bus")+"/", parent+"/legacy_bus",
             ["--exclude=runs", "--exclude=weights", "--exclude=*jobs.json"])
        sync(node, EXPERIMENTS / "csbapr_bus/evaluate_recovery.py", parent+"/csbapr_bus")
        sync(node, EXPERIMENTS / "mcwm_bus/prefix_config.json", parent+"/mcwm_bus")
        # Existing MC-WM/H2O and data packs are already resident. Never restage datasets.
        source = EXPERIMENTS.parents[1]
        sync(node, str(source / "CS-BAPR/csbapr")+"/", base+"/runtime/cs/CS-BAPR/csbapr",
             ["--include=*/", "--include=*.py", "--exclude=*"])
        sync(node, source / "CS-BAPR/scripts/test_multiline_convergence.py", base+"/runtime/cs/CS-BAPR/scripts")
        for folder in ("sim_core", "calibrated_env"):
            sync(node, str(source / "offline-sumo/env" / folder)+"/", base+"/runtime/cs/offline-sumo/env/"+folder,
                 ["--exclude=_line_envs"])
        print("staged " + node, flush=True)


def spec(kind, arm, seed, node, smoke=False, case=None):
    cpu_training = kind == "cs" and node in CPUS and case is None
    label = f"{kind}/{arm}_seed{seed}"
    relative = ("smoke/" if smoke else "runs/")+label
    remote = root(node)+"/"+relative
    command = (f"{executable(node)} -u {root(node)}/{kind}.py --arm {arm} --seed {seed} "
               f"--out {remote}" + (" --smoke" if smoke else ""))
    if case is not None:
        command += f" --evaluate --case {case}"
    elif kind == "cs":
        command += " --device " + ("cpu" if cpu_training else "cuda")
    task = dict(project="Thesis", description=f"Enhancement {relative} " + (f"eval{case}" if case is not None else "train"),
                signature=f"thesis-enhancement-v1/{relative}/"+(f"eval{case}" if case is not None else "train"),
                cwd=str(HERE), cmd=command, require_node=node, skip_launch_staging=True,
                stage_excludes=["runs", "smoke", "runtime"],
                cpu=1, allow_gpu_over_one_third=True,
                allow_no_ckpt=True, allow_no_resume=True, env=environment(node, case is not None))
    task.update(resource_spec(kind, arm, node, smoke=smoke,
                              evaluation=case is not None, cpu_training=cpu_training))
    if cpu_training:
        task.update(allow_cpu_training=True, cpu_training_justification=(
            "Measured single-thread CS-BAPR bus training; complete four-arm seed groups use the same CPU node."))
    if case is not None:
        task["wait_for_files"] = [str(policy_gate(kind, arm, seed, smoke))]
    elif not smoke:
        task["wait_for_files"] = [str(test_gate(kind, arm))]
    if kind == "legacy" and case is None:
        task.update(cmd=command+" --resume", ckpt_dir=str(HERE / relative / "checkpoints"),
                    resume_flag="", resume_managed_by_cmd=True, skip_resume_scan=True)
    return task


def policy_gate(kind, arm, seed, smoke):
    return HERE / "gates" / f"{'smoke' if smoke else 'runs'}_{kind}_{arm}_{seed}_policy.json"


def test_gate(kind, arm):
    return HERE / "gates" / f"{kind}_{arm}_smoke_test.json"


def make_jobs():
    jobs, locations = [], []
    placements_path = HERE / "cs_cpu_placements.json"
    cs_cpu_nodes = json.loads(placements_path.read_text()) if placements_path.exists() else {}
    gpu_placements_path = HERE / "gpu_placements.json"
    gpu_nodes = json.loads(gpu_placements_path.read_text()) if gpu_placements_path.exists() else {}
    for k, kind in enumerate(("legacy", "cs", "mc")):
        for a, arm in enumerate(P[kind+"_arms"]):
            node = GPUS[(a+k) % len(GPUS)]
            jobs.append(spec(kind, arm, 22000, node, smoke=True))
            jobs.append(spec(kind, arm, 22000, CPUS[(a+k) % len(CPUS)], smoke=True, case=0))
            locations.append(dict(kind=kind, arm=arm, seed=22000, node=node, smoke=True))
        for i, seed in enumerate(P["training_seeds"]):
            node = GPUS[(i+k) % len(GPUS)]
            node = gpu_nodes.get(f"{kind}/{seed}", node)
            if kind == "cs":
                node = cs_cpu_nodes.get(str(seed), node)
            for arm in P[kind+"_arms"]:
                jobs.append(spec(kind, arm, seed, node))
                locations.append(dict(kind=kind, arm=arm, seed=seed, node=node, smoke=False))
                for case in range(10 if kind == "mc" else 9):
                    cpu = CPUS[(len(jobs)+case) % len(CPUS)]
                    jobs.append(spec(kind, arm, seed, cpu, case=case))
    write(HERE / "locations.json", locations)
    return jobs


def submit(jobs, name):
    path = HERE / f"{name}_jobs.json"
    write(path, jobs)
    result = json.loads(subprocess.check_output(["python3", SCHEDULER, "submit-jsonl", "--file",
                                                str(path), "--trusted", "--json"], text=True))
    write(HERE / f"{name}_submission.json", result)
    print(json.dumps(dict(name=name, submitted=len(result["submitted"]))), flush=True)
    return [t["id"] for t in result["submitted"]]


def ssh(node, command):
    return shlex.split(scheduler._ssh_rsync_shell_for_node(node)) + [scheduler._ssh_target_for_node(node), command]


def transfer(item):
    label = ("smoke/" if item["smoke"] else "runs/")+item["kind"]+f"/{item['arm']}_seed{item['seed']}"
    source, destination = root(item["node"])+"/"+label, root("node001")+"/"+label
    files = "bundle/controller.pt" if item["kind"] == "legacy" else "policy.pt"
    scheduler.run_on(item["node"], "test -s "+shlex.quote(source+"/"+files), timeout=45, check=True)
    if source != destination:
        scheduler.run_on("node001", "mkdir -p "+shlex.quote(destination), timeout=45, check=True)
        # A pipe relays the compact policy only; no checkpoint touches local disk.
        sender = subprocess.Popen(ssh(item["node"], f"tar -C {shlex.quote(source)} -czf - {files}"), stdout=subprocess.PIPE)
        try:
            receiver = subprocess.run(ssh("node001", f"tar -C {shlex.quote(destination)} -xzf -"),
                                      stdin=sender.stdout, timeout=180)
        finally:
            sender.stdout.close()
        if sender.wait(timeout=30) or receiver.returncode:
            raise RuntimeError("Policy transfer failed: "+label)
    write(HERE / label / "eval_ready.json", dict(evaluation_storage="node001-node006", **item))
    write(policy_gate(item["kind"], item["arm"], item["seed"], item["smoke"]), item)


def protect_outputs():
    with scheduler.state_lock(purpose="protect thesis experiment output directories"):
        state = scheduler.load_state()
        count = 0
        for task in state["tasks"]:
            if task.get("signature", "").startswith("thesis-enhancement-v1/"):
                task["stage_excludes"] = ["runs", "smoke", "runtime"]
                parts = task["signature"].split("/")
                if len(parts) >= 5 and parts[1] in ("smoke", "runs"):
                    kind = parts[2]
                    arm, seed = parts[3].rsplit("_seed", 1)
                    if parts[4].startswith("eval"):
                        task["wait_for_files"] = [str(policy_gate(kind, arm, int(seed), parts[1] == "smoke"))]
                        task.setdefault("extra_env", {}).update(environment(task["require_node"], True))
                    elif parts[1] == "runs":
                        task["wait_for_files"] = [str(test_gate(kind, arm))]
                count += 1
        scheduler.save_state(state)
    print(f"Protected outputs for {count} campaign task records", flush=True)


def watch():
    deadline = time.monotonic()+48*3600
    while time.monotonic() < deadline:
        locations = json.loads((HERE / "locations.json").read_text())
        for node in GPUS:
            for folder in ("smoke", "runs"):
                destination = HERE / folder
                destination.mkdir(exist_ok=True)
                subprocess.run(["rsync", "-a", "--include=*/", "--include=*.json", "--exclude=*", "-e",
                    scheduler._ssh_rsync_shell_for_node(node),
                    scheduler._ssh_target_for_node(node)+":"+root(node)+"/"+folder+"/", str(destination)+"/"],
                    stdout=subprocess.DEVNULL, timeout=60)
        for item in locations:
            label = ("smoke/" if item["smoke"] else "runs/")+item["kind"]+f"/{item['arm']}_seed{item['seed']}"
            if (HERE / label / "results/ready.json").exists() and not policy_gate(
                    item["kind"], item["arm"], item["seed"], item["smoke"]).exists():
                try:
                    transfer(item)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    print(f"Handoff pending for {label}: {error}", flush=True)
            if item["smoke"] and (HERE / label / "results/eval_00.json").exists():
                gate = test_gate(item["kind"], item["arm"])
                if not gate.exists():
                    write(gate, dict(kind=item["kind"], arm=item["arm"], interface_test_complete=True))
        train_done = len(list((HERE / "runs").glob("*/*/results/ready.json")))
        eval_done = len(list((HERE / "runs").glob("*/*/results/eval_*.json")))
        print(json.dumps(dict(training_ready=train_done, evaluation_complete=eval_done)), flush=True)
        if train_done == 80 and eval_done == 750:
            return
        time.sleep(45)
    raise TimeoutError("Enhancement study remains incomplete after 48 hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--protect-outputs", action="store_true")
    args = parser.parse_args()
    if args.protect_outputs:
        protect_outputs()
    if args.stage:
        stage()
    if args.submit:
        ids = submit(make_jobs(), "campaign")
        watcher = dict(project="Thesis", description="Enhancement policy handoff and small JSON collection",
                       signature="thesis-enhancement-v1/watcher", cwd=str(HERE),
                       cmd=f"python3 -u {HERE}/deploy.py --watch", require_node="local", cpu=1, ram_mb=512,
                       vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)
        ids += submit([watcher], "watcher")
        subprocess.run(["python3", SCHEDULER, "dispatch"]+[v for i in ids for v in ("--task-id", i)],
                       check=True, stdout=subprocess.DEVNULL)
    if args.watch:
        watch()
