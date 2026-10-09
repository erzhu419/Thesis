"""Independent family gates; reuse resident data and parallel policy/test shards."""

import argparse
import json
import shlex
import subprocess
import sys
import time

from followup_common import HERE, EXPERIMENTS, P, label, policies, needs_training, evaluation_cases, speed_controls, write
sys.path.insert(0, str(EXPERIMENTS / "bus_control_repair_v3"))
sys.path.insert(0, str(EXPERIMENTS / "enhancement_v1"))
from repair_common import module
infra = module("alignment_followup_infra", EXPERIMENTS / "enhancement_v1/deploy.py")


def remote(path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def marker(name):
    return HERE / "gates" / (name.replace("/", "_")+".json")


def task(item, phase, layer="native", case=0):
    if item["family"] == "mc" and layer == "native":
        layer = "guarded"
    name = label(item)+("/train" if phase == "train" else f"/eval/{layer}/{case}")
    fields = " ".join(f"--{key} {value}" for key, value in item.items() if key != "smoke")
    fields += f" --phase {phase} --layer {layer} --case {case}" + (" --smoke" if item.get("smoke") else "")
    threads = 4 if phase == "train" and item["family"] == "mc" else 1
    waits = [marker("contracts/"+item["family"])] if item.get("smoke") else [marker("smoke-pass/"+item["family"])]
    if phase == "evaluate" and not item["arm"].startswith("zero_"):
        waits.append(marker(label(item)))
    memory = 3200 if phase == "train" else 1600
    if phase == "evaluate" and item["family"] != "mc" and not item.get("smoke"):
        scene = list(P["cs_scenes"])[case//3]
        memory = dict(static10=10000, static5=6000).get(scene, 3500)
    return dict(project="Thesis", description="Bus alignment v5 "+name, signature=P["name"]+"/"+name,
        cwd=str(HERE), cmd=f"{infra.PYTHON} -u {HERE}/node_run.py {fields}",
        allowed_nodes=P["nodes"], cpu=threads, ram_mb=memory, vram=0,
        resource_family=P["name"]+"/"+item["family"]+"/"+phase,
        env=dict(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1"),
        wait_for_files=list(map(str, waits)), skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True,
        allow_cpu_training=True, cpu_training_justification="Independent small bus learners; paired seeds, resident compact offline replay, no GPU requirement.")


def jobs():
    rows = []
    for smoke in (True, False):
        for item in policies(smoke):
            if needs_training(item):
                rows.append(task(item, "train"))
            rows.extend(task(item, "evaluate", layer, case) for layer, case in evaluation_cases(item))
    for item in speed_controls():
        rows.extend(task(item, "evaluate", layer, case) for layer, case in evaluation_cases(item))
    rows.append(dict(project="Thesis", description="Bus alignment v5 compact gate handoff",
        signature=P["name"]+"/watch", cwd=str(HERE), cmd=f"{sys.executable} -u {HERE}/launch.py watch",
        require_node="local", cpu=1, ram_mb=256, vram=0, skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True))
    return rows


def stage():
    infra.scheduler.run_on("node007", "mkdir -p "+shlex.quote(remote()), timeout=40, check=True)
    files = sorted(HERE.glob("*.py"))+[HERE / "protocol.json"]
    subprocess.run(["rsync", "-az", "--timeout=35", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        *map(str, files), infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/"], check=True, timeout=90)


def contracts():
    for family in ("cs_recipe", "cs_context", "mc"):
        cmd = f"{infra.PYTHON} -u {remote()}/node_run.py --family {family} --phase contracts"
        rc, out, err = infra.scheduler.run_on("node007", cmd, timeout=90, check=False)
        print(out[-1400:]+err[-2000:], flush=True)
        if rc:
            raise RuntimeError("Contract failed: "+family)
    collect()


def collect():
    rc, out, err = infra.scheduler.run_on("node007", infra.PYTHON+" -u "+shlex.quote(remote()+"/collect_results.py"), timeout=90, check=False)
    if rc:
        raise RuntimeError((out+err)[-2000:])
    subprocess.run(["rsync", "-az", "--timeout=35", "-e", infra.scheduler._ssh_rsync_shell_for_node("node007"),
        infra.scheduler._ssh_target_for_node("node007")+":"+remote()+"/summary.json", str(HERE)+"/"], check=True, timeout=60)
    report = json.loads((HERE / "summary.json").read_text())
    for name in report["ready"]:
        if not marker(name).exists():
            write(marker(name), dict(ready=name))
    completed = {P["name"]+"/"+name for name in report["completed_labels"]}
    corrected = []
    with infra.scheduler.state_lock(purpose="reconcile verified bus alignment endpoints"):
        state = infra.scheduler.load_state()
        for row in state["tasks"]:
            reason = row.get("last_block_reason") or ""
            quiet = row.get("status") == "failed" and ("log only" in reason or "no success marker" in reason)
            pending = row.get("status") == "running" and "process exited; terminal log/result diagnosis pending" in reason
            if row.get("signature") in completed and row.get("exit_code") == 0 and (quiet or pending):
                row.update(status="done", last_block_reason="", failure_category=None,
                    completion_verified_from="Validated resident endpoint and zero process exit")
                if not row.get("finished_at"):
                    row["finished_at"] = time.time()
                corrected.append(row["id"])
        if corrected:
            infra.scheduler.save_state(state)
    if corrected:
        print(dict(completion_statuses_corrected=corrected), flush=True)
    return report


def watch():
    deadline = time.monotonic()+48*3600
    while time.monotonic()<deadline:
        try:
            report = collect()
            print({k:report[k] for k in ("smoke_pass", "formal_training", "formal_evaluations", "complete")}, flush=True)
            if report["complete"]:
                print("DONE bus alignment v5", flush=True)
                return
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-600:], flush=True)
        time.sleep(40)
    raise TimeoutError("Inspect incomplete resident follow-up shards")


def submit():
    if (HERE / "submission.json").exists():
        raise ValueError("Campaign already submitted")
    path = HERE / "jobs.json"
    write(path, jobs())
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(path), "--trusted", "--json"], text=True))
    write(HERE / "submission.json", result)
    print(dict(submitted=len(result["submitted"]), first=result["submitted"][0]["id"], last=result["submitted"][-1]["id"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "contracts", "submit", "collect", "watch"))
    mode = parser.parse_args().mode
    if mode == "collect":
        report = collect()
        print({k:report[k] for k in ("smoke_pass", "formal_training", "formal_evaluations", "complete")})
    else:
        globals()[mode]()
