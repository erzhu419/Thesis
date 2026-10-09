"""Stage code only, submit paired CPU jobs, and hand off compact result metadata."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from confirmation_common import (HERE, P, CONTEXTS, module, write, bundle_path,
    initializer_folder, policy_folder)

infra = module("confirmation_infra", HERE.parent / "enhancement_v1/deploy.py")
COLLECT_NODES = ["node007", "node004", "node005", "node006"]


def remote(path=None):
    path = HERE if path is None else path
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC)


def placements(smoke=False):
    seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
    scenes = CONTEXTS[-1:] if smoke else CONTEXTS
    return [dict(seed=seed, scenario=scene, arm=arm, smoke=smoke,
        name=f"{scene}/{arm}_seed{seed}") for seed in seeds for scene in scenes for arm in P["arms"]]


def init_gate(seed, smoke=False):
    root = HERE.parent / P["initializer_experiment"]
    return root / "gates" / f"{'smoke_' if smoke else ''}initializer_seed{seed}.json"


def policy_gate(item):
    return HERE / "gates" / (("smoke_" if item["smoke"] else "")+item["name"].replace("/", "_")+".json")


def specification(description, signature, cmd, *, evaluation=False, smoke=False):
    phase = "eval" if evaluation else "smoke" if smoke else "train"
    family = P["name"]+"/"+phase
    row = dict(project="Thesis", description=description, signature=P["name"]+"/"+signature,
        cwd=str(HERE), cmd=cmd, allowed_nodes=P["training_nodes"], cpu=1,
        ram_mb=2500 if evaluation else 4000 if smoke else 13000, vram=0,
        env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        resource_family=family, ram_resource_family=family, vram_resource_family=family,
        skip_launch_staging=True, stage_excludes=["inputs", "smoke_inputs", "runs", "smoke", "gates", "rules"],
        allow_no_ckpt=True, allow_no_resume=True)
    if not evaluation:
        row.update(allow_cpu_training=True,
            cpu_training_justification="Paired single-thread CPU learner and independently trained IQL prefixes on six 192-core nodes.")
    return row


def init_task(seed, smoke=False):
    cmd = f"{HERE}/node_python -u {HERE}/node_run.py --initialize --seed {seed}"+(" --smoke" if smoke else "")
    phase = "smoke_inputs" if smoke else "inputs"
    return specification(f"Fresh IQL seed{seed}", f"{phase}/seed{seed}/train", cmd, smoke=smoke)


def task(item, case=None):
    part = "smoke" if item["smoke"] else "runs"
    folder = Path(part) / item["name"]
    cmd = (f"{HERE}/node_python -u {HERE}/node_run.py --arm {item['arm']} "
        f"--scenario {item['scenario']} --seed {item['seed']} --out {folder}")
    if item["smoke"]:
        cmd += " --smoke"
    phase = "train" if case is None else f"eval{case}"
    if case is not None:
        cmd += f" --evaluate --case {case}"
    row = specification(f"{HERE.name} {item['name']} {phase}", f"{part}/{item['name']}/{phase}",
        cmd, evaluation=case is not None, smoke=item["smoke"])
    row["wait_for_files"] = [str(init_gate(item["seed"], item["smoke"]) if case is None else policy_gate(item))]
    return row


def rule_task(rule, case, smoke=False):
    part = "rule_smoke" if smoke else "rules"
    cmd = f"{HERE}/node_python -u {HERE}/node_run.py --rule {rule} --case {case}"+(" --smoke" if smoke else "")
    return specification(f"{HERE.name} rule {rule} case{case}", f"{part}/{rule}/case{case}",
        cmd, evaluation=True, smoke=smoke)


def jobs(smoke=False):
    items = placements(smoke)
    cases = [0, 10] if smoke else range(20)
    rows = []
    if HERE.name == P["initializer_experiment"]:
        rows.extend(init_task(seed, smoke) for seed in (P["training_seeds"][:1] if smoke else P["training_seeds"]))
    rows.extend(task(i) for i in items if i["arm"] in P["training_arms"])
    rows.extend(task(i, c) for i in items for c in cases)
    rows.extend(rule_task(rule, c, smoke) for rule in P["rules"] for c in cases)
    return rows


def stage():
    if HERE.name == P["initializer_experiment"]:
        for index, seed in enumerate(P["training_seeds"]):
            frozen_seed = 6201+index % 2
            bundle = json.loads((HERE.parent / "mcwm_context_control_v10/adaptation" / f"seed{frozen_seed}.json").read_text())
            write(bundle_path(seed), dict(bundle, consumer_seed=seed, recognizer_origin_consumer=frozen_seed,
                selection_scope=P["selection"], test_outcomes_used=False))
        infra.sync("node007", str(HERE / "adaptation")+"/", remote(HERE / "adaptation"))
    infra.sync("node007", str(HERE)+"/", remote(),
        ["--include=*.py", "--include=protocol.json", "--exclude=*"])
    runtime = dict(train=dict(python=infra.executable("node007"), env=infra.environment("node007")),
        evaluate=dict(python=infra.executable("node001"), env=infra.environment("node001", True)))
    path = HERE / "deployments/node007/node_runtime.json"
    write(path, runtime)
    infra.sync("node007", path, remote())
    infra.scheduler.run_on("node007", "ln -sfn "+shlex.quote(infra.executable("node007"))+" "+
        shlex.quote(remote()+"/node_python"), timeout=40, check=True)
    required = [HERE.parent / "mcwm_alignment_v2/inputs/h2oplus_seed6201.pt",
        HERE.parent / "mcwm_context_mapping_v11/run.py", HERE.parent / "daganzo_baseline_v1/rules.py"]
    infra.scheduler.run_on("node007", "test "+" -a ".join("-s "+shlex.quote(remote(path)) for path in required),
        timeout=40, check=True)
    print("Staged code and small recognizer JSON only; data and policies stay on shared storage", flush=True)


def collect(smoke=False):
    failures = []
    prefix = "smoke_" if smoke else ""
    for node in COLLECT_NODES:
        try:
            command = (shlex.quote(infra.executable(node))+" -u "+shlex.quote(remote()+"/collect_results.py")
                + (" --smoke" if smoke else ""))
            infra.scheduler.run_on(node, command, timeout=120, check=True)
            sources = [infra.scheduler._ssh_target_for_node(node)+":"+remote()+"/"+prefix+name
                for name in ("analysis_summary.json", "handoff_status.json")]
            result = subprocess.run(["rsync", "-az", "-e", infra.scheduler._ssh_rsync_shell_for_node(node),
                *sources, str(HERE)+"/"],
                capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-250:])
            return node
        except (RuntimeError, subprocess.SubprocessError) as error:
            failures.append(f"{node}: {str(error)[-180:]}")
    raise RuntimeError("Shared JSON collection unavailable: "+"; ".join(failures))


def handoff(smoke=False):
    prefix = "smoke_" if smoke else ""
    status = json.loads((HERE / (prefix+"handoff_status.json")).read_text())
    assert status["experiment"] == P["name"] and status["smoke"] == smoke
    expected_updates = 64 if smoke else P["offline_updates_in_initializer"]
    for row in status["initializers"]:
        seed = row["seed"]
        assert seed in (P["training_seeds"][:1] if smoke else P["training_seeds"])
        gate = init_gate(seed, smoke)
        if not gate.exists():
            assert row["fresh_initialization"] and row["online_events"] == 0
            assert row["offline_updates"] == expected_updates
            write(gate, dict(seed=seed, offline_updates=expected_updates, shared_storage=True))
    policies = {(r["scenario"], r["arm"], r["seed"]):r for r in status["policies"]}
    for item in placements(smoke):
        row = policies.get((item["scenario"], item["arm"], item["seed"]))
        gate = policy_gate(item)
        if row is not None and not gate.exists():
            assert row["fresh_initialization"] and row["initializer_seed"] == item["seed"]
            assert row["continuation_updates"] == (64 if smoke else 20000)
            write(gate, dict(item, shared_storage=True, reference_experiment=P.get("reference_experiment")))


def submit(rows, name):
    output = HERE / (name+"_submission.json")
    if output.exists():
        raise ValueError("Submission already recorded: "+str(output))
    path = HERE / (name+"_jobs.json")
    write(path, rows)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(path),
        "--trusted", "--json"], text=True))
    write(output, result)
    ids = [r["id"] for r in result["submitted"]]
    print(json.dumps(dict(batch=name, count=len(ids), first=ids[0], last=ids[-1])), flush=True)


def watch(smoke=False):
    prefix = "smoke_" if smoke else ""
    deadline = time.monotonic()+48*3600
    while time.monotonic() < deadline:
        try:
            node = collect(smoke)
            handoff(smoke)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(str(error)[-700:], flush=True)
            time.sleep(45)
            continue
        report = json.loads((HERE / (prefix+"analysis_summary.json")).read_text())
        assert report["experiment"] == P["name"] and report["analysis_location"] == "server"
        write(HERE / (prefix+"summary.json"), report)
        if report.get("selection") is not None:
            write(HERE / "selection.json", report["selection"])
        print(json.dumps(dict(collect_node=node, **{k:report[k] for k in
            ("complete", "initialization_ready", "training_ready", "reference_ready", "evaluation_ready", "rule_ready")})), flush=True)
        if report["complete"]:
            return
        time.sleep(45)
    raise TimeoutError("Frozen experiment remains incomplete after 48 hours; no retuning or automatic seed expansion")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "smoke", "submit", "collect", "watch", "watch-smoke"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "collect":
        collect()
        handoff()
    elif mode.startswith("watch"):
        watch(mode == "watch-smoke")
    else:
        smoke = mode == "smoke"
        name = "smoke" if smoke else "campaign"
        submit(jobs(smoke), name)
        submit([dict(project="Thesis", description=f"{HERE.name} {name} JSON handoff",
            signature=P["name"]+f"/{name}/watch", cwd=str(HERE),
            cmd=f"python3 -u {HERE}/launch.py {'watch-smoke' if smoke else 'watch'}", require_node="local",
            cpu=1, ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], name+"_watch")


if __name__ == "__main__":
    main()
