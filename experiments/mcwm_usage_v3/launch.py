"""Stage isolated code and fan out full-day evaluations across six CPU nodes."""

from usage_common import HERE, EXPERIMENTS, P, module, write
import argparse
import json
import shlex
import subprocess
import time

infra = module("usage_infra", EXPERIMENTS/"enhancement_v1/deploy.py")
old_launch = module("usage_old_launch", EXPERIMENTS/"mcwm_alignment_v2/launch.py")


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


def placements(smoke=False):
    if smoke:
        arms = ["h2o_matched", "scaled_nomodel", "mcwm", "normalized", "fixed_half", "source_repair", "offline_only"]
        return [dict(arm=a, seed=6201, scenario="joint75", node="node007",
                     name=f"joint75/{a}_seed6201", index=i, smoke=True) for i, a in enumerate(arms)]
    nodes = ["node007", "jtl110gpu", "jtl311linux", "node007", "jtl110gpu2", "node007"]
    items = []
    for i, seed in enumerate(P["training_seeds"]):
        for j, scenario in enumerate(P["scenarios"]):
            arms = P["arms"] + (["offline_only"] if j == 0 else [])
            for arm in arms:
                items.append(dict(arm=arm, seed=seed, scenario=scenario["name"], node=nodes[3*i+j],
                                  name=f"{scenario['name']}/{arm}_seed{seed}", index=len(items), smoke=False))
    return items


def gate(item):
    return HERE/"gates"/(item["name"].replace("/", "_")+".json")


def task(item, case=None):
    evaluation = case is not None
    node = infra.CPUS[(item["index"]*10+case) % 6] if evaluation else item["node"]
    folder = HERE/("smoke" if item["smoke"] else "runs")/item["name"]
    cmd = (f"{infra.executable(node)} -u {remote(node)}/run.py --arm {item['arm']} "
           f"--scenario {item['scenario']} --seed {item['seed']} --out {remote(node, folder)}")
    if item["smoke"]:
        cmd += " --smoke"
    if evaluation:
        cmd += f" --evaluate --case {case}"
    phase = f"eval{case}" if evaluation else "train"
    family = P["name"]+("/eval" if evaluation else "/smoke" if item["smoke"] else "/train")
    row = dict(project="Thesis", description=f"MC-WM use {item['name']} {phase}",
        signature=f"{P['name']}/{'smoke' if item['smoke'] else 'runs'}/{item['name']}/{phase}",
        cwd=str(HERE), cmd=cmd, require_node=node, cpu=1,
        ram_mb=2500 if evaluation else 4000 if item["smoke"] else 13000,
        vram=0 if evaluation else 2200 if node == "jtl110gpu" else 1024,
        env=infra.environment(node, evaluation), resource_family=family,
        ram_resource_family=family, vram_resource_family=family+"/"+node,
        skip_launch_staging=True, stage_excludes=["runs", "smoke", "gates"],
        allow_no_ckpt=True, allow_no_resume=True, allow_gpu_over_one_third=True)
    if evaluation and not item["smoke"]:
        row["wait_for_files"] = [str(gate(item))]
    return row


def submit(tasks, name):
    output = HERE/(name+"_submission.json")
    if output.exists():
        raise ValueError("Submission already recorded: "+str(output))
    path = HERE/(name+"_jobs.json")
    write(path, tasks)
    result = json.loads(subprocess.check_output(["python3", infra.SCHEDULER, "submit-jsonl", "--file", str(path),
                                                "--trusted", "--json"], text=True))
    write(output, result)
    print(json.dumps(dict(batch=name, ids=[r["id"] for r in result["submitted"]])), flush=True)


def stage():
    for node in infra.GPUS:
        infra.sync(node, str(HERE)+"/", remote(node), ["--include=*.py", "--include=protocol.json", "--exclude=*"])
        path = remote(node, EXPERIMENTS/"mcwm_alignment_v2/inputs")
        cmd = "test -s "+shlex.quote(path+"/h2oplus_seed6201.pt")+" -a -s "+shlex.quote(path+"/h2oplus_seed6202.pt")
        infra.scheduler.run_on(node, cmd, timeout=40, check=True)
        print("staged "+node, flush=True)


def collect(items):
    # Once a host's policies are handed off, only CPU-node results remain to pull.
    nodes = ["node001"] + [n for n in infra.GPUS if not n.startswith("node")
                           and any(i["node"] == n and not gate(i).exists() for i in items)]
    for node in nodes:
        try:
            infra.scheduler.run_on(node, "true", timeout=30, check=True)
            for part in ("runs", "smoke"):
                (HERE/part).mkdir(exist_ok=True)
                r = subprocess.run(["rsync", "-az", "--include=*/", "--include=*.json", "--exclude=*", "-e",
                    infra.scheduler._ssh_rsync_shell_for_node(node),
                    infra.scheduler._ssh_target_for_node(node)+":"+remote(node)+f"/{part}/", str(HERE/part)+"/"],
                    capture_output=True, text=True, timeout=60)
                if r.returncode not in (0, 23):
                    print(f"collection pending {node}: {r.stderr[-300:]}", flush=True)
        except (RuntimeError, subprocess.SubprocessError) as error:
            print(f"collection pending {node}: {error}", flush=True)


def watch():
    items = placements()
    summarize = module("usage_collector", HERE/"collect_results.py").summarize
    deadline = time.monotonic()+24*3600
    while time.monotonic() < deadline:
        collect(items)
        for item in items:
            folder = HERE/"runs"/item["name"]
            if gate(item).exists() or not (folder/"results/ready.json").exists():
                continue
            try:
                if not item["node"].startswith("node"):
                    infra.scheduler.run_on(item["node"], "true", timeout=30, check=True)
                    old_launch.relay(item["node"], remote(item["node"], folder), ["policy.pt"], "node001", remote("node001", folder))
                write(gate(item), item)
            except (RuntimeError, subprocess.SubprocessError) as error:
                print(f"handoff pending {item['name']}: {error}", flush=True)
        result = summarize()
        write(HERE/"summary.json", result)
        print(json.dumps({k:result[k] for k in ("complete", "training_ready", "evaluation_ready")}), flush=True)
        if result["complete"]:
            return
        time.sleep(60)
    raise TimeoutError("MC-WM model-use experiment remains incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-eval", "submit", "collect", "watch"])
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "smoke":
        submit([task(i) for i in placements(True)], "smoke")
    elif mode == "smoke-eval":
        submit([task(placements(True)[5], 0)], "smoke_eval")
    elif mode == "collect":
        collect(placements())
    elif mode == "watch":
        watch()
    else:
        items = placements()
        write(HERE/"locations.json", items)
        submit([task(i) for i in items]+[task(i, c) for i in items for c in range(10)], "campaign")
        submit([dict(project="Thesis", description="MC-WM use result handoff", signature=P["name"]+"/watch",
            cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1,
            ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], "watch")
