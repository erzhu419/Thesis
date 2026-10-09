"""Recover only smoke outputs removed by the initial scheduler code sync."""

import json
import shlex
import subprocess
from pathlib import Path
import deploy


def main():
    locations = [i for i in json.loads((deploy.HERE / "locations.json").read_text()) if i["smoke"]]
    missing = []
    for node in deploy.GPUS:
        items = [i for i in locations if i["node"] == node]
        code = """
import json
from pathlib import Path
root = Path(ROOT)
result = []
for item in ITEMS:
    path = root / 'smoke' / item['kind'] / (item['arm']+'_seed22000')
    policy = path / ('bundle/controller.pt' if item['kind']=='legacy' else 'policy.pt')
    ready = path / 'results/ready.json'
    old = ready.with_name(ready.name+'.staging_failed')
    exists = policy.is_file()
    if exists and not ready.exists() and old.exists():
        old.replace(ready)
    if not exists and ready.exists():
        ready.replace(old)
    result.append(dict(item, policy_present=exists))
print(json.dumps(result))
""".replace("ROOT", repr(deploy.root(node))).replace("ITEMS", repr(items))
        _, out, _ = deploy.scheduler.run_on(node, deploy.executable(node)+" -c "+shlex.quote(code),
                                            timeout=45, check=True)
        rows = json.loads(out)
        missing.extend(i for i in rows if not i["policy_present"])
        print(node, [(i["kind"], i["arm"], i["policy_present"]) for i in rows], flush=True)
    queue = json.loads((Path.home() / ".claude/scheduler/queue.json").read_text())["tasks"]
    jobs = []
    for item in missing:
        path = deploy.HERE / "smoke" / item["kind"] / (item["arm"]+"_seed22000")
        for marker in (path / "results/ready.json", path / "eval_ready.json"):
            if marker.exists():
                marker.replace(marker.with_name(marker.name+".staging_failed"))
        prefix = f"thesis-enhancement-v1/smoke/{item['kind']}/{item['arm']}_seed22000/train"
        if any(t.get("signature", "").startswith(prefix) and
               t["status"] in ("running", "launching", "queued") for t in queue):
            print("Existing producer remains active:", prefix, flush=True)
            continue
        job = deploy.spec(item["kind"], item["arm"], 22000, item["node"], smoke=True)
        job["signature"] += "/gatefix"
        jobs.append(job)
    for task in queue:
        if (task.get("signature", "").startswith("thesis-enhancement-v1/smoke/")
                and task["signature"].endswith("/eval0") and task["status"] == "failed"):
            if any(t.get("signature", "").startswith(task["signature"]+"/") and
                   t["status"] in ("running", "launching", "queued") for t in queue):
                continue
            _, kind, label, _ = task["signature"].split("/")[1:]
            arm = label.rsplit("_seed", 1)[0]
            job = deploy.spec(kind, arm, 22000, task["require_node"], smoke=True, case=0)
            job["signature"] += "/gatefix"
            jobs.append(job)
    ids = deploy.submit(jobs, "smoke_gate_repair")
    watcher = json.loads((deploy.HERE / "watcher_jobs.json").read_text())[0]
    watcher["signature"] += "/gatefix"
    watcher["stage_excludes"] = ["runs", "smoke", "runtime"]
    ids += deploy.submit([watcher], "watcher_gate_repair")
    subprocess.run(["python3", deploy.SCHEDULER, "dispatch"]+
                   [v for i in ids for v in ("--task-id", i)], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
