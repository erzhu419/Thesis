"""Move only unstarted campaign cells to the staged GPU environments."""

import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

HERE = Path(__file__).resolve().parent
nodes = ["node007", "jtl110gpu", "node007", "jtl311linux"]
profiles = {node: json.loads((HERE / f"deployment_{node}.json").read_text())
            for node in set(nodes)}
changed = []
with scheduler.state_lock(purpose="thesis-campaign-placement"):
    state = scheduler.load_state()
    for task in state["tasks"]:
        if not task.get("signature", "").startswith("thesis-bus-campaign-v3/"):
            continue
        if task["status"] != "queued" or "--skip-evaluation" in task["cmd"]:
            continue
        node = nodes[len(changed) % len(nodes)]
        profile = profiles[node]
        args = shlex.split(task["cmd"])
        args[0] = profile["python"]
        args = [arg.replace(str(HERE), profile["cwd"]) for arg in args]
        args.append("--skip-evaluation")
        task.update(cmd=shlex.join(args), cwd=profile["cwd"], require_node=node,
                    extra_env=profile["env"],
                    est_vram_mb=5200 if "--blocks 6" in task["cmd"] else 2400,
                    allow_gpu_over_one_third=True)
        for key in ("result_dir", "ckpt_dir"):
            task[key] = task[key].replace(str(HERE), profile["cwd"])
        for key in ("resume_scan_at", "resume_scan_completed_at", "resume_scan_key",
                    "resume_locations", "last_dispatch_error"):
            task.pop(key, None)
        changed.append({"id": task["id"], "node": node, "signature": task["signature"],
                        "local_result_dir": task["local_result_dir"]})
    scheduler.save_state(state)
if changed:
    (HERE / "campaign_v3_placement.json").write_text(json.dumps(changed, indent=2)+"\n")
print(json.dumps(changed, indent=2))
