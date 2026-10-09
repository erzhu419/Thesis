"""Prepare retries only for failed cells without a live or successful successor."""

import json
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
queue = json.loads(Path("/home/erzhu419/.claude/scheduler/queue.json").read_text())
latest = {}
for task in queue["tasks"]:
    if task.get("signature", "").startswith("thesis-bus-campaign-v3/"):
        latest[task["signature"]] = task
profile = json.loads((HERE / "deployment_jtl110gpu2.json").read_text())
specs = []
for signature, task in latest.items():
    if task["status"] != "failed":
        continue
    args = shlex.split(task["cmd"])
    args[0] = profile["python"]
    args = [arg.replace(task["cwd"], profile["cwd"]) for arg in args]
    specs.append(dict(project="Thesis", description=task["description"]+" (deployment repaired)",
                      signature=signature, cmd=shlex.join(args), cwd=profile["cwd"],
                      require_node="jtl110gpu2", env=profile["env"],
                      vram=5200, ram_mb=10000, cpu=2, allow_gpu_over_one_third=True,
                      ckpt_dir=task["ckpt_dir"].replace(task["cwd"], profile["cwd"]),
                      ckpt_glob="latest.pt", resume_flag="--resume", resume_managed_by_cmd=True,
                      result_dir=task["result_dir"].replace(task["cwd"], profile["cwd"]),
                      local_result_dir=task["local_result_dir"], skip_launch_staging=True))
(HERE / "deployment_retry_jobs.json").write_text(json.dumps(specs, indent=2)+"\n")
print("Failed cells needing a retry:", len(specs))
