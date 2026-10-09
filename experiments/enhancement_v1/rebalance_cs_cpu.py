"""Move only complete, unstarted CS-BAPR seed groups to shared CPU nodes."""

import json
from datetime import datetime
from deploy import HERE, P, scheduler, spec
from study_common import write


def queued_groups(tasks):
    groups = {}
    for seed in P["training_seeds"]:
        signatures = {f"thesis-enhancement-v1/runs/cs/{arm}_seed{seed}/train"
                      for arm in P["cs_arms"]}
        rows = [t for t in tasks if t.get("signature") in signatures]
        if len(rows) == len(signatures) and all(t["status"] == "queued" for t in rows):
            groups[seed] = rows
    return groups


def move():
    destinations = {22101: "node001", 22102: "node004", 22103: "node005", 22105: "node006"}
    changes = []
    with scheduler.state_lock(purpose="move queued CS-BAPR groups to CPU"):
        state = scheduler.load_state()
        groups = queued_groups(state["tasks"])
        path = HERE / "cs_cpu_placements.json"
        placements = json.loads(path.read_text()) if path.exists() else {}
        locations = json.loads((HERE / "locations.json").read_text())
        for seed, node in destinations.items():
            rows = groups.get(seed, [])
            for task in rows:
                arm = task["signature"].split("/")[3].rsplit("_seed", 1)[0]
                target = spec("cs", arm, seed, node)
                updates = dict(cmd=target["cmd"], require_node=node, est_vram_mb=0,
                               est_vram_mb_explicit=True, extra_env=target["env"],
                               allow_cpu_training=True,
                               cpu_training_justification=target["cpu_training_justification"],
                               last_block_reason=None)
                changes.append(dict(id=task["id"], signature=task["signature"],
                                    before={key: task.get(key) for key in updates}, after=updates))
                task.update(updates)
            if rows:
                placements[str(seed)] = node
                for item in locations:
                    if item["kind"] == "cs" and item["seed"] == seed and not item["smoke"]:
                        item["node"] = node
        write(HERE / "cs_cpu_migration.json", dict(time=datetime.now().isoformat(), changes=changes))
        write(path, placements)
        write(HERE / "locations.json", locations)
        scheduler.save_state(state)
    print(json.dumps(dict(moved=len(changes), task_ids=[c["id"] for c in changes]), indent=2))


if __name__ == "__main__":
    move()
