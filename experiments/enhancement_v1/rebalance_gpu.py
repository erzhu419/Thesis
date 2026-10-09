"""Use measured training budgets and move intact queued MC-WM groups to node007."""

import json
from datetime import datetime

from deploy import HERE, P, scheduler, spec
from study_common import write


def movable_groups(tasks):
    groups = {}
    for seed in (22104, 22105):
        signatures = {f"thesis-enhancement-v1/runs/mc/{arm}_seed{seed}/train"
                      for arm in P["mc_arms"]}
        rows = [t for t in tasks if t.get("signature") in signatures
                and t.get("status") != "cancelled"]
        if len(rows) == len(signatures) and all(t["status"] == "queued" for t in rows):
            groups[seed] = rows
    return groups


def rebalance():
    changes = []
    with scheduler.state_lock(purpose="right-size Thesis training and move complete GPU groups"):
        state = scheduler.load_state()
        groups = movable_groups(state["tasks"])
        moved_ids = {t["id"] for rows in groups.values() for t in rows}
        locations = json.loads((HERE / "locations.json").read_text())
        placement_path = HERE / "gpu_placements.json"
        placements = json.loads(placement_path.read_text()) if placement_path.exists() else {}
        for task in state["tasks"]:
            signature = task.get("signature") or ""
            if not signature.startswith("thesis-enhancement-v1/runs/") or not signature.endswith("/train"):
                continue
            if task["status"] not in ("running", "queued", "done"):
                continue
            _, _, kind, label, _ = signature.split("/")
            arm, seed = label.rsplit("_seed", 1)
            node = "node007" if task["id"] in moved_ids else task.get("node") or task["require_node"]
            target = spec(kind, arm, int(seed), node)
            updates = {k: target[k] for k in ("resource_family", "vram_resource_family")}
            if task["status"] != "done":
                updates.update(est_vram_mb=target["vram"], ram_mb=target["ram_mb"],
                               est_vram_mb_explicit=True, ram_mb_explicit=True)
            if task["id"] in moved_ids:
                updates.update(cmd=target["cmd"], require_node=node, extra_env=target["env"],
                               last_block_reason=None)
            changed = {k: v for k, v in updates.items() if task.get(k) != v}
            if changed:
                changes.append(dict(id=task["id"], signature=signature, status=task["status"],
                                    before={k: task.get(k) for k in changed}, after=changed))
                task.update(changed)
        for seed in groups:
            placements[f"mc/{seed}"] = "node007"
            for item in locations:
                if item["kind"] == "mc" and item["seed"] == seed and not item["smoke"]:
                    item["node"] = "node007"
        report = dict(time=datetime.now().isoformat(), moved_ids=sorted(moved_ids), changes=changes)
        write(HERE / "gpu_resource_rebalance.json", report)
        write(placement_path, placements)
        write(HERE / "locations.json", locations)
        scheduler.save_state(state)
    print(json.dumps(dict(moved_ids=report["moved_ids"], updated=len(changes)), indent=2))


if __name__ == "__main__":
    rebalance()
