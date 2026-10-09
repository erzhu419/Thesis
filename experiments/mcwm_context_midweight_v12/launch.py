"""Unpinned CPU training and paired per-day SUMO evaluation."""

import argparse
import shlex

from midweight_common import HERE, P, CONTEXTS, bundle_path, module, write

previous = module("midweight_launcher", HERE.parent / "mcwm_context_mapping_v11/launch.py")
previous.HERE, previous.P, previous.CONTEXTS, previous.bundle_path = HERE, P, CONTEXTS, bundle_path
base, infra = previous.base, previous.infra
base.HERE, base.P, base.TRAIN_NODES = HERE, P, P["training_nodes"]


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(path)


previous.remote = base.remote = remote
collect, watch = previous.collect, previous.watch


def task(item, case=None):
    row = previous.task(item, case)
    row["description"] = f"Intermediate TD weight {item['name']} "+("train" if case is None else f"eval{case}")
    if case is None:
        row["cpu_training_justification"] = "Unchanged single-thread v10 CPU learner; 24 matched policies eligible for six 192-core nodes."
    return row


def stage():
    previous.stage()
    paths = [HERE.parent / "mcwm_context_mapping_v11" / name for name in
        ("run.py", "mapping_common.py", "routing.py", "protocol.json")]
    infra.scheduler.run_on("node007", "test "+" -a ".join("-s "+shlex.quote(remote("node007", p))
        for p in paths), timeout=40, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-eval", "submit", "collect", "watch"])
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode == "smoke":
        base.submit([task(i) for i in base.placements(True)], "smoke")
    elif mode == "smoke-eval":
        items = base.located_items(True)
        collect(items)
        base.handoff(items)
        assert all(base.gate(i).exists() for i in items)
        base.submit([task(i, 10) for i in items], "smoke_eval")
    elif mode == "collect":
        items = base.located_items()
        collect(items)
        base.handoff(items)
    elif mode == "watch":
        watch()
    else:
        items = base.placements()
        write(HERE / "locations.json", items)
        base.submit([task(i) for i in items]+[task(i, c) for i in items for c in range(10, 20)], "campaign")
        base.submit([dict(project="Thesis", description="Intermediate-weight result handoff", signature=P["name"]+"/watch",
            cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch", require_node="local", cpu=1,
            ram_mb=512, vram=0, allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True)], "watch")
