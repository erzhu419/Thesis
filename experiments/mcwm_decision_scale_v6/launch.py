"""Reuse the unpinned launcher and node-local deployment from v5."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_usage_v3"))
from usage_common import module, write
base = module("decision_scale_launcher", HERE.parent/"mcwm_explore_v5/launch.py")
base.HERE = HERE
base.P = dict(base.P, **json.loads((HERE/"protocol.json").read_text()))


def remote(node, path=HERE):
    return str(path).replace("/home/erzhu419/mine_code", base.infra.HPC) if node.startswith("node") else str(path)


base.remote = remote


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["stage", "smoke", "smoke-eval", "submit", "collect", "watch"])
    mode = parser.parse_args().mode
    if mode == "stage":
        base.stage()
    elif mode == "smoke":
        base.submit([base.task(i) for i in base.placements(True)], "smoke")
    elif mode == "smoke-eval":
        items = base.located_items(True)
        base.collect(items)
        base.handoff(items)
        assert all(base.gate(i).exists() for i in items)
        base.submit([base.task(i, 0) for i in items], "smoke_eval")
    elif mode == "collect":
        items = base.located_items()
        base.collect(items)
        base.handoff(items)
    elif mode == "watch":
        base.watch()
    else:
        items = base.placements()
        write(HERE/"locations.json", items)
        base.submit([base.task(i) for i in items]+[base.task(i,c) for i in items for c in range(10)], "campaign")
        base.submit([dict(project="Thesis", description="Decision scale results handoff",
            signature=base.P["name"]+"/watch", cwd=str(HERE), cmd=f"python3 -u {HERE}/launch.py watch",
            require_node="local", cpu=1, ram_mb=512, vram=0, allow_no_ckpt=True,
            allow_no_resume=True, skip_launch_staging=True)], "watch")
