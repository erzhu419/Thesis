"""Parallel collection, shared prefixes, independent continuations and test days."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("headway_dispatch",HERE.parent / "mcwm_attribution_v15/launch.py")
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def jobs(smoke=False):
    prefix = "smoke" if smoke else "runs"
    extra = " --smoke" if smoke else ""
    rows = []
    shards = range(2 if smoke else P["offline_shards"])
    for shard in shards:
        rows.append(dispatch.task(HERE,P,f"{prefix}/data/shard{shard}",f"--phase collect --shard {shard}"+extra))
    merged = f"{prefix}/data/merge"
    rows.append(dispatch.task(HERE,P,merged,"--phase merge"+extra,[f"{prefix}/data/shard{s}" for s in shards],ram=1800))
    seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
    scenes = P["scenes"][:1] if smoke else P["scenes"]
    for seed in seeds:
        init = f"{prefix}/init/{seed}"
        rows.append(dispatch.task(HERE,P,init,f"--phase initialize --seed {seed}"+extra,[merged],training=True,ram=2500))
        for scene in scenes:
            for arm in P["arms"]:
                label = f"{prefix}/{scene}/{arm}_seed{seed}"
                fields = f"--scene {scene} --arm {arm} --seed {seed}"+extra
                rows.append(dispatch.task(HERE,P,label,"--phase train "+fields,[init],training=True,ram=3000))
                for layer in P["executors"]:
                    for case in ([0] if smoke else P["cases"]):
                        rows.append(dispatch.task(HERE,P,label+f"/eval/{layer}/{case}",
                            f"--phase evaluate --layer {layer} --case {case} "+fields,[label]))
    for arm in ("daganzo_duet","zero"):
        for layer in P["executors"]:
            for case in ([0] if smoke else P["cases"]):
                label = f"{prefix}/rule/{arm}/{layer}/{case}"
                rows.append(dispatch.task(HERE,P,label,f"--phase evaluate --arm {arm} --layer {layer} --case {case}"+extra))
    if smoke:
        rows.append(dispatch.watcher(HERE,P))
    return rows


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("stage","smoke","submit","watch","collect"))
    mode = parser.parse_args().mode
    if mode=="stage":
        runtime = json.loads((HERE.parent / "bus_control_v2/node_runtime.json").read_text())
        dispatch.stage(HERE,runtime)
    elif mode=="watch":
        dispatch.watch(HERE)
    elif mode=="collect":
        print(json.dumps(dispatch.collect(HERE)))
    else:
        smoke = mode=="smoke"
        dispatch.submit(HERE,P,jobs(smoke),"smoke" if smoke else "campaign")
