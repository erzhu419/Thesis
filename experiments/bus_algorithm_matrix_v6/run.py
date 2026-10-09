"""Use one frozen multi-line runner for all matched comparisons."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PREVIOUS = HERE.parent / "bus_alignment_followup_v5"
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location("matrix6_cs_runner", PREVIOUS / "cs_run.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
import numpy as np
import torch
from resac_adapter import Agent, encode


def folder(item):
    prefix = "full_smoke" if item.get("full_smoke") else "smoke" if item["smoke"] else "runs"
    return HERE / prefix / item["mode"] / item["objective"] / f"{item['arm']}_seed{item['seed']}"


def reference_path(item):
    if item["arm"] in P["arms"]:
        return folder(item)
    if item["family"] == "cs_context" and item["mode"] == "holding":
        arm = "smooth_soft" if item["arm"] == "constant_symbolic" else "soft_no_symbolic"
        return HERE.parent / "bus_control_repair_v3/cs_runs" / item["objective"] / f"{arm}_seed{item['seed']}"
    return original_parent(item)


original_build, original_parent = previous.build, previous.parent
active_env = None


def metadata_environment(runtime):
    from contextlib import redirect_stdout
    from io import StringIO
    with tempfile.TemporaryDirectory(prefix="thesis-category-metadata-") as directory:
        directory = Path(directory)
        shutil.copy2(runtime["source"] / "config.json", directory / "config.json")
        (directory / "data").symlink_to(runtime["source"] / "data", target_is_directory=True)
        with redirect_stdout(StringIO()):
            return runtime["Env"](str(directory))


def build(item, kind, runtime):
    if item["arm"] not in P["arms"]:
        return original_build(item, kind, runtime)
    runtime["seed_all"](item["seed"])
    return Agent(active_env, item["arm"], item["mode"], item["seed"], P)


def contracts(args):
    runtime = previous.previous.setup("cs_native")
    global active_env
    active_env = metadata_environment(runtime)
    rows = []
    for mode in P["modes"]:
        dim = 1 if mode == "holding" else 2
        state = np.array([3/12, 4/25, 5/55, 2/24, 1., .4, .6, .2, .5, .1, .2, .1, .3, .4, .6], np.float32)
        assert np.array_equal(encode(state)[:5], [3, 4, 5, 2, 1])
        for arm in P["arms"]:
            item = dict(arm=arm, mode=mode, seed=args.seed)
            agent = build(item, "cs_native", runtime)
            a = np.linspace(-.5, .5, dim, dtype=np.float32)
            for i in range(P["batch_size"]):
                agent.replay_buffer.push(state, a, -float(i+1), state, 0.)
            assert np.allclose(agent.replay_buffer.actions[0], 30*(a+1))
            metrics = [agent.update(), agent.update()]
            assert all(np.isfinite(v) for row in metrics for v in row.values())
            command = previous.previous.physical_action(agent.select_action(state, True), "cs_native")
            assert 0 <= command[0] <= 60 and .8 <= command[1] <= 1.2
            if dim == 1:
                assert command[1] == 1.
            rows.append(dict(mode=mode, arm=arm, physical_action=[float(x) for x in command]))
    previous.write(HERE / "contracts.json", dict(passed=True, cases=rows))
    print(json.dumps(dict(passed=True, cases=len(rows))), flush=True)


def run(args):
    if args.phase == "contracts":
        contracts(args)
        return
    global active_env
    if args.arm in P["arms"]:
        # Only RE-SAC needs category metadata; symbolic references select their own frozen library.
        runtime = previous.previous.setup("cs_native")
        active_env = metadata_environment(runtime)
    previous.P = dict(previous.P, cs_episodes=1 if getattr(args, "full_smoke", False) else P["episodes"], cs_horizon_seconds=P["horizon_seconds"],
                      cs_scenes=P["scenes"], cs_evaluation_seeds=P["evaluation_seeds"])
    previous.folder, previous.parent, previous.build = folder, reference_path, build
    previous.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("contracts", "train", "evaluate"), required=True)
    parser.add_argument("--family", choices=("cs_recipe", "cs_context"), default="cs_recipe")
    parser.add_argument("--mode", choices=P["modes"], default="dual")
    parser.add_argument("--objective", choices=P["objectives"], default="headway")
    parser.add_argument("--arm", default="resac")
    parser.add_argument("--seed", type=int, default=P["training_seeds"][0])
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--layer", default="native")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-smoke", action="store_true")
    torch.set_num_threads(1)
    run(parser.parse_args())
    print("DONE matched algorithm matrix", flush=True)
