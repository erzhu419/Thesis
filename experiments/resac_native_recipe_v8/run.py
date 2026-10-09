"""New recipe endpoints, using the existing frozen multi-line evaluation."""

import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import importlib.util
import inspect
import json
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

import torch

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
sys.path.insert(0, str(HERE.parent / "bus_algorithm_matrix_v6"))
spec = importlib.util.spec_from_file_location("native8_multiline", HERE.parent / "bus_algorithm_matrix_v6/run.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
from agent import Agent, check


def folder(args):
    prefix = "full_smoke" if args.phase == "preflight" else "runs"
    root = HERE / prefix
    if args.phase == "preflight":
        root /= args.device
    return root / args.mode / args.objective / f"{args.arm}_seed{args.seed}"


def episode_function():
    # Retain the frozen settlement logic; drain one update per five transitions.
    old = base.previous.previous.episode
    tree = ast.parse(inspect.getsource(old))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
            comparisons = [x for x in node.test.values if isinstance(x, ast.Compare)]
            if any(isinstance(x.left, ast.Name) and x.left.id == "queued" for x in comparisons):
                replacement = ast.parse("""if train:
    while queued >= config.training_freq:
        queued -= config.training_freq
        if agent.replay_buffer.size >= config.batch_size:
            latest = agent.update(recent_rollout={k:list(v) for k,v in recent.items()} if len(recent['reward']) >= 8 else None)
            if not all(np.isfinite(v) for k,v in latest.items() if not k.startswith('L_eff')):
                raise RuntimeError('Nonfinite native RE-SAC update')
            updates += 1
""").body[0]
                node.test, node.body = replacement.test, replacement.body
                break
    namespace = dict(old.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(old), "exec"), namespace)
    return namespace["episode"]


def run(args):
    runtime = base.previous.previous.setup("cs_native")
    runtime["seed_all"](args.seed)
    metadata = base.metadata_environment(runtime)
    agent = Agent(metadata, args.arm, args.mode, args.seed, P,
                  "cpu" if args.phase == "evaluate" else args.device)
    check(agent)
    out = folder(args)
    endpoint = out / f"budget{args.budget}"
    if args.phase == "evaluate":
        path = endpoint / f"eval_{args.case:02d}.json"
        if path.exists():
            return
        payload = torch.load(endpoint / "policy.pt", map_location="cpu", weights_only=False)
        agent.actor.load_state_dict(payload["actor"])
        agent.actor.eval()
    elif (out / "ready.json").exists():
        return
    runtime["observed_protocol"]["horizon_seconds"] = P["horizon_seconds"]
    runtime["Env"].objective = args.objective
    started = time.monotonic()
    episode = episode_function()
    with tempfile.TemporaryDirectory(prefix="thesis-resac-native-") as directory:
        directory = Path(directory)
        shutil.copy2(runtime["source"] / "config.json", directory / "config.json")
        (directory / "data").symlink_to(runtime["source"] / "data", target_is_directory=True)
        with redirect_stdout(StringIO()):
            env = runtime["Env"](str(directory))
        for line in env.line_map.values():
            line.max_time = max(line.max_time, P["horizon_seconds"] + line.time_step)
        if args.phase == "evaluate":
            scenes = base.P["scenes"]
            scene = list(scenes)[args.case // 3]
            test_seed = base.P["evaluation_seeds"][args.case % 3]
            runtime["seed_all"](test_seed)
            env.set_ood_schedule(scenes[scene])
            with redirect_stdout(StringIO()):
                row = episode(env, agent, runtime["normalize"], "cs_native", train=False, deterministic=True)
            assert env.current_time == P["horizon_seconds"]
            ledger = getattr(env, runtime["ledger"])
            people = [p for line in env.line_map.values() for station in line.stations for p in station.total_passenger]
            row.update(item=dict(mode=args.mode, objective=args.objective, arm=args.arm, seed=args.seed),
                budget=args.budget, case=args.case, scene=scene, evaluation_seed=test_seed,
                generated_passengers=len(people), cost=(2*ledger.wait+ledger.ride)/len(people),
                completion_rate=sum(p.arrived for p in people)/len(people), horizon_seconds=env.current_time)
            base.previous.write(path, row)
        else:
            history = []
            count = 1 if args.phase == "preflight" else P["episodes"]
            for index in range(count):
                runtime["seed_all"](args.seed*1000 + index)
                with redirect_stdout(StringIO()):
                    row = episode(env, agent, runtime["normalize"], "cs_native", train=True,
                                  deterministic=False, config=agent.config)
                row.update(episode=index+1, wall_seconds=time.monotonic()-started,
                           horizon_seconds=env.current_time, device=args.device)
                assert env.current_time == P["horizon_seconds"] and row["updates"] > 0
                history.append(row)
                base.previous.write(out / "progress.json", row)
                print(json.dumps(row), flush=True)
                if index+1 in P["checkpoints"]:
                    checkpoint = out / f"budget{index+1}"
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(actor=agent.actor.state_dict(), config=asdict(agent.config)), checkpoint / "policy.pt")
                    base.previous.write(checkpoint / "ready.json", dict(episodes=index+1, device=args.device,
                        horizon_seconds=env.current_time, wall_seconds=row["wall_seconds"]))
            base.previous.write(out / "train_summary.json", dict(history=history, config=asdict(agent.config)))
            base.previous.write(out / "ready.json", dict(episodes=count, device=args.device,
                wall_seconds=time.monotonic()-started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                peak_cuda_reserved_mb=torch.cuda.max_memory_reserved()/1024**2 if args.device == "cuda" else 0.))
    print("DONE native RE-SAC recipe", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--mode", choices=P["modes"], default="holding")
    parser.add_argument("--objective", choices=P["objectives"], default="headway")
    parser.add_argument("--arm", choices=P["arms"], default="resac")
    parser.add_argument("--seed", type=int, default=P["training_seeds"][0])
    parser.add_argument("--budget", type=int, choices=P["checkpoints"], default=100)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    torch.set_num_threads(1)
    run(parser.parse_args())
