"""Time CPU/CUDA configurations without touching formal policies or outputs."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
runtime = json.loads((HERE.parent / "bus_control_v2/node_runtime.json").read_text())["train"]
os.environ.update(runtime["env"])
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  CS_BUS_ENV_ROOT=str(HERE.parent / "enhancement_v1/runtime/cs"))

import numpy as np
import torch

import cs
from repair_common import write
from settings import day_folder


def build(arm, device, threads):
    torch.set_num_threads(threads)
    agent = cs.build_agent(SimpleNamespace(arm=arm, seed=64001, objective="headway"))
    if device == "cuda":
        agent.device = torch.device("cuda")
        agent.config.device = "cuda"
        for net in (agent.actor, agent.critic, agent.target_critic, agent.pi_sym_torch):
            net.to(agent.device)
        agent.log_alpha = agent.log_alpha.detach().to(agent.device).requires_grad_(True)
        agent.alpha_optimizer = torch.optim.Adam([agent.log_alpha], lr=agent.config.lr_alpha)
        agent.actor_optimizer = torch.optim.Adam(
            (p for p in agent.actor.parameters() if p.requires_grad),
            lr=agent.config.lr_actor, weight_decay=agent.config.actor_weight_decay)
        agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=agent.config.lr_critic)
    return agent


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    output = HERE / "compute_benchmark/hybrid.json"
    gpu = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    torch.cuda.set_device(gpu)
    report = dict(gpu=torch.cuda.get_device_name(gpu), gpu_index=gpu,
                  micro=[], episodes=[], complete=False)
    with np.load(day_folder(64001, 3) / "transitions.npz") as data:
        indices = np.linspace(0, len(data["states"])-1, 1024, dtype=int)
        arrays = {key: data[key][indices] for key in ("states", "actions", "next_states", "costs")}

    def record(section, row):
        report[section].append(row)
        write(output, report)
        print(json.dumps(dict(section=section, **row)), flush=True)

    for arm in ("numeric_old", "smooth_hard"):
        for device, threads in [("cpu", 1), ("cpu", 8)] + [("cuda", t) for t in (1, 2, 4, 8)]:
            agent = build(arm, device, threads)
            for i in range(len(indices)):
                agent.store_transition(arrays["states"][i], arrays["actions"][i],
                    -float(arrays["costs"][i, 0]), arrays["next_states"][i], False)
            for _ in range(3):
                agent.update()
            seconds = []
            for _ in range(10):
                sync(device)
                started = time.perf_counter()
                agent.update()
                sync(device)
                seconds.append(time.perf_counter()-started)
            sync(device)
            started = time.perf_counter()
            for state in arrays["states"][:64]:
                agent.select_action(state, deterministic=True)
            sync(device)
            action_seconds = (time.perf_counter()-started)/64
            record("micro", dict(arm=arm, device=device, threads=threads,
                update_seconds=statistics.median(seconds), decision_seconds=action_seconds,
                compute_proxy_seconds=463*statistics.median(seconds)+9415*action_seconds,
                gpu_peak_mb=torch.cuda.max_memory_allocated()/1048576 if device == "cuda" else 0))
            del agent
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        candidates = [row for row in report["micro"]
                      if row["arm"] == arm and row["device"] == "cuda" and row["threads"] > 1]
        best = min(candidates, key=lambda row: row["compute_proxy_seconds"])["threads"]
        configurations = [("cpu", 1), ("cpu", 8), ("cuda", 1), ("cuda", best)]
        for device, threads in configurations:
            agent = build(arm, device, threads)
            durations = dict(update=0., decision=0.)
            original_update, original_action = agent.update, agent.select_action

            def measured_update(*args, **kwargs):
                before = time.perf_counter()
                value = original_update(*args, **kwargs)
                durations["update"] += time.perf_counter()-before
                return value

            def measured_action(*args, **kwargs):
                before = time.perf_counter()
                value = original_action(*args, **kwargs)
                durations["decision"] += time.perf_counter()-before
                return value

            agent.update, agent.select_action = measured_update, measured_action
            cs.old.seed_all(64001*1000)
            sync(device)
            started = time.perf_counter()
            print(json.dumps(dict(section="episode-start", arm=arm, device=device,
                                  threads=threads)), flush=True)
            with cs.old.environment("headway") as env:
                setup_seconds = time.perf_counter()-started
                with redirect_stdout(StringIO()):
                    rewards, decisions, updates = cs.old.run_episode(env, agent, config=agent.config)
            sync(device)
            record("episodes", dict(arm=arm, device=device, threads=threads,
                wall_seconds=time.perf_counter()-started, setup_seconds=setup_seconds,
                update_wall_seconds=durations["update"], decision_wall_seconds=durations["decision"],
                decisions=decisions, updates=updates, native_reward=sum(rewards.values()),
                gpu_peak_mb=torch.cuda.max_memory_allocated()/1048576 if device == "cuda" else 0))
            del agent.update, agent.select_action
            del agent, original_update, original_action
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    report["complete"] = True
    write(output, report)
    print(json.dumps(dict(complete=True, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
