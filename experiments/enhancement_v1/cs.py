"""NAU/MLP head package crossed with a fixed behavior-cloning coefficient."""

from study_common import EXPERIMENTS, P, arguments, seed_all, write
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time

os.environ["THESIS_CSBAPR_ROOT"] = str(EXPERIMENTS / "enhancement_v1/runtime/cs")
os.environ["OFFLINE_SUMO_ENV"] = str(Path(os.environ["THESIS_CSBAPR_ROOT"]) / "offline-sumo/env")
sys.path.insert(0, str(EXPERIMENTS / "csbapr_bus"))
from evaluate_recovery import (ObservedEnv, ObservedPolicy, ROOT, run_episode_multiline,
                               CSBAPRAgent, CSBAPRConfig, PROTOCOL, MultiLineEnv)
import torch


def configuration(arm, device):
    nau = arm.startswith("nau")
    return CSBAPRConfig(env_name="MultiLineBus", state_dim=15, action_dim=1,
                       max_episodes=P["episodes"], hidden_dim=64, num_critics=3, batch_size=256,
                       warmup_steps=500, lr_actor=3e-4, lr_critic=3e-4,
                       use_nau_actor=nau, beta_bc=.001 if arm.endswith("bc1") else 0.,
                       jac_weight=0., weight_sym=0., nau_reg_weight=.01 if nau else 0.,
                       actor_weight_decay=1e-4, device=device)


def main():
    args = arguments("cs")
    torch.set_num_threads(1)
    seed_all(args.seed)
    config = configuration(args.arm, "cpu" if args.evaluate else args.device)
    agent = CSBAPRAgent(15, 1, config)
    # Actor construction consumes different RNG counts across architectures.
    # Give all four arms exactly the same critic initialization for each seed.
    from csbapr.networks.critic import EnsembleQNet
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 100000)
        critic = EnsembleQNet(15, 1, 64, 3)
    agent.critic.load_state_dict(critic.state_dict())
    agent.target_critic.load_state_dict(critic.state_dict())
    del critic
    source = ROOT / "offline-sumo/env/calibrated_env"
    with tempfile.TemporaryDirectory(prefix="thesis-component-") as directory:
        directory = Path(directory)
        shutil.copy2(source / "config.json", directory / "config.json")
        os.symlink(source / "data", directory / "data", target_is_directory=True)
        if args.evaluate:
            payload = torch.load(args.out / "policy.pt", map_location="cpu", weights_only=False)
            agent.actor.load_state_dict(payload["actor"])
            agent.actor.eval()
            scene = list(P["cs_scenes"])[args.case // len(P["evaluation_seeds"])]
            seed = P["evaluation_seeds"][args.case % len(P["evaluation_seeds"])]
            seed_all(seed)
            started = time.monotonic()
            with redirect_stdout(StringIO()):
                env = ObservedEnv(str(directory))
                env.set_ood_schedule(P["cs_scenes"][scene])
                reward, decisions, _ = run_episode_multiline(
                    env, ObservedPolicy(agent, env), deterministic=True, train=False)
            if env.current_time != PROTOCOL["horizon_seconds"]:
                raise RuntimeError("Incomplete CS-BAPR test day")
            row = dict(arm=args.arm, training_seed=args.seed, evaluation_seed=seed,
                       scene=scene, case=args.case, reward=sum(reward.values()),
                       decisions=decisions, horizon_seconds=env.current_time,
                       windows=env.windows, elapsed_seconds=time.monotonic()-started)
            write(args.out / "results" / f"eval_{args.case:02d}.json", row)
            print(json.dumps({k: v for k, v in row.items() if k != "windows"}), flush=True)
            return
        if (args.out / "results/ready.json").exists() and (args.out / "policy.pt").exists():
            return
        if args.resume:
            raise ValueError("CS factorial uses uninterrupted fresh training, not actor-only resume")
        write(args.out / "results/protocol.json", dict(arm=args.arm, config=asdict(config),
                                                       training_seed=args.seed, hostname=socket.gethostname()))
        with redirect_stdout(StringIO()):
            env = MultiLineEnv(str(directory))
        history = []
        episodes = 1 if args.smoke else P["episodes"]
        for episode in range(episodes):
            seed_all(args.seed * 1000 + episode)
            started = time.monotonic()
            with redirect_stdout(StringIO()):
                rewards, decisions, updates = run_episode_multiline(env, agent, config=config)
            row = dict(episode=episode+1, reward=sum(rewards.values()), decisions=decisions,
                       updates=updates, horizon_seconds=env.current_time,
                       elapsed_seconds=time.monotonic()-started)
            history.append(row)
            write(args.out / "results/progress.json", row)
            print(json.dumps(row), flush=True)
        torch.save(dict(actor=agent.actor.state_dict(), config=asdict(config)), args.out / "policy.pt")
        write(args.out / "results/train_summary.json", dict(history=history,
                    device=config.device, hostname=socket.gethostname(),
                    actor_parameters=sum(p.numel() for p in agent.actor.parameters()),
                    critic_parameters=sum(p.numel() for p in agent.critic.parameters())))
        write(args.out / "results/ready.json", dict(arm=args.arm, training_seed=args.seed,
                                                   episodes=episodes))


if __name__ == "__main__":
    main()
