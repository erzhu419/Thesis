"""Full dual actions in the frozen CS learner, execution and per-vehicle replay."""

from collections import defaultdict, deque
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from torch import nn

from actions import physical_action
from common import EXPERIMENTS, P, folder, module, write


def setup(kind):
    if kind == "cs_native":
        inherited = module("dual_cs_native_reward", EXPERIMENTS / "reward_cross_v1/cs.py")
        base = inherited.old
        return dict(inherited=inherited, base=base, Env=inherited.ObjectiveEnv,
            normalize=inherited.run_episode_multiline.__globals__["normalize_obs"],
            seed_all=base.seed_all, source=base.ROOT / "offline-sumo/env/calibrated_env",
            ledger="passenger_ledger", observed_protocol=base.PROTOCOL)
    repair = module("dual_cs_soft_repair", EXPERIMENTS / "bus_control_repair_v3/cs.py")
    base = repair.old
    runtime = __import__("bus_runtime")
    import csbapr.agent as agent_module
    import csbapr.losses.q_loss as q_module
    agent_module.compute_jacobian_loss = holding_jacobian_loss
    q_module.compute_jacobian_loss = holding_jacobian_loss
    return dict(inherited=repair, base=base, Env=runtime.ObjectiveEnv,
        normalize=runtime.run_episode.__globals__["normalize_obs"], seed_all=base.seed_all,
        source=runtime.ENVROOT / "offline-sumo/env/calibrated_env", ledger="ledger",
        observed_protocol=runtime.recovery.PROTOCOL)


def holding_jacobian_loss(policy, reference, batch):
    from csbapr.losses.jacobian import action_jacobian, deterministic_action
    with torch.enable_grad():
        states = batch.detach().requires_grad_(True)
        policy_jac = action_jacobian(deterministic_action(policy, states)[:, :1], states, True)
        target_jac = action_jacobian(reference(states), states, False).detach()
        error = (policy_jac-target_jac).square().sum((-2, -1))
        return (error/(1+target_jac.square().sum((-2, -1)))).mean()


def build_agent(args, runtime):
    dimension = 2 if args.mode == "dual" else 1
    base = runtime["base"]
    runtime["seed_all"](args.seed)
    if args.kind == "cs_native":
        config = base.configuration("nau_bc0" if args.arm == "legacy_nau" else "mlp_bc0", "cpu")
        config.action_dim = dimension
        config.max_episodes = P["episodes"]
        agent = base.CSBAPRAgent(15, dimension, config)
    else:
        symbolic = args.arm == "smooth_soft"
        config = base.CSBAPRConfig(env_name="MultiLineBus", state_dim=15, action_dim=dimension,
            max_episodes=P["episodes"], hidden_dim=64, num_critics=3, batch_size=256,
            warmup_steps=500, use_nau_actor=False, actor_type="tanh-mlp", beta_bc=0.,
            weight_reg=0., jac_weight=.01 if symbolic else 0., weight_sym=.01 if symbolic else 0.,
            nau_reg_weight=0., actor_weight_decay=1e-4, track_L_eff=False, device="cpu", alpha_cap=.6)
        agent = base.MeasuredAgent(15, dimension, config)
        from csbapr.networks.policy import GaussianPolicy
        from adaptation import equilibrium_states

        class DualSoftPolicy(GaussianPolicy):
            def __init__(self):
                super().__init__(15, dimension, 64, activation="tanh")
                self.equilibrium_mean = nn.Parameter(torch.zeros(dimension))

            def forward(self, states):
                mean, log_std = super().forward(states)
                projected_mean, _ = super().forward(equilibrium_states(states))
                return self.equilibrium_mean+mean-projected_mean, log_std

            def compute_L_eff(self):
                return float("inf")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed+50000)
            agent.actor = DualSoftPolicy()
        agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=config.lr_actor,
            weight_decay=config.actor_weight_decay)
        agent.pi_sym_torch = runtime["inherited"].reference(args.seed, args.objective) if symbolic else None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed+100000)
        from csbapr.networks.critic import EnsembleQNet
        critic = EnsembleQNet(15, dimension, 64, 3)
    agent.critic.load_state_dict(critic.state_dict())
    agent.target_critic.load_state_dict(critic.state_dict())
    return agent


def episode(env, agent, normalize, kind, *, train, deterministic, config=None):
    env.reset()
    states, _, _ = env.initialize_state()
    pending, commands = {}, {line:{bus:0. for bus in range(value.max_agent_num)}
                             for line, value in env.line_map.items()}
    recent = {name:deque(maxlen=getattr(config, "surprise_window", 1024)) for name in ("state", "action", "reward")}
    rewards, decisions, updates, queued = defaultdict(float), 0, 0, 0
    measured, latest = [], None

    def select(line, bus, raw_state):
        state = normalize(raw_state)
        action = np.asarray(agent.select_action(state, deterministic=deterministic), np.float32)
        command = physical_action(action, kind)
        pending[(line, bus)] = (state, action.copy())
        commands[line][bus] = command
        measured.append((float(command[0]), float(command[1]),
            .5*(abs(float(state[5]-state[8]))+abs(float(state[6]-state[8])))*600,
            abs(float(state[5]-state[6]))*600))

    for line, buses in states.items():
        for bus, observations in buses.items():
            if observations:
                select(line, bus, observations[-1])
    done = False
    while not done:
        states, reward_dict, done = env.step_to_event(commands)
        commands = {line:{bus:0. for bus in buses} for line, buses in commands.items()}
        for line, buses in states.items():
            for bus, observations in buses.items():
                if not observations:
                    continue
                state = normalize(observations[-1])
                reward = float(reward_dict.get(line, {}).get(bus, 0.))
                previous = pending.get((line, bus))
                # Retain the published runner's same-vehicle station settlement.
                if previous and int(previous[0][2]*55) != int(state[2]*55):
                    previous_state, previous_action = previous
                    if train:
                        agent.replay_buffer.push(previous_state, previous_action, reward, state, 0.)
                        recent["state"].append(previous_state)
                        recent["action"].append(previous_action)
                        recent["reward"].append(reward)
                    rewards[line] += reward
                    decisions += 1
                    queued += 1
                select(line, bus, observations[-1])
        if train and queued >= 20:
            queued = 0
            if agent.replay_buffer.size >= config.batch_size:
                latest = agent.update(recent_rollout={k:list(v) for k,v in recent.items()} if len(recent["reward"]) >= 8 else None)
                if not all(np.isfinite(v) for k,v in latest.items() if not k.startswith("L_eff")):
                    raise RuntimeError("Nonfinite CS learner update")
                updates += 1
    values = np.asarray(measured)
    return dict(native_reward=sum(rewards.values()), decisions=decisions, updates=updates,
        metrics=latest, mean_hold_seconds=float(values[:, 0].mean()),
        mean_speed_ratio=float(values[:, 1].mean()), headway_plan_mae_seconds=float(values[:, 2].mean()),
        headway_imbalance_seconds=float(values[:, 3].mean()))


def run(args):
    runtime = setup(args.kind)
    out = folder(args.kind, args.mode, args.objective, args.arm, args.seed, args.smoke)
    path = out / "results" / ("ready.json" if args.phase == "train" else f"eval_{args.case:02d}.json")
    if path.exists():
        return
    dimension = 2 if args.mode == "dual" else 1
    if args.arm == "fixed_speed":
        from types import SimpleNamespace
        agent = SimpleNamespace(select_action=lambda *values, **kw: np.array([-1., 1.][:dimension], np.float32))
    else:
        agent = build_agent(args, runtime)
        if args.phase == "evaluate":
            agent.actor.load_state_dict(torch.load(out / "policy.pt", map_location="cpu", weights_only=False)["actor"])
            agent.actor.eval()
    horizon = 3600 if args.smoke else P["horizon_seconds"][args.kind]
    runtime["observed_protocol"]["horizon_seconds"] = horizon
    runtime["Env"].objective = args.objective
    started, history = time.monotonic(), []
    with tempfile.TemporaryDirectory(prefix="thesis-legacy-dual-") as directory:
        directory = Path(directory)
        shutil.copy2(runtime["source"] / "config.json", directory / "config.json")
        (directory / "data").symlink_to(runtime["source"] / "data", target_is_directory=True)
        with redirect_stdout(StringIO()):
            env = runtime["Env"](str(directory))
        for line in env.line_map.values():
            line.max_time = max(line.max_time, horizon+line.time_step)
        if args.phase == "train":
            write(out / "results/protocol.json", dict(protocol=P, config=asdict(agent.config),
                kind=args.kind, mode=args.mode, objective=args.objective, arm=args.arm, seed=args.seed))
            for index in range(1 if args.smoke else P["episodes"]):
                runtime["seed_all"](args.seed*1000+index)
                with redirect_stdout(StringIO()):
                    row = episode(env, agent, runtime["normalize"], args.kind, train=True, deterministic=False, config=agent.config)
                row.update(episode=index+1, horizon_seconds=env.current_time, elapsed_seconds=time.monotonic()-started)
                history.append(row)
                write(out / "results/progress.json", row)
                print(row, flush=True)
            torch.save(dict(actor=agent.actor.state_dict(), config=asdict(agent.config), action_dim=dimension), out / "policy.pt")
            write(out / "results/train_summary.json", dict(history=history))
            write(path, dict(kind=args.kind, mode=args.mode, arm=args.arm, objective=args.objective,
                seed=args.seed, episodes=len(history), action_dim=dimension, training_wall_seconds=time.monotonic()-started))
        else:
            scene = list(P["cs_scenes"])[args.case//3]
            eval_seed = P["evaluation_seeds"][args.kind][args.case % 3]
            runtime["seed_all"](eval_seed)
            env.set_ood_schedule(P["cs_scenes"][scene])
            with redirect_stdout(StringIO()):
                row = episode(env, agent, runtime["normalize"], args.kind, train=False, deterministic=True)
            if env.current_time != horizon:
                raise RuntimeError("Incomplete fixed-horizon evaluation")
            ledger = getattr(env, runtime["ledger"])
            people = [p for line in env.line_map.values() for station in line.stations for p in station.total_passenger]
            row.update(kind=args.kind, mode=args.mode, arm=args.arm, objective=args.objective,
                training_seed=args.seed, scene=scene, evaluation_seed=eval_seed, case=args.case,
                horizon_seconds=horizon, generated_passengers=len(people), action_dim=dimension,
                completion_rate=sum(p.arrived for p in people)/len(people),
                generalized_seconds_per_passenger=(2*ledger.wait+ledger.ride)/len(people),
                wait_passenger_seconds=ledger.wait, ride_passenger_seconds=ledger.ride,
                wall_seconds=time.monotonic()-started)
            write(path, row)
