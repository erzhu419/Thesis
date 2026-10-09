"""Constrain physical control without changing the recorded offline tuples."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def module(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("action18_previous_training", path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def run(args):
    headway = args.objective == "headway"
    previous = module(HERE.parent / ("mcwm_headway_alignment_v16/run.py" if headway else "bus_dual_aligned_v4/run.py"))
    original = previous.original if headway else previous
    import torch
    from dual_networks import Actor
    from SimpleSAC.transit_v4.networks import HurdleTransitActor

    class HoldingActor(Actor):
        def soft_action_components(self, states, *, deterministic_positive=False):
            parts = HurdleTransitActor.soft_action_components(self, states, deterministic_positive=deterministic_positive)
            for key in ("zero_action", "positive_action"):
                parts[key] = torch.cat((parts[key], torch.zeros_like(parts[key])), -1)
            return parts

        def action_log_prob(self, states, actions):
            return HurdleTransitActor.action_log_prob(self, states, actions[:, :1])

    original_bindings = original.bindings
    original.bindings = lambda: dict(original_bindings(), actor_class=HoldingActor)
    out = HERE / ("smoke" if args.smoke else "runs") / args.objective / args.scene / f"{args.arm}_seed{args.seed}"
    logical_arm = args.arm
    if args.phase == "contracts":
        actor = HoldingActor((256, 256))
        # Real offline states exercise the encoder's categorical and event-slot contract.
        if headway:
            from vehicle_data import Replay
        else:
            from chronological import Replay
        replay = Replay(args.objective, True)
        s = torch.as_tensor(replay.offline["observations"][:32], dtype=torch.float32)
        for deterministic in (False, True):
            a, logp = actor(s, deterministic=deterministic)
            assert a.shape == (32, 2) and torch.equal(a[:, 1], torch.zeros(32))
            assert torch.isfinite(logp).all()
            components = actor.soft_action_components(s)
            assert torch.equal(components["positive_action"][:, 1], torch.zeros(32))
        a, _ = actor(s, deterministic=True)
        b = a.clone(); b[:, 1] = .7
        assert torch.equal(actor.action_log_prob(s, a), actor.action_log_prob(s, b))
        a, _ = actor.repeated_actions_and_log_prob(s, 2)
        assert torch.equal(a[:, 1], torch.zeros(64))
        original.write(HERE / f"contracts_{args.objective}.json", dict(passed=True, policy_degrees_of_freedom=1,
            recorded_offline_speed_min=float(replay.offline["actions"][:, 1].min()),
            recorded_offline_speed_max=float(replay.offline["actions"][:, 1].max())))
        print("PASS holding action/likelihood/entropy contract", flush=True)
        return
    if headway:
        previous.folder = lambda *a, **k: out
        if args.phase == "evaluate":
            import evaluate_headway
            evaluate_headway.folder = lambda *a, **k: out
            import evaluate_dual
            evaluate_dual.Actor = HoldingActor
        if args.arm == "mcwm":
            args.arm = "mc_state"
        previous.run(args)
    else:
        original.policy_folder = lambda *a, **k: out
        if args.phase == "evaluate":
            import evaluate_dual
            evaluate_dual.policy_folder = lambda *a, **k: out
            evaluate_dual.Actor = HoldingActor
            evaluate_dual.evaluate(args)
        else:
            original.train(args)
    path = out / ("ready.json" if args.phase == "train" else args.layer+f"/case{args.case:02d}.json")
    row = json.loads(path.read_text())
    row.update(experiment=P["name"], arm=logical_arm, control_mode="holding", policy_degrees_of_freedom=1,
               critic_action_coordinates=2, offline_actions_preserved=True)
    if args.phase == "evaluate":
        assert row["mean_speed_ratio"] == 1.
    original.write(path, row)
    print("DONE matched holding control", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("contracts", "train", "evaluate"), required=True)
    parser.add_argument("--objective", choices=P["objectives"], required=True)
    parser.add_argument("--arm", choices=P["arms"], required=True)
    parser.add_argument("--scene", choices=P["scenes"], default="calibrated")
    parser.add_argument("--seed", type=int, default=P["training_seeds"][0])
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--layer", choices=P["executors"], default="guarded")
    parser.add_argument("--smoke", action="store_true")
    import torch
    torch.set_num_threads(1)
    run(parser.parse_args())
