"""Two prespecified headway repairs, matched across MC-WM and H2O backbones."""

import argparse
import copy
import json
import sys

import torch

from followup_common import HERE, EXPERIMENTS, P, folder, write
sys.path.insert(0, str(EXPERIMENTS / "bus_dual_aligned_v4"))
import run as original
from dual_networks import Actor

OriginalTrainer = original.Trainer
OriginalReplay = original.Replay
original_config = original.config


def speed_kl(actor, reference, states):
    current = actor.parameters_for_state(states)
    with torch.no_grad():
        prior = reference.parameters_for_state(states)
    mean, log_std = current[:, 3], current[:, 4].clamp(-5, 2)
    prior_mean, prior_log_std = prior[:, 3], prior[:, 4].clamp(-5, 2)
    return (prior_log_std - log_std +
        (log_std.mul(2).exp() + (mean-prior_mean).square()) / (2*prior_log_std.mul(2).exp()) - .5).mean()


class Trainer(OriginalTrainer):
    variant = None

    def load(self, path, *, load_optimizers=True):
        result = super().load(path, load_optimizers=load_optimizers)
        if self.variant == "clip10":
            from dataclasses import replace
            self.config = replace(self.config, gradient_clip_norm=P["mc_gradient_clip_norm"])
        else:
            self.speed_reference = copy.deepcopy(self.actor).eval().requires_grad_(False)
        return result

    def _update_sac_actor(self, actor_states, *, temperature_states=None,
                         temperature_action_samples=1, collect_metrics=True):
        if self.variant != "speed_prior":
            return super()._update_sac_actor(actor_states, temperature_states=temperature_states,
                temperature_action_samples=temperature_action_samples, collect_metrics=collect_metrics)
        if not hasattr(self, "speed_reference"):
            # First online actor update follows the unchanged offline initializer.
            self.speed_reference = copy.deepcopy(self.actor).eval().requires_grad_(False)
        temperature_states = actor_states if temperature_states is None else temperature_states
        self._set_critic_grad(False)
        try:
            components = self.actor.soft_action_components(actor_states)
            actions = torch.cat((components["zero_action"], components["positive_action"]))
            features = self.critic.encode_states(actor_states)
            values = self._reduce_actor_q(self.critic.forward_features(torch.cat((features, features)), actions))
            zero_q, positive_q = values.split(len(actor_states))
            log_probability = (components["zero_probability"]*components["zero_log_prob"] +
                components["hold_probability"]*components["positive_log_prob"])
            sac_loss = (components["zero_probability"]*(self.alpha.detach()*components["zero_log_prob"]-zero_q) +
                components["hold_probability"]*(self.alpha.detach()*components["positive_log_prob"]-positive_q)).mean()
            prior_loss = speed_kl(self.actor, self.speed_reference, actor_states)
            actor_loss = sac_loss + P["mc_speed_prior_weight"]*prior_loss
            hold_log_probability = components["positive_log_prob"]-components["positive_conditional_log_prob"]
            self._last_actor_diagnostics = {
                "train/hybrid_entropy": -log_probability.mean().detach(),
                "train/hybrid_gate_entropy": -(components["zero_probability"]*components["zero_log_prob"] +
                    components["hold_probability"]*hold_log_probability).mean().detach(),
                "train/hybrid_weighted_positive_entropy":
                    -(components["hold_probability"]*components["positive_conditional_log_prob"]).mean().detach(),
                "train/hold_probability": components["hold_probability"].mean().detach(),
                "train/positive_action_mean": components["positive_action"].mean().detach(),
                "train/speed_prior_kl": prior_loss.detach(),
            }
            if self.config.temperature_state_source == "actor" and temperature_action_samples == 1:
                temperature_log_probability = log_probability.detach()
            else:
                with torch.no_grad():
                    temperature_log_probability = self._temperature_log_probability(temperature_states, temperature_action_samples)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = self._gradient_norm(self.actor.parameters(), collect_metrics=collect_metrics)
            self.actor_optimizer.step()
        finally:
            self._set_critic_grad(True)
        alpha_loss = -(self.log_alpha*(temperature_log_probability+self.config.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return actor_loss, alpha_loss, actor_grad_norm


class Replay(OriginalReplay):
    def __init__(self, objective, smoke=False):
        super().__init__(objective, smoke=False)


def contracts():
    from test_contract import example_state
    torch.manual_seed(901)
    actor = Actor((32, 32))
    reference = copy.deepcopy(actor).requires_grad_(False)
    states = torch.from_numpy(example_state()[None]).repeat(8, 1)
    assert abs(float(speed_kl(actor, reference, states))) < 1e-6
    with torch.no_grad():
        actor.network.network[-1].bias[3].add_(.5)
    loss = speed_kl(actor, reference, states)
    loss.backward()
    assert float(loss) > 0 and float(actor.network.network[-1].bias.grad[3]) > 0
    assert all(p.grad is None for p in reference.parameters())
    write(HERE / "contracts/mc.json", dict(passed=True, shifted_speed_kl=float(loss)))


def run(args):
    if args.phase == "contracts":
        contracts()
        return
    item = vars(args).copy()
    for key in ("phase", "case", "layer"):
        item.pop(key)
    out = folder(item)
    Trainer.variant = args.variant
    original.Trainer, original.Replay = Trainer, Replay

    original.config = original_config
    original.initializer = lambda value: EXPERIMENTS / "bus_dual_aligned_v4/inputs/headway" / f"seed{value.seed}/initializer.pt"
    original.policy_folder = lambda *a, **k: out
    args.objective = "headway"
    if args.phase == "train":
        original.train(args)
        record = json.loads((out / "ready.json").read_text())
        record.update(variant=args.variant, speed_prior_weight=P["mc_speed_prior_weight"] if args.variant == "speed_prior" else 0.,
                      gradient_clip_norm=P["mc_gradient_clip_norm"] if args.variant == "clip10" else None,
                      initializer=str(original.initializer(args)))
        write(out / "ready.json", record)
    else:
        import evaluate_dual
        evaluate_dual.policy_folder = lambda *a, **k: out
        evaluate_dual.evaluate(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="mc")
    parser.add_argument("--phase", choices=("contracts", "train", "evaluate"), required=True)
    parser.add_argument("--variant", choices=P["mc_variants"], default="speed_prior")
    parser.add_argument("--scene", choices=P["mc_scenes"], default="calibrated")
    parser.add_argument("--objective", default="headway")
    parser.add_argument("--arm", choices=P["mc_arms"], default="mcwm")
    parser.add_argument("--seed", type=int, default=P["mc_seeds"][0])
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--layer", choices=("direct", "guarded"), default="guarded")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.phase == "train" else 1)
    run(args)
    print("DONE MC headway followup", vars(args), flush=True)
