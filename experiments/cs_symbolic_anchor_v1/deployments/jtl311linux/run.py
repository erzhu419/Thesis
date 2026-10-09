"""Full symbolic-policy adaptation on the established passenger contract."""

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, str(HERE / "runtime"))
from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
import csbapr.agent as agent_module
import csbapr.losses.q_loss as q_module

path = HERE.parent / "cs_source_anchor_v1/run.py"
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("symbolic_anchor_parent", path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from policy import CenteredPolicy, StageReference, equilibrium_states, normalized_jacobian_loss
torch, base = old.torch, old.base


@dataclass
class InputConfig(CSBAPRConfig):
    # Preserve the two old diagnostic fields; neither enters learner updates.
    physics_smoothness_M: float = 0.0
    ood_confidence_delta: float = 0.05


base.CSBAPRConfig = InputConfig
agent_module.compute_jacobian_loss = normalized_jacobian_loss
q_module.compute_jacobian_loss = normalized_jacobian_loss


class Agent(old.AnchoredAgent):
    def update(self, **kwargs):
        row = super().update(**kwargs)
        # L_eff is undefined for this composed policy and is not a loss.
        return None if row is None else {k: v for k, v in row.items() if not k.startswith("L_eff")}


def build_agent(seed, weight, cap, symbolic=True):
    reference_actor = old.teacher(seed)
    native = base.CSBAPRAgent
    base.CSBAPRAgent = Agent
    try:
        agent = base.new_agent("global_coverage", seed)
    finally:
        base.CSBAPRAgent = native
    if symbolic:
        agent.config.actor_type = "tanh-mlp"
        agent.config.jac_weight = P["jac_weight"]
        agent.config.weight_sym = P["weight_sym"]
        agent.config.track_L_eff = False
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 50000)
            agent.actor = CenteredPolicy()
        agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=agent.config.lr_actor,
                                                weight_decay=agent.config.actor_weight_decay)
        agent.pi_sym_torch = StageReference(seed).eval().requires_grad_(False)
    agent.source_anchor = old.SourceAnchor(agent.actor, reference_actor, cap, weight, seed,
                                           P["reservoir_size"], P["anchor_batch_size"])
    return agent


def native_check(folder):
    reference = old.PRIOR / "preflight/source_clip/seed64001"
    before = torch.load(reference / "policy.pt", map_location="cpu", weights_only=False)
    after = torch.load(folder / "policy.pt", map_location="cpu", weights_only=False)
    assert all(after["config"][k] == v for k, v in before["config"].items())
    assert before["actor"].keys() == after["actor"].keys()
    assert all(torch.equal(v, after["actor"][k]) for k, v in before["actor"].items())
    a = json.loads((reference / "train_summary.json").read_text())["history"]
    b = json.loads((folder / "train_summary.json").read_text())["history"]
    assert len(a) == len(b) == 2
    for x, y in zip(a, b):
        for key in ["cost", "exact_cost", "decisions", "transitions", "updates", "mean_holding_seconds"]:
            assert x[key] == y[key], (key, x[key], y[key])
    return dict(native_config_fields_unchanged=True, final_actor_identical=True, execution_identical=True,
                added_inactive_config_fields=sorted(after["config"].keys() - before["config"].keys()))


def contracts():
    cap = json.loads((old.old.THRESHOLDS / "inputs/envelope.json").read_text())["daily_caps"][0]
    records = []
    for seed in P["training_seeds"]:
        a, b = build_agent(seed, 0., cap), build_agent(seed, 10., cap)
        assert asdict(a.config) == asdict(b.config)
        for component in ["actor", "critic", "target_critic"]:
            assert all(torch.equal(v, getattr(b, component).state_dict()[k])
                       for k, v in getattr(a, component).state_dict().items())
        states = torch.full((16, 92), .2)
        states[:, 0] = torch.arange(16) % 12 / 12
        states[:, 5] = .3
        states[:, 6] = .7
        states[:, 8] = .5
        mean, _ = a.actor(equilibrium_states(states))
        assert torch.equal(mean, a.actor.equilibrium_mean.expand_as(mean))
        target = a.pi_sym_torch(states)
        assert target.shape == (16, 1) and bool(torch.isfinite(target).all())
        assert torch.equal(target, a.pi_sym_torch(states[:, :15]))
        assert all(not p.requires_grad for p in a.pi_sym_torch.parameters())
        for state in states.numpy():
            b.source_anchor.observe(state)
        value, samples = b.source_anchor.prepare()
        loss_b = torch.tanh(b.actor(states)[0]).square().mean() + normalized_jacobian_loss(b.actor, b.pi_sym_torch, states)
        loss_a = torch.tanh(a.actor(states)[0]).square().mean() + normalized_jacobian_loss(a.actor, a.pi_sym_torch, states)
        (loss_a + 10 * old.SourceAnchor(a.actor, old.teacher(seed), cap, 0., seed).loss(states)).backward()
        loss_b.backward()
        b.source_anchor.clear()
        for x, y in zip(a.actor.parameters(), b.actor.parameters()):
            assert (x.grad is None) == (y.grad is None)
            if x.grad is not None:
                assert torch.allclose(x.grad, y.grad, atol=1e-6, rtol=1e-5)
        assert all(p.grad is None for p in b.source_anchor.teacher.parameters())
        records.append(dict(seed=seed, paired_initialization_identical=True, zero_residual_passed=True,
                            action_reference_shape_passed=True, anchor_symbolic_gradient_sum_passed=True,
                            anchor_loss=value, anchor_samples=samples))
    base.write(HERE / "contracts.json", dict(passed=True, records=records))
    return dict(symbolic_contracts_passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "train", "eval"])
    parser.add_argument("--arm", choices=list(P["arms"]) + ["identity"], required=True)
    parser.add_argument("--seed", type=int, choices=P["training_seeds"], default=64001)
    parser.add_argument("--domain", choices=["source", "target"])
    parser.add_argument("--day", type=int)
    parser.add_argument("--scene", choices=sorted(set(P["source_scenes"] + P["target_scenes"])))
    args = parser.parse_args()
    args.smoke = args.mode == "preflight"
    torch.set_num_threads(1)
    old.HERE, old.P = HERE, P
    old.new_agent = lambda seed, weight, cap: build_agent(seed, weight, cap, args.arm != "identity")
    old.native_check, old.gradient_check = native_check, contracts
    if args.mode == "eval":
        old.old.shared.GaussianPolicy = CenteredPolicy
        old.evaluate(args)
    else:
        old.train(args)
