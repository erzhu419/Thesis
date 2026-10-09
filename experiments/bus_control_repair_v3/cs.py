"""Independent paired CS-BAPR repairs; the frozen experiment stays unchanged."""

import argparse
import sys

import torch

from repair_common import EXPERIMENTS, P, cs_folder, module

sys.path.insert(0, str(EXPERIMENTS / "csbapr_symbolic_bus_v2"))
old = module("repair_cs_previous", EXPERIMENTS / "csbapr_symbolic_bus_v2/run.py")
from adaptation import ResidualPolicy, load_reference
from symbolic import SoftResidualPolicy, normalized_jacobian_loss, reference
import csbapr.agent as agent_module
import csbapr.losses.q_loss as q_module

# Both actor and detached critic penalties must use the same normalized quantity.
agent_module.compute_jacobian_loss = normalized_jacobian_loss
q_module.compute_jacobian_loss = normalized_jacobian_loss
old.run_folder = cs_folder
old.P = dict(old.P, name=P["name"], episodes=P["cs_episodes"], arms=P["cs_arms"],
             repair=P, preservation="Independent repair outputs; original policies and data unchanged")


def build_agent(args):
    symbolic = args.arm != "soft_no_symbolic"
    config = old.CSBAPRConfig(env_name="MultiLineBus", state_dim=15, action_dim=1,
        max_episodes=P["cs_episodes"], hidden_dim=64, num_critics=3, batch_size=256,
        warmup_steps=500, use_nau_actor=False, actor_type="tanh-mlp", beta_bc=0.,
        weight_reg=0., jac_weight=.01 if symbolic else 0., weight_sym=.01 if symbolic else 0.,
        nau_reg_weight=0., actor_weight_decay=1e-4, track_L_eff=False, device="cpu",
        alpha_cap=P["alpha_cap"])
    old.seed_all(args.seed)
    agent = old.MeasuredAgent(15, 1, config)
    target = (load_reference(old.model_folder(args.seed), args.objective)
              if args.arm == "numeric_old" else reference(args.seed, args.objective))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 50000)
        agent.actor = ResidualPolicy(target, True) if args.arm in ("numeric_old", "smooth_hard") else SoftResidualPolicy()
        torch.manual_seed(args.seed + 100000)
        from csbapr.networks.critic import EnsembleQNet
        critic = EnsembleQNet(15, 1, 64, 3)
    agent.critic.load_state_dict(critic.state_dict())
    agent.target_critic.load_state_dict(critic.state_dict())
    agent.actor_optimizer = torch.optim.Adam((p for p in agent.actor.parameters() if p.requires_grad),
        lr=config.lr_actor, weight_decay=config.actor_weight_decay)
    agent.pi_sym_torch = target if symbolic else None
    return agent


old.build_agent = build_agent
# The direct-reference control is evaluated with the same repaired feedback.
old.load_reference = lambda folder, objective: reference(int(folder.name.removeprefix("seed")), objective)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--objective", choices=P["cs_objectives"], required=True)
    parser.add_argument("--arm", choices=P["cs_arms"] + ["fitted_reference"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    getattr(old, args.phase)(args)


if __name__ == "__main__":
    main()
