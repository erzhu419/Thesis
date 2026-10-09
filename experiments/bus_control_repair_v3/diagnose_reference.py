"""Diagnose reference saturation and derivative scale on the independent ID day."""

import sys

import numpy as np
import torch

from repair_common import HERE, EXPERIMENTS, P, write
from symbolic import reference, SoftResidualPolicy, normalized_jacobian_loss

sys.path.insert(0, str(EXPERIMENTS / "csbapr_symbolic_bus_v2"))
from adaptation import ResidualPolicy, load_reference
from settings import day_folder, model_folder
from csbapr.losses.jacobian import action_jacobian


def main():
    torch.set_num_threads(1)
    results = []
    for seed in P["cs_seeds"]:
        with np.load(day_folder(seed, 3) / "transitions.npz") as data:
            indices = np.linspace(0, len(data["states"])-1, 256, dtype=int)
            sample = torch.as_tensor(data["states"][indices], dtype=torch.float32)
        for objective in P["cs_objectives"]:
            for style in ("original", "smooth"):
                target = (load_reference(model_folder(seed), objective) if style == "original"
                          else reference(seed, objective))
                states = sample.detach().requires_grad_(True)
                actions = target(states)
                jac = action_jacobian(actions, states, False).detach()
                torch.manual_seed(seed+50000)
                hard = ResidualPolicy(target, True)
                torch.manual_seed(seed+50000)
                soft = SoftResidualPolicy()
                # Separate fixed draws here do not affect any training task's RNG.
                torch.manual_seed(seed+70000)
                logp = hard.evaluate(sample)[1].detach()
                hold = 30*(actions.detach()+1)
                results.append(dict(seed=seed, objective=objective, reference=style,
                    mean_reference_hold_seconds=float(hold.mean()),
                    saturated_reference_fraction=float((actions.detach().abs() > .99).float().mean()),
                    jacobian_squared_mean=float(jac.square().sum((-2, -1)).mean()),
                    jacobian_squared_p95=float(torch.quantile(jac.square().sum((-2, -1)), .95)),
                    hard_actor_initial_mean_log_probability=float(logp.mean()),
                    alpha_update_direction_at_initialization="increase" if float(logp.mean()) > 1 else "decrease",
                    normalized_soft_actor_initial_penalty=float(normalized_jacobian_loss(soft, target, sample).detach())))
    write(HERE / "reference_diagnosis.json", dict(sample="256 fixed observations per seed from independent identification day 3",
                                                results=results))
    print(results, flush=True)


if __name__ == "__main__":
    main()
