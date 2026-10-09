"""Compare frozen responses on one shared source-state sample, server-side."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
import tempfile

import run as runtime
from policy import CenteredPolicy, StageReference, equilibrium_states
from csbapr.losses.jacobian import action_jacobian, deterministic_action

HERE = Path(__file__).resolve().parent
torch, np, base = runtime.torch, runtime.old.np, runtime.base


def describe(values):
    values = values.detach().cpu().numpy().reshape(-1)
    return {name: float(value) for name, value in dict(
        mean=values.mean(), std=values.std(), minimum=values.min(),
        q05=np.quantile(values, .05), q95=np.quantile(values, .95),
        maximum=values.max()).items()}


def main():
    torch.set_num_threads(1)
    day, limit = 69426, 512
    rng, samples, seen = random.Random(day), [], 0
    checkpoint = torch.load(HERE / "runs/symbolic0/seed64001/policy.pt",
                            map_location="cpu", weights_only=False)

    class FixedZero:
        def select_action(self, state, deterministic):
            nonlocal seen
            assert deterministic
            seen += 1
            if len(samples) < limit:
                samples.append(state.copy())
            else:
                index = rng.randrange(seen)
                if index < limit:
                    samples[index] = state.copy()
            return np.array([-1.], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="thesis-symbolic-response-") as tmp:
        with redirect_stdout(StringIO()):
            env = base.make_env(Path(tmp), runtime.P["horizon_seconds"], "normal")
            env.schedule = base.source_schedule("normal", runtime.P["horizon_seconds"])
            with runtime.old.old.representation("source_clip", checkpoint["source_cap"]):
                rollout = base.rollout(env, FixedZero(), "global_event", day)
    states = torch.as_tensor(np.stack(samples))
    records = []
    for seed in runtime.P["training_seeds"]:
        reference = StageReference(seed).eval().requires_grad_(False)
        with torch.no_grad():
            ref_holds = 30 * (reference(states) + 1)
        inputs = states.detach().requires_grad_(True)
        jac_ref = action_jacobian(reference(inputs), inputs, False)[..., :15].detach()
        for arm in runtime.P["arms"]:
            payload = torch.load(HERE / "runs" / arm / f"seed{seed}/policy.pt",
                                 map_location="cpu", weights_only=False)
            actor = CenteredPolicy(92, 1, payload["config"]["hidden_dim"]).eval()
            actor.load_state_dict(payload["actor"])
            with torch.no_grad():
                holds = 30 * (deterministic_action(actor, states) + 1)
                projected_holds = 30 * (deterministic_action(actor, equilibrium_states(states)) + 1)
                delta = (holds - projected_holds).abs()
            inputs = states.detach().requires_grad_(True)
            jac_actor = action_jacobian(deterministic_action(actor, inputs), inputs, False)[..., :15].detach()
            discrepancy = ((jac_actor - jac_ref).square().sum((-2, -1))
                           / (1 + jac_ref.square().sum((-2, -1))))
            records.append(dict(arm=arm, seed=seed, holding_seconds=describe(holds),
                equilibrium_holding_seconds=describe(projected_holds),
                absolute_residual_holding_seconds=describe(delta),
                reference_holding_seconds=describe(ref_holds),
                normalized_symbolic_jacobian_discrepancy=describe(discrepancy),
                actor_local_jacobian_norm=describe(jac_actor.norm(dim=(-2, -1))),
                reference_local_jacobian_norm=describe(jac_ref.norm(dim=(-2, -1)))))
    result = dict(purpose="frozen same-state response diagnostic, not a policy performance test",
        scene="source normal", day=day, trajectory_controller="fixed zero holding",
        decisions=seen, sampled_states=len(samples), records=records,
        trajectory_timestamp_cost_identity_passed=rollout["timestamp_cost_identity_passed"])
    base.write(HERE / "response_probe.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
