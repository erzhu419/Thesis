"""Separate fitted action ranking from softmin averaging on frozen source states."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
import tempfile

import run as evaluation

HERE = Path(__file__).resolve().parent
runtime = evaluation.runtime
torch, np, base = runtime.torch, runtime.old.np, runtime.base


def stats(values):
    values = values.detach().numpy().reshape(-1)
    return dict(mean=float(values.mean()), minimum=float(values.min()),
                median=float(np.median(values)), maximum=float(values.max()))


def grid_costs(reference, states):
    n, width = len(states), len(reference.grid)
    tiled = states[:, None, :15].expand(n, width, 15).reshape(-1, 15)
    lines = (tiled[:, 0] * 12).round().long().clamp(0, 11)
    x = torch.cat((tiled[:, reference.indices], reference.grid.repeat(n)[:, None]), -1)
    x = torch.cat((x, torch.ones(len(x), 1)), -1)
    features = torch.ones(len(x), len(reference.factors))
    for ids in reference.factors.T:
        features *= x[:, ids]
    return (reference.coefficients[lines] * features).sum(-1).reshape(n, width)


def main():
    torch.set_num_threads(1)
    day, limit = 69426, 512
    samples, seen, rng = [], 0, random.Random(day)
    _, cap = evaluation.load_actor("symbolic0", 64001)

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

    with tempfile.TemporaryDirectory(prefix="thesis-symbolic-feedback-") as tmp:
        with redirect_stdout(StringIO()):
            env = base.make_env(Path(tmp), evaluation.P["horizon_seconds"], "normal")
            env.schedule = base.source_schedule("normal", evaluation.P["horizon_seconds"])
            with runtime.old.old.representation("source_clip", cap):
                row = base.rollout(env, FixedZero(), "global_event", day)
    states = torch.as_tensor(np.stack(samples))
    records = []
    with torch.no_grad():
        for seed in evaluation.P["training_seeds"]:
            reference = runtime.StageReference(seed).eval()
            costs = grid_costs(reference, states)
            hard = reference.grid[costs.argmin(-1)]
            original = reference(states).squeeze(-1)
            soft = (torch.softmax(-costs / runtime.P["reference_temperature"], -1) @ reference.grid)
            assert torch.allclose(soft.clamp(-runtime.P["reference_action_clip"], runtime.P["reference_action_clip"]),
                                  original, atol=1e-6, rtol=0)
            temperatures = {str(t): stats(30 * (torch.softmax(-costs / t, -1) @ reference.grid + 1))
                            for t in [.25, .05, .01, .002]}
            records.append(dict(seed=seed, current_soft_holding_seconds=stats(30 * (original + 1)),
                minimum_cost_holding_seconds=stats(30 * (hard + 1)),
                zero_is_minimum_cost_fraction=float((costs.argmin(-1) == 0).float().mean()),
                cost_increasing_on_whole_grid_fraction=float(((costs[:, 1:] - costs[:, :-1]) >= -1e-6).all(-1).float().mean()),
                normalized_cost_grid_range=stats(costs.max(-1).values - costs.min(-1).values),
                temperature_response=temperatures))
    result = dict(purpose="fitted action-ranking and temperature diagnostic; no controller performance or selection",
        source_scene="normal", day=day, trajectory_controller="fixed zero holding", sampled_states=len(samples),
        decisions=seen, timestamp_cost_identity_passed=row["timestamp_cost_identity_passed"], records=records)
    base.write(HERE / "feedback_probe.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
