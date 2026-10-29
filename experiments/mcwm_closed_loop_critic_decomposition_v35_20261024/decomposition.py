"""Read-only, same-weight residual and ensemble-min decomposition of v34.

The memory learner's base has its own learned parameters. Comparing that base
to the zero-memory learner measures between-learner base drift; masking its
residuals measures direct memory effects at fixed learned weights.
"""
from copy import deepcopy

import numpy as np
import torch

from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _restore


MODELS = ("zero_memory", "memory")


def summarize(q):
    """Two-action scores with each action's own minimum ensemble head."""
    minimum, heads = q.min(dim=0)
    mean = q.mean(dim=0)
    low_head = int(heads[0])
    delta = q[:, 1]-q[:, 0]
    minimum_delta = float(minimum[1]-minimum[0])
    fixed_low_delta = float(delta[low_head])
    return dict(q_heads=q.tolist(), delta_heads=delta.tolist(), min_heads=heads.tolist(),
        min_head_changes=bool(heads[0] != heads[1]), q_min=minimum.tolist(), q_mean=mean.tolist(),
        delta_min=minimum_delta, delta_mean=float(mean[1]-mean[0]),
        choice=int(minimum.argmax()), fixed_low_head=low_head, fixed_low_head_delta=fixed_low_delta,
        selection_gap=minimum_delta-fixed_low_delta)


def _matches(summary, reference):
    return (summary["q_min"] == reference["q_min"] and summary["q_mean"] == reference["q_mean"]
            and summary["choice"] == reference["predicted_choice"])


@torch.no_grad()
def decompose(learners, selected, actions, *, reference):
    """Replay v34's two executed actions and decompose their frozen Q values.

    ``selected`` is the saved causal issue record (s, memory_s.vector), ``actions``
    its low/high dictionary, and ``reference`` its saved prediction dictionary.
    Executed actions never pass through F again. All head indices are zero-based.
    """
    saved_rng = _rng()
    private = {arm: {name: deepcopy(getattr(learner, name).bit_generator.state)
                    for name in ("rng", "diagnostic_rng")} for arm, learner in learners.items()}
    try:
        memory_learner = learners["memory"]
        device = memory_learner.device
        states = torch.as_tensor(np.asarray(selected["s"]), dtype=torch.float32,
                                 device=device).reshape(1, 229).expand(2, -1)
        memory = torch.as_tensor(np.asarray(selected["memory_s"]["vector"]), dtype=states.dtype,
                                 device=device).reshape(1, 84).expand(2, -1)
        executed = torch.as_tensor([actions[arm] for arm in ("low", "high")],
                                  dtype=states.dtype, device=device).reshape(2, 1)
        critic = memory_learner.critic
        explicit = torch.cat((states, memory), dim=1)
        canonical, context = explicit[:, :229], explicit[:, 229:]
        # Match native input views as well as its addition order: a broadcast
        # versus a materialized context can select different float32 GEMM sums.
        base = critic.base(canonical, executed)
        level = (context @ critic.level_weight.T).T
        action = ((executed[:, 0]+1.)*.5)[None, :]*(context @ critic.action_weight.T).T
        native_full = critic(explicit, executed)
        reconstructed = base+(level+action)
        zero = learners["zero_memory"].critic(states, executed)
        matrices = dict(zero_memory=zero, memory_base=base, level_only=base+level,
                        action_only=base+action, full=native_full)
        variants = {name: summarize(q) for name, q in matrices.items()}
        full, baseline, comparator = variants["full"], variants["memory_base"], variants["zero_memory"]
        order_error = float((native_full-reconstructed).abs().max())
        scale = max(1., float(base.abs().max()), float(level.abs().max()), float(action.abs().max()))
        tolerance = 4.*torch.finfo(base.dtype).eps*scale
        checks = dict(memory_full_saved_prediction_exact=_matches(full, reference["memory"]),
            zero_saved_prediction_exact=_matches(comparator, reference["zero_memory"]),
            same_actual_actions=all(reference[arm]["executed_actions"] == executed[:, 0].tolist() for arm in MODELS),
            native_order_algebra_closes=order_error <= tolerance,
            level_same_head_action_delta_within_float32=float((level[:, 1]-level[:, 0]).abs().max()) <= tolerance)
        result = dict(variants=variants,
            components=dict(level_heads=level.tolist(), action_heads=action.tolist(),
                level_delta_heads=(level[:, 1]-level[:, 0]).tolist(),
                level_algebraic_delta_heads=[0.]*len(level),
                level_delta_max_roundoff=float((level[:, 1]-level[:, 0]).abs().max()),
                action_delta_heads=(action[:, 1]-action[:, 0]).tolist(),
                level_only_delta_roundoff_heads=((base+level)[:, 1]-(base+level)[:, 0]
                    -(base[:, 1]-base[:, 0])).tolist(),
                native_order_max_error=order_error, native_order_tolerance=tolerance,
                left_associative_max_error=float((native_full-((base+level)+action)).abs().max()),
                addition_order="native full = base + (level + action); all native tensors float32"),
            between_learner_base_drift=dict(head_q_difference=(base-zero).tolist(),
                head_delta_difference=((base[:, 1]-base[:, 0])-(zero[:, 1]-zero[:, 0])).tolist(),
                min_advantage_difference=baseline["delta_min"]-comparator["delta_min"],
                mean_advantage_difference=baseline["delta_mean"]-comparator["delta_mean"]),
            direct_memory_same_weights=dict(
                full_minus_base_min=full["delta_min"]-baseline["delta_min"],
                level_only_minus_base_min=variants["level_only"]["delta_min"]-baseline["delta_min"],
                action_only_minus_base_min=variants["action_only"]["delta_min"]-baseline["delta_min"],
                min_nonadditivity=full["delta_min"]-variants["level_only"]["delta_min"]
                    -variants["action_only"]["delta_min"]+baseline["delta_min"],
                full_minus_base_mean=full["delta_mean"]-baseline["delta_mean"]),
            total_memory_minus_zero_min=full["delta_min"]-comparator["delta_min"],
            executed_actions=executed[:, 0].tolist(), checks=checks)
    finally:
        for arm, learner in learners.items():
            for name, state in private[arm].items():
                getattr(learner, name).bit_generator.state = state
        _restore(saved_rng)
    result["checks"]["global_and_private_rng_restored"] = _equal(saved_rng, _rng()) and all(
        _equal(state, getattr(learners[arm], name).bit_generator.state)
        for arm, values in private.items() for name, state in values.items())
    result["integrity"] = dict(all_pass=all(result["checks"].values()))
    if not result["integrity"]["all_pass"]:
        raise RuntimeError("v35 frozen decomposition failed: "+str(
            [key for key, value in result["checks"].items() if not value]))
    return result
