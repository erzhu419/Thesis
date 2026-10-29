"""Frozen two-action critic attribution with the original forward as reference.

These models share the memory level across heads. A same-weight deletion can
identify its direct path; a between-learner base change includes all past
learning and does not identify a particular target term as its cause.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.score import _aliases
from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import SharedLevelQEnsemble


def summarize(q):
    """Keep both action levels, head contrasts and the actual min-head choice."""
    minimum, heads = q.min(0)
    mean = q.mean(0)
    contrasts = q[:, 1] - q[:, 0]
    low_head = int(heads[0])
    return dict(q_heads=q.tolist(), q_min=minimum.tolist(), q_mean=mean.tolist(),
        min_head=heads.tolist(), min_head_switch=bool(heads[0] != heads[1]),
        delta_heads=contrasts.tolist(), delta_min=float(minimum[1]-minimum[0]),
        delta_mean=float(mean[1]-mean[0]), predicted_choice=int(minimum.argmax()),
        fixed_low_head=low_head, fixed_low_head_delta=float(contrasts[low_head]),
        selection_gap=float(minimum[1]-minimum[0]-contrasts[low_head]))


@torch.no_grad()
def decompose_frozen(learner, state, memory, executed_actions, *, reference=None,
                     critic_name="critic"):
    """Score an issue's already executed low/high actions, without calling F.

    The batched input and alias handling match v55 ``score_frozen``. Returned
    values are plain scalars/lists. No actor, target estimator, RNG, optimizer,
    replay admission or parameter mutation is used. ``reference`` optionally
    checks the complete saved head matrix, min/mean, choice and action values.
    """
    critic = getattr(learner, critic_name)
    if not isinstance(critic, SharedLevelQEnsemble):
        raise TypeError("v56 requires the frozen v54 shared-level critic")
    states = torch.as_tensor(np.asarray(state), dtype=torch.float32,
        device=learner.device).reshape(1, 229)
    vector = memory["vector"] if isinstance(memory, dict) else memory
    context = torch.as_tensor(np.asarray(vector), dtype=states.dtype,
        device=learner.device).reshape(1, 84)
    actions = states.new_tensor(executed_actions).reshape(-1, 1)
    if len(actions) != 2:
        raise ValueError("v56 decomposes one declared low/high action pair")
    aliases = _aliases(actions[:, 0].tolist())
    representatives = [aliases.index(i) for i in range(max(aliases)+1)]
    explicit = critic_states(learner, states, context).repeat(len(representatives), 1)
    canonical, repeated_memory = explicit[:, :229], explicit[:, 229:]
    unique_actions = actions[representatives]
    base = critic.base(canonical, unique_actions)
    shared = (repeated_memory @ critic.level_weight.T).T.expand_as(base)
    interaction = ((unique_actions[:, 0]+1.)*.5)[None, :] * (
        repeated_memory @ critic.action_weight.T).T
    native = critic(explicit, unique_actions)
    # Match native addition order. This tests the implemented decomposition;
    # the double-precision residual below measures native float32 roundoff.
    reconstructed = base + (shared + interaction)
    closure32 = float((native-reconstructed).abs().max())
    components = {name: value[:, aliases] for name, value in (
        ("base", base), ("shared_level", shared),
        ("action_interaction", interaction))}
    full = native[:, aliases]
    matrices = dict(full=full, no_level=components["base"]+components["action_interaction"],
        no_interaction=components["base"]+components["shared_level"], base=components["base"])
    variants = {name: summarize(value) for name, value in matrices.items()}
    sum64 = sum(value.double() for value in components.values())
    contrast64 = sum(value[:, 1].double()-value[:, 0].double()
                     for value in components.values())
    native_contrast64 = full[:, 1].double()-full[:, 0].double()
    scale = max(1., *(float(value.abs().max()) for value in components.values()))
    tolerance = 4.*torch.finfo(full.dtype).eps*scale
    full_summary = variants["full"]
    shared_roundoff = float((components["shared_level"][:, 1]-components["shared_level"][:, 0]).abs().max())
    checks = dict(native_forward_reconstructed_float32=closure32 == 0.,
        shared_level_action_contrast_within_float32_roundoff=shared_roundoff <= tolerance,
        float64_sum_within_native_float32_roundoff=float((full.double()-sum64).abs().max()) <= tolerance)
    if reference is not None:
        checks["complete_saved_prediction_exact"] = all(
            full_summary[field] == reference[field]
            for field in ("q_heads", "q_min", "q_mean", "min_head", "predicted_choice"))
        checks["saved_executed_actions_exact"] = actions[:, 0].tolist() == reference["executed_actions"]
    if not all(checks.values()):
        raise RuntimeError("Frozen critic decomposition failed: "+str(
            [name for name, passed in checks.items() if not passed]))
    return dict(critic_name=critic_name, variants=variants,
        components={name: value.tolist() for name, value in components.items()},
        component_delta_heads={name: (value[:, 1]-value[:, 0]).tolist()
                               for name, value in components.items()},
        shared_level_algebraic_delta_heads=[0.]*len(full),
        same_weight_direct_effect=dict(
            level_on_min_advantage=full_summary["delta_min"]-variants["no_level"]["delta_min"],
            interaction_on_min_advantage=full_summary["delta_min"]-variants["no_interaction"]["delta_min"],
            both_on_min_advantage=full_summary["delta_min"]-variants["base"]["delta_min"]),
        numeric=dict(native_dtype=str(full.dtype), native_float32_forward_max_abs_error=closure32,
            float64_component_sum_vs_native_max_abs_error=float((full.double()-sum64).abs().max()),
            float64_contrast_sum_vs_native_max_abs_error=float((native_contrast64-contrast64).abs().max()),
            shared_level_native_action_contrast_max_roundoff=shared_roundoff,
            native_float32_roundoff_bound=tolerance,
            addition_order="base + (shared_level + action_interaction)"),
        executed_actions=actions[:, 0].tolist(), alias_groups=aliases,
        checks=checks, integrity=dict(all_pass=True))


def compare_frozen(one_step, trace):
    """Separate trace-minus-single-step levels, slopes and min-head selection.

    Arithmetic uses the saved native float32 values promoted to float64. The
    original native contrast remains separately reported in each arm. This is
    learned-function attribution, not an attribution to individual updates.
    """
    if one_step["executed_actions"] != trace["executed_actions"]:
        raise ValueError("Both frozen models must score identical executed actions")
    matrices = {}
    for name in ("base", "shared_level", "action_interaction", "full"):
        def matrix(value):
            raw = value["variants"]["full"]["q_heads"] if name == "full" else value["components"][name]
            return torch.tensor(raw, dtype=torch.float64)
        left, right = matrix(one_step), matrix(trace)
        change = right-left
        matrices[name] = dict(head_level_change=change.tolist(),
            head_common_level_shift=change.mean(1).tolist(),
            head_action_contrast_increment=(change[:, 1]-change[:, 0]).tolist())
    left = torch.tensor(one_step["variants"]["full"]["q_heads"], dtype=torch.float64)
    right = torch.tensor(trace["variants"]["full"]["q_heads"], dtype=torch.float64)
    head = one_step["variants"]["full"]["min_head"][0]
    left_fixed, right_fixed = float(left[head, 1]-left[head, 0]), float(right[head, 1]-right[head, 0])
    left_min, right_min = float(left.min(0).values.diff()[0]), float(right.min(0).values.diff()[0])
    fixed_increment = right_fixed-left_fixed
    selection_increment = (right_min-right_fixed)-(left_min-left_fixed)
    full_increment = right_min-left_min
    anchored = {name: values["head_action_contrast_increment"][head]
                for name, values in matrices.items()}
    return dict(parts=matrices,
        min_advantage=dict(one_step=left_min, trace=right_min,
            increment=full_increment, one_step_low_action_reference_head=head,
            fixed_reference_head_contrast_increment=fixed_increment,
            min_head_selection_gap_increment=selection_increment,
            decomposition_error=full_increment-fixed_increment-selection_increment,
            one_step_min_heads=one_step["variants"]["full"]["min_head"],
            trace_min_heads=trace["variants"]["full"]["min_head"]),
        reference_head_parts=anchored,
        reference_head_component_sum_error=anchored["full"]-sum(
            anchored[name] for name in ("base", "shared_level", "action_interaction")),
        interpretation="Shared level has no direct action contrast. Base/interactions reflect learned parameter changes; their training causes are not identified by deletions.")
