"""One critic change: share the memory level across all ensemble heads.

The head-specific action interaction, native base, replay, actor/value inputs,
Bellman target and update schedule remain the v33 implementation. A shared
level shifts every head equally at one state; it cannot directly re-rank heads
or alter same-state action advantages except native floating-point roundoff.
"""
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024 import learning as v33
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng


ARMS = ("zero_memory", "per_head_level", "shared_level")
MEMORY_DIM, STATE_DIM = v33.MEMORY_DIM, v33.STATE_DIM
critic_states = v33.critic_states


class SharedLevelQEnsemble(v33.MemoryQEnsemble):
    def __init__(self, native):
        super().__init__(native)
        self.level_weight = torch.nn.Parameter(torch.zeros_like(self.level_weight[:1]))

    def _residual(self, actions, memory, indices=None):
        if memory.ndim != 2 or memory.shape != (len(actions), MEMORY_DIM):
            raise ValueError("Each executed action needs its own84-field context")
        interaction = self.action_weight
        if indices is not None:
            interaction = interaction.index_select(0, indices)
        level = (memory @ self.level_weight.T).T
        action = ((actions[:, 0]+1.)*.5)[None, :]*(memory @ interaction.T).T
        return level+action


def attach_shared_level(learner):
    """Attach before feedback; keep the original critic's parameter/Adam objects."""
    if learner.replay_buffer.arm != "fresh" or learner.replay_buffer.recent:
        raise ValueError("Attach shared level before admitting factual feedback")
    if learner.config.online_cql_alpha != 0 or learner.world is not None:
        raise ValueError("v36 retains the resident real-only non-CQL objective")
    rng_before = _rng()
    actor, value, transform, optimizer = learner.actor, learner.value, learner.action_transform, learner.critic_optimizer
    old_parameters = tuple(learner.critic.parameters())
    old_adam = dict(optimizer.state)
    learner.__class__ = v33.MemoryOutcomeTrainer
    learner.replay_buffer.__class__ = v33.MemoryOutcomeReplay
    learner.critic = SharedLevelQEnsemble(learner.critic)
    learner.target_critic = SharedLevelQEnsemble(learner.target_critic)
    learner.target_critic.eval()
    for parameter in learner.target_critic.parameters():
        parameter.requires_grad_(False)
    optimizer.param_groups[0]["params"].extend((learner.critic.level_weight, learner.critic.action_weight))
    learner.shared_level_attachment = dict(
        native_parameter_objects_retained=all(left is right for left, right in
            zip(old_parameters, learner.critic.base.parameters())),
        old_adam_entries_retained=all(optimizer.state[key] is state for key, state in old_adam.items()),
        residual_adam_starts_empty=all(parameter not in optimizer.state for parameter in
            (learner.critic.level_weight, learner.critic.action_weight)),
        actor_value_F_objects_unchanged=learner.actor is actor and learner.value is value and learner.action_transform is transform,
        optimizer_object_unchanged=learner.critic_optimizer is optimizer,
        constructor_rng_unchanged=_equal(rng_before, _rng()))
    if not all(learner.shared_level_attachment.values()):
        raise RuntimeError("Shared-level attachment changed the matched initialization")
    return learner


def make_learner(seed, *, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    original_arm = "memory" if arm == "per_head_level" else "zero_memory"
    learner, replay, contract, stations, initializer = v33.make_learner(seed, arm=original_arm)
    if arm == "shared_level":
        attach_shared_level(learner)
    learner.memory_arm = arm
    return learner, replay, contract, stations, initializer


@torch.no_grad()
def initial_preflight(learner):
    """Probe already loaded real archived rows before the first feedback packet."""
    replay = learner.replay_buffer
    saved_rng = _rng()
    if replay.target_received or replay.recent:
        raise ValueError("Zero-initialization preflight must precede feedback")
    states = torch.cat(tuple(torch.as_tensor(table["s"][:3], dtype=torch.float32, device=learner.device)
                             for table in (replay.history, replay.source)))
    actions = torch.cat(tuple(torch.as_tensor(table["a"][:3], dtype=torch.float32, device=learner.device).reshape(-1, 1)
                              for table in (replay.history, replay.source)))
    context = torch.arange(MEMORY_DIM, dtype=states.dtype, device=learner.device)[None, :].expand(len(states), -1)/MEMORY_DIM
    actor_before = learner.actor(states, deterministic=True)[0]
    value_before = learner.value(states)
    checks = dict(fresh_feedback_empty=True)
    for name in ("critic", "target_critic"):
        critic = getattr(learner, name)
        is_memory = isinstance(critic, v33.MemoryQEnsemble)
        base = critic.base if is_memory else critic
        explicit = critic_states(learner, states, context)
        indices = torch.tensor([0, base.ensemble_size-1], device=learner.device)
        expected = base(states, actions)
        expected_subset = base.forward_subset(states, actions, indices)
        features = critic.encode_states(explicit)
        native_features = base.encode_states(states)
        checks[name+"_full_exact"] = torch.equal(critic(explicit, actions), expected)
        checks[name+"_subset_exact"] = torch.equal(critic.forward_subset(explicit, actions, indices), expected_subset)
        checks[name+"_encoder_exact"] = torch.equal(features[:, :native_features.shape[1]], native_features)
        checks[name+"_feature_forward_exact"] = torch.equal(critic.forward_features(features, actions), expected)
        checks[name+"_feature_subset_exact"] = torch.equal(
            critic.forward_features_subset(features, actions, indices), expected_subset)
        with torch.enable_grad():
            action_input = actions.detach().clone().requires_grad_(True)
            native_gradient = torch.autograd.grad(base(states, action_input).sum(), action_input)[0]
            full_gradient = torch.autograd.grad(critic(explicit, action_input).sum(), action_input)[0]
            subset_gradient = torch.autograd.grad(
                critic.forward_subset(explicit, action_input, indices).sum(), action_input)[0]
            native_subset_gradient = torch.autograd.grad(
                base.forward_subset(states, action_input, indices).sum(), action_input)[0]
        checks[name+"_action_gradient_exact"] = torch.equal(full_gradient, native_gradient)
        checks[name+"_subset_action_gradient_exact"] = torch.equal(subset_gradient, native_subset_gradient)
        checks[name+"_residual_zero"] = not is_memory or (
            torch.count_nonzero(critic.level_weight).item() == torch.count_nonzero(critic.action_weight).item() == 0)
    checks["actor_value_unchanged"] = (torch.equal(actor_before, learner.actor(states, deterministic=True)[0])
                                          and torch.equal(value_before, learner.value(states)))
    checks["rng_unchanged"] = _equal(saved_rng, _rng())
    checks.update(getattr(learner, "shared_level_attachment", {}))
    proof = dict(all_pass=all(checks.values()), checks=checks, rows=len(states),
        level_shape=list(learner.critic.level_weight.shape) if isinstance(learner.critic, v33.MemoryQEnsemble) else None,
        action_shape=list(learner.critic.action_weight.shape) if isinstance(learner.critic, v33.MemoryQEnsemble) else None,
        archived_rows_only=True, new_training_labels=0, optimizer_updates=0)
    if not proof["all_pass"]:
        raise RuntimeError("Initial native parity failed: "+str([key for key, value in checks.items() if not value]))
    return proof
