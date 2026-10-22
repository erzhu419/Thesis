"""Memory-only, common-draw eight-update forks of one released feedback packet."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from types import MethodType

import numpy as np
import torch

from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import (
    TransitH2OTrainerV7, parameter_vector,
)
from experiments.mcwm_closed_loop_v1.learning import row_table


BRANCHES = ("Base", "Local", "Local-W", "Local-C")


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def transition_batch(packet, device):
    table = row_table(packet)
    keys = dict(observations="s", actions="a", next_observations="ns", rewards="r",
                delta_t_seconds="dt", terminals="terminal")
    batch = {key: torch.as_tensor(table[value], dtype=torch.float32, device=device)
             for key, value in keys.items()}
    batch["actions"] = batch["actions"].reshape(-1, 1)
    for key in ("rewards", "delta_t_seconds", "terminals"):
        batch[key] = batch[key].reshape(-1)
    batch["discounts"] = torch.ones(len(packet), device=device)
    return batch


class FrozenBellmanEvaluator:
    """One common anchor value standard; every call uses identical target draws.

    Only target-relevant networks/configuration are copied.  Optimizers, replay
    and world are deliberately absent: this evaluator never learns.
    """
    def __init__(self, learner, seed=None):
        self.learner = object.__new__(type(learner))
        for name in ("actor", "critic", "target_critic", "value", "config", "log_alpha",
                     "action_transform", "device"):
            setattr(self.learner, name, copy.deepcopy(getattr(learner, name)))
        self.seed = int(seed if seed is not None else learner.world.seed+90000+learner.total_updates)
        self.anchor_update = int(learner.total_updates)

    @torch.no_grad()
    def target(self, batch):
        before = rng_state()
        try:
            torch.manual_seed(self.seed)
            return TransitH2OTrainerV7._target(self.learner, batch, backup="h2oplus").detach().clone()
        finally:
            restore_rng(before)


def _delta(after, before):
    value = after.detach()-before.detach()
    return dict(l2=float(value.norm()), max_absolute=float(value.abs().max()),
                relative_l2=float(value.norm()/before.norm()) if before.norm() > 0 else None)


def _paired(after, before):
    value = after.detach()-before.detach()
    return dict(mean_delta=float(value.mean()), mean_absolute_delta=float(value.abs().mean()),
                max_absolute_delta=float(value.abs().max()))


def _learner_state(learner):
    """Actual mutable training state needed by the Local reproduction gate."""
    state = {name: getattr(learner, name).state_dict() for name in (
        "actor", "critic", "target_critic", "value", "actor_optimizer", "critic_optimizer",
        "value_optimizer", "alpha_optimizer")}
    state.update(log_alpha=learner.log_alpha, updates=learner.total_updates,
        online_updates=learner.online_critic_updates, rng=learner.rng.bit_generator.state,
        diagnostic_rng=learner.diagnostic_rng.bit_generator.state,
        world_rng=learner.world.rng.bit_generator.state,
        decoder_rng=learner.world.generator.get_state(),
        world_models={name: model.state_dict() for name, model in learner.world.models.items()},
        trust_net=learner.world.trust.net.state_dict(),
        trust_target=learner.world.trust.target.state_dict(),
        trust_optimizer=learner.world.trust.optimizer.state_dict(), trust_tau=learner.world.trust._tau)
    if hasattr(learner, "ratio"):
        state["ratio"] = learner.ratio.state_dict()
        state["ratio_optimizer"] = learner.ratio.optimizer.state_dict()
    return state


def _clear_counterfactual(learner):
    learner.world.capture_counterfactual = False
    for name in ("counterfactual_batch", "head_counterfactual_batches",
                 "head_counterfactual_weights", "head_counterfactual_outputs"):
        setattr(learner.world, name, None)


def _run_packet(learner):
    for step in range(8):
        learner.world.capture_counterfactual = step == 0
        learner.local_counterfactual = learner.head_counterfactual = None
        learner.continue_step(collect_diagnostics=step == 0)
        _clear_counterfactual(learner)


def _configure_branch(learner, branch):
    world = learner.world
    if branch == "Base":
        world.local_enabled = False
    elif branch != "Local":
        original = world.predict
        head = {"Local-W": 2, "Local-C": 1}[branch]
        def removed(self, s, a, family=None, calibrated=None):
            output = original(s, a, family=family, calibrated=calibrated)
            if calibrated is not False:
                base = original(s, a, family=family, calibrated=False)
                output[:, head] = base[:, head]
            return output
        world.predict = MethodType(removed, world)


@torch.no_grad()
def _candidate_pair(states, transform):
    low_u = states.new_full((len(states), 1), -1.)
    high_u = states.new_full((len(states), 1), 1.)
    low, high = transform(states, low_u), transform(states, high_u)
    separation = (transform.hold_seconds(high)-transform.hold_seconds(low)).reshape(-1)
    return low, high, separation > 1e-5, separation


def _action_record(learner, states, candidates):
    # Diagnostics have an independent fixed draw and restore all global RNGs.
    before = rng_state()
    try:
        torch.manual_seed(99173)
        with torch.no_grad():
            u = learner.actor(states, deterministic=True)[0]
            a = learner.action_transform(states, u)
            components = learner.actor.soft_action_components(states)
            positive = components["positive_action"].detach()
            zero = learner.action_transform(states, components["zero_action"])
            low, high, valid, separation = candidates
            # Apply the same reduction used by the live actor objective.
            # Both candidate actions are physical F outputs.
            q = learner.critic(torch.cat((states, states)), torch.cat((low, high)))
            low_q, high_q = learner._reduce_actor_q(q).split(len(states))
            contrast = high_q-low_q
        with torch.enable_grad():
            proposal = positive.clone().requires_grad_(True)
            executed = learner.action_transform(states, proposal)
            derivative = torch.autograd.grad(executed.sum(), proposal)[0].reshape(-1)
        h = learner.action_transform.hold_seconds
        gap = (h(executed.detach())-h(zero)).abs().reshape(-1)
        geometry = dict(rows=len(states), positive_amplitude_derivative_mean=float(derivative.mean()),
            zero_amplitude_derivative_fraction=float((derivative.abs() <= 1e-8).float().mean()),
            zero_positive_gate_merged_fraction=float((gap <= 1e-5).float().mean()),
            mean_positive_gate_executed_gap_seconds=float(gap.mean()))
        qcontrast = dict(rows=len(states), valid_rows=int(valid.sum()),
            no_distinguishable_executed_hold_rows=int((~valid).sum()),
            mean_executed_hold_gap_seconds=float(separation[valid].mean()) if valid.any() else None,
            mean_q_high_minus_low=float(contrast[valid].mean()) if valid.any() else None,
            mean_absolute_q_high_minus_low=float(contrast[valid].abs().mean()) if valid.any() else None,
            fraction_prefer_high=float((contrast[valid] > 0).float().mean()) if valid.any() else None,
            reduction=learner.config.actor_q_reduction,
            candidates="high/low are F(s,+1)/F(s,-1); common actor Q reduction")
        return dict(proposal=h(u).detach(), executed=h(a).detach(),
                    contrast=contrast.detach(), valid=valid, geometry=geometry, qcontrast=qcontrast)
    finally:
        restore_rng(before)


@dataclass
class _PacketToken:
    before: object
    rng: dict
    states: torch.Tensor
    deadline: float
    now: float
    weights: list
    original_generate: object


class PacketShadowProbe:
    """Fork Base/Local/Local-W/Local-C only at four prespecified packets."""
    def __init__(self, action_transform=None, deadlines=(8400, 10200, 13800, 17100)):
        self.action_transform = action_transform
        self.deadlines = tuple(float(value) for value in deadlines)
        self.anchors, self.next_packet, self._pending = [], [], []

    def begin_packet(self, learner, packet, deadline, now):
        if float(deadline) not in self.deadlines:
            return None
        if not learner.world.local_enabled:
            raise ValueError("Eight-step Local reproduction requires active local calibration")
        before_rng = rng_state()
        # At this boundary no autograd graph is live: continue_step cleared its
        # model loss; all modules/replay/optimizers and independent RNGs fork.
        before = copy.deepcopy(learner)
        states = torch.as_tensor(np.stack([row.s for row in packet]),
                                 dtype=torch.float32, device=learner.device)
        token = _PacketToken(before, before_rng, states, float(deadline), float(now), [],
                             learner.world.generate)
        def capture(world, actor, size):
            generated = token.original_generate(actor, size)
            if generated is not None:
                token.weights.append(generated[1].detach().clone())
            return generated
        learner.world.generate = MethodType(capture, learner.world)
        return token

    def finish_packet(self, learner, token):
        if token is None:
            return None
        learner.world.generate = token.original_generate
        after_rng = rng_state()
        try:
            if learner.total_updates-token.before.total_updates != 8 or len(token.weights) != 8:
                raise RuntimeError("Shadow anchor must cover exactly eight live model updates")
            actual_state = _learner_state(learner)
            candidates = _candidate_pair(token.states, learner.action_transform)
            initial_actions = _action_record(token.before, token.states, candidates)
            before_vectors = {name: parameter_vector(getattr(token.before, name)).clone()
                              for name in ("actor", "critic")}
            evaluator = FrozenBellmanEvaluator(token.before)
            record = dict(deadline=token.deadline, processed_at=token.now, packet_rows=len(token.states),
                steps=8, active_model_batches=len(token.weights),
                trust_weights_fixed_to_active=True, trust_weight_rows=sum(w.numel() for w in token.weights),
                active_trust_mean=float(torch.cat(token.weights).mean()), local_exact_active=False,
                branches={})
            held = dict(anchor_deadline=token.deadline, evaluator=evaluator, critics={})
            branch_values, branch_vectors = {}, {}
            for branch in BRANCHES:
                shadow = copy.deepcopy(token.before)
                _configure_branch(shadow, branch)
                original_generate = shadow.world.generate
                weight_index = [0]
                def fixed(world, actor, size):
                    generated = original_generate(actor, size)
                    if generated is None:
                        raise RuntimeError("A shadow generated no batch while the live batch existed")
                    index = weight_index[0]
                    weight_index[0] += 1
                    return generated[0], token.weights[index].to(world.device)
                shadow.world.generate = MethodType(fixed, shadow.world)
                restore_rng(token.rng)
                _run_packet(shadow)
                branch_rng = rng_state()
                shadow.world.generate = original_generate
                if weight_index[0] != 8:
                    raise RuntimeError("A shadow did not consume all eight fixed trust arrays")
                if branch == "Local":
                    exact = _same(_learner_state(shadow), actual_state) and _same(branch_rng, after_rng)
                    record["local_exact_active"] = bool(exact)
                    if not exact:
                        raise RuntimeError("Eight-step Local shadow diverged from the live learner")
                values = _action_record(shadow, token.states, candidates)
                branch_values[branch] = values
                branch_vectors[branch] = {name: parameter_vector(getattr(shadow, name)).clone()
                                          for name in ("actor", "critic")}
                result = {name+"_parameter_change": _delta(parameter_vector(getattr(shadow, name)),
                            before_vectors[name]) for name in ("actor", "critic")}
                result.update(proposal_versus_before=_paired(values["proposal"], initial_actions["proposal"]),
                    executed_versus_before=_paired(values["executed"], initial_actions["executed"]),
                    action_geometry=values["geometry"], candidate_q_contrast=values["qcontrast"])
                valid = values["valid"]
                result["q_contrast_change_versus_before"] = (
                    _paired(values["contrast"][valid], initial_actions["contrast"][valid])
                    if valid.any() else None)
                result["q_ranking_flip_versus_before_fraction"] = (float(
                    ((values["contrast"][valid] > 0) != (initial_actions["contrast"][valid] > 0))
                    .float().mean()) if valid.any() else None)
                record["branches"][branch] = result
                held["critics"][branch] = copy.deepcopy(shadow.critic).eval()
                del shadow
            local = branch_values["Local"]
            for branch, values in branch_values.items():
                result = record["branches"][branch]
                for module in ("actor", "critic"):
                    result[module+"_parameter_difference_versus_local"] = _delta(
                        branch_vectors[branch][module], branch_vectors["Local"][module])
                result["proposal_versus_local"] = _paired(values["proposal"], local["proposal"])
                result["executed_versus_local"] = _paired(values["executed"], local["executed"])
                valid = values["valid"]
                result["q_contrast_versus_local"] = (_paired(values["contrast"][valid], local["contrast"][valid])
                                                       if valid.any() else None)
            record["semantics"] = ("Eight actual optimizer steps from identical full learner/optimizer/replay/RNG; "
                "branches use their own evolving actors, common draws and live per-row trust weights. "
                "Action/value effects are on fixed released packet states, without a target environment fork.")
            self.anchors.append(record)
            self._pending.append(held)
            return record
        finally:
            restore_rng(after_rng)

    @torch.no_grad()
    def resolve_next_packet(self, packet, now):
        if not self._pending:
            return []
        if any(row.available_at > now or row.completed_at > now for row in packet):
            raise ValueError("Future packet evaluated before its outcomes were released")
        records = []
        for held in self._pending:
            evaluator = held["evaluator"]
            batch = transition_batch(packet, evaluator.learner.device)
            target = evaluator.target(batch)
            branches = {}
            for name, critic in held["critics"].items():
                q = critic(batch["observations"], batch["actions"])
                residual = q-target[None]
                branches[name] = dict(mean_absolute_td_error=float(residual.abs().mean()),
                    mean_squared_td_error=float(residual.square().mean()),
                    mean_signed_td_error=float(residual.mean()))
            records.append(dict(anchor_deadline=held["anchor_deadline"], scored_at=float(now),
                rows=len(packet), common_frozen_anchor_target=True,
                anchor_update=evaluator.anchor_update, branches=branches,
                semantics="First subsequent released packet, before world/refit/policy learning. "
                    "All fork critics use one pre-update anchor Bellman target and the same actual outcomes/actions; "
                    "these are conditional factual TD errors, not counterfactual policy returns."))
        self._pending.clear()
        self.next_packet.extend(records)
        return records

    def summary(self):
        return dict(anchors=self.anchors, next_packet=self.next_packet,
                    pending_anchor_count=len(self._pending),
                    future_evaluation="one common frozen pre-update value standard; future real data not trained yet",
                    target_environment_forks=0)
