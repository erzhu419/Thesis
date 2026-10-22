"""Common-draw eight-step policy forks isolating initialized world weights."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MethodType

import numpy as np
import torch

from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    FrozenBellmanEvaluator, _action_record, _candidate_pair, _delta,
    _learner_state as _v5_learner_state, _paired, _run_packet, _same,
    restore_rng, rng_state, transition_batch,
)
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import parameter_vector


BRANCHES = ("Online", "Frozen")


def _learner_state(learner):
    """Control-training state; passive paired logs/twin trust do not drive it."""
    state = _v5_learner_state(learner)
    state.update(world_norm=learner.world.norm.state_dict(), world_family=learner.world.family)
    return state


@dataclass
class _PacketToken:
    before: object
    rng: dict
    states: torch.Tensor
    deadline: float
    now: float
    batches: list
    original_generate: object


class PacketShadowProbe:
    """Two memory-only eight-update forks with the actual per-row trust arrays.

    Both branches retain the live selected model family. Only Frozen replaces
    neural weights and their initial normalizer. It still trains its copied
    trust critic, but those branch weights never enter the policy fork loss.
    """
    def __init__(self, action_transform=None, deadlines=(8400, 10200, 13800, 17100)):
        self.action_transform = action_transform
        self.deadlines = tuple(float(value) for value in deadlines)
        self.anchors, self.next_packet, self._pending = [], [], []

    def begin_packet(self, learner, packet, deadline, now):
        if float(deadline) not in self.deadlines:
            return None
        if learner.world.local_enabled:
            raise ValueError("Paired-world shadows require the action_model baseline")
        before_rng = rng_state()
        before = copy.deepcopy(learner)
        # Twin diagnostics do not affect active training. Avoid copying logs
        # into new records or doing a second unrelated paired-trust fit here.
        before.world.diagnostic_enabled = False
        states = torch.as_tensor(np.stack([row.s for row in packet]),
                                 dtype=torch.float32, device=learner.device)
        token = _PacketToken(before, before_rng, states, float(deadline), float(now), [],
                             learner.world.generate)

        def capture(world, actor, size):
            decoder_before = world.generator.get_state().clone()
            generated = token.original_generate(actor, size)
            if generated is not None:
                batch, weights = generated
                token.batches.append(dict(weights=weights.detach().clone(),
                    states=batch["observations"].detach().clone(),
                    decoder_before=decoder_before, decoder_after=world.generator.get_state().clone()))
            return generated

        learner.world.generate = MethodType(capture, learner.world)
        return token

    def finish_packet(self, learner, token):
        if token is None:
            return None
        learner.world.generate = token.original_generate
        after_rng = rng_state()
        try:
            if learner.total_updates-token.before.total_updates != 8 or len(token.batches) != 8:
                raise RuntimeError("Shadow anchor must cover exactly eight live model updates")
            actual_state = _learner_state(learner)
            candidates = _candidate_pair(token.states, learner.action_transform)
            initial_actions = _action_record(token.before, token.states, candidates)
            before_vectors = {name: parameter_vector(getattr(token.before, name)).clone()
                              for name in ("actor", "critic")}
            weights = [batch["weights"] for batch in token.batches]
            record = dict(deadline=token.deadline, processed_at=token.now, packet_rows=len(token.states),
                steps=8, active_model_batches=len(weights), trust_weights_fixed_to_active=True,
                trust_weight_rows=sum(w.numel() for w in weights),
                active_trust_mean=float(torch.cat(weights).mean()), online_exact_active=False,
                common_family=token.before.world.family,
                frozen_own_selected_family=token.before.world.frozen_family,
                initialized_world_weights_only=True, local_readout_enabled=False,
                common_state_rows_matched=True, common_rng_streams_matched=True, branches={})
            held = dict(anchor_deadline=token.deadline, evaluator=FrozenBellmanEvaluator(token.before),
                        critics={})
            branch_values, branch_vectors = {}, {}
            for branch in BRANCHES:
                shadow = copy.deepcopy(token.before)
                if branch == "Frozen":
                    shadow.world.models = copy.deepcopy(shadow.world.frozen_models)
                    shadow.world.norm = copy.deepcopy(shadow.world.frozen_norm)
                original_generate = shadow.world.generate
                weight_index = [0]

                def fixed(world, actor, size):
                    index = weight_index[0]
                    expected = token.batches[index]
                    if not torch.equal(world.generator.get_state(), expected["decoder_before"]):
                        raise RuntimeError("A shadow entered the decoder with a different draw stream")
                    generated = original_generate(actor, size)
                    if generated is None:
                        raise RuntimeError("A shadow generated no batch while the live batch existed")
                    if not torch.equal(generated[0]["observations"], expected["states"]):
                        raise RuntimeError("A shadow sampled different world starting states")
                    if not torch.equal(world.generator.get_state(), expected["decoder_after"]):
                        raise RuntimeError("A shadow consumed different decoder random draws")
                    weight_index[0] += 1
                    return generated[0], expected["weights"].to(world.device)

                shadow.world.generate = MethodType(fixed, shadow.world)
                restore_rng(token.rng)
                _run_packet(shadow)
                branch_rng = rng_state()
                shadow.world.generate = original_generate
                if weight_index[0] != 8:
                    raise RuntimeError("A shadow did not consume all eight fixed trust arrays")
                if not (_same(shadow.rng.bit_generator.state, learner.rng.bit_generator.state)
                        and _same(shadow.world.rng.bit_generator.state, learner.world.rng.bit_generator.state)
                        and _same(branch_rng, after_rng)):
                    raise RuntimeError("A shadow did not retain the active replay and global draw streams")
                if branch == "Online":
                    exact = _same(_learner_state(shadow), actual_state)
                    record["online_exact_active"] = bool(exact)
                    if not exact:
                        raise RuntimeError("Eight-step Online shadow diverged from the live learner")
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
            online = branch_values["Online"]
            for branch, values in branch_values.items():
                result = record["branches"][branch]
                for module in ("actor", "critic"):
                    result[module+"_parameter_difference_versus_online"] = _delta(
                        branch_vectors[branch][module], branch_vectors["Online"][module])
                result["proposal_versus_online"] = _paired(values["proposal"], online["proposal"])
                result["executed_versus_online"] = _paired(values["executed"], online["executed"])
                valid = values["valid"]
                result["q_contrast_versus_online"] = (
                    _paired(values["contrast"][valid], online["contrast"][valid]) if valid.any() else None)
                result["q_ranking_flip_versus_online_fraction"] = (float(
                    ((values["contrast"][valid] > 0) != (online["contrast"][valid] > 0))
                    .float().mean()) if valid.any() else None)
            record["semantics"] = ("Eight actual optimizer steps from identical learner/Adam/replay/RNG. "
                "Both forks use the same live selected family, start states, random draws and live per-row "
                "trust weights; Frozen substitutes initially fitted neural weights. Actors evolve separately. "
                "Effects are evaluated on fixed released packet states without a target environment fork.")
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
                residual = critic(batch["observations"], batch["actions"])-target[None]
                branches[name] = dict(mean_absolute_td_error=float(residual.abs().mean()),
                    mean_squared_td_error=float(residual.square().mean()),
                    mean_signed_td_error=float(residual.mean()))
            records.append(dict(anchor_deadline=held["anchor_deadline"], scored_at=float(now),
                rows=len(packet), common_frozen_anchor_target=True,
                anchor_update=evaluator.anchor_update, branches=branches,
                semantics="First subsequent released packet before learning; all fork critics use one "
                    "pre-update anchor Bellman target and the same real outcomes/actions. These conditional "
                    "factual TD errors are not counterfactual policy returns."))
        self._pending.clear()
        self.next_packet.extend(records)
        return records

    def summary(self):
        return dict(anchors=self.anchors, next_packet=self.next_packet,
                    pending_anchor_count=len(self._pending),
                    future_evaluation="one common frozen pre-update value standard; future data not trained yet",
                    target_environment_forks=0)
