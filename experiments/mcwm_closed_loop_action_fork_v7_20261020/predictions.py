"""Saved pre-intervention predictions and one common value standard per fork.

Candidates are executable F(s, -1/+1) actions.  Their global next event may
belong to a different vehicle, or occur at zero elapsed time; neither is
replaced with a same-vehicle transition.  No diagnostic outcome is trained on.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    FrozenBellmanEvaluator, restore_rng, rng_state, transition_batch,
)
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.world import (
    decoder_global_projection,
)
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation, losses, row_table, targets


GLOBAL_FIELDS = ("duration_seconds", "cost_passenger_seconds", "waiting_increment", "onboard_increment")
BRANCHES = ("online", "frozen_common")
CANDIDATES = ("low", "high")


def direction(value):
    """A positive gain always means that the larger executed hold is better."""
    return "high" if value > 0 else "low" if value < 0 else "tie"


def paired_effect(low, high):
    """Keep raw high-minus-low effects separate from maximized gains."""
    effect = dict(global_decoded=(np.asarray(high["global_decoded"])
                                  -np.asarray(low["global_decoded"])).tolist(),
                  bellman={key: high["bellman"][key]-low["bellman"][key]
                           for key in ("scaled_reward", "bootstrap", "target")})
    gains = dict(next_event_cost=-effect["global_decoded"][1],
                 bellman_target=effect["bellman"]["target"],
                 scaled_reward=effect["bellman"]["scaled_reward"],
                 bootstrap=effect["bellman"]["bootstrap"])
    return dict(high_minus_low=effect, gains=gains,
                directions={key: direction(value) for key, value in gains.items()})


def compare_records(prediction, low, high, *, low_cost_300, high_cost_300):
    """Join small JSON branch results; no models or checkpoints are required."""
    if low["candidate"] != "low" or high["candidate"] != "high":
        raise ValueError("Factual records must retain low/high branch order")
    for key in ("anchor_update", "world_version", "common_family", "target_seed"):
        if low[key] != prediction[key] or high[key] != prediction[key]:
            raise ValueError(f"Fork branches do not share the prediction anchor: {key}")
    actual = paired_effect(low, high)
    gain300 = float(low_cost_300)-float(high_cost_300)
    actual["gains"]["cost_300_seconds"] = gain300
    actual["directions"]["cost_300_seconds"] = direction(gain300)
    actual["high_minus_low"]["cost_300_seconds"] = -gain300
    comparison = {}
    for branch in BRANCHES:
        predicted = prediction["branches"][branch]
        comparison[branch] = dict(
            gain_signed_error={key: predicted["gains"][key]-actual["gains"][key]
                               for key in predicted["gains"]},
            next_event_cost_direction_matches=(predicted["directions"]["next_event_cost"]
                ==actual["directions"]["next_event_cost"]),
            bellman_direction_matches=(predicted["directions"]["bellman_target"]
                ==actual["directions"]["bellman_target"]),
            bellman_direction_matches_cost_300=(predicted["directions"]["bellman_target"]
                ==actual["directions"]["cost_300_seconds"]))
    return dict(valid_pair=prediction["valid_pair"], factual=actual, predicted_versus_factual=comparison,
        actual_branches_have_same_next_vehicle=(low["next_vehicle_id"]==high["next_vehicle_id"]),
        next_global_event_time_difference_seconds=high["completed_at"]-low["completed_at"],
        semantics="The two next-global-event endpoints may differ. The 300-second passenger ledger uses the common intervention-clock window; it is a separate control outcome, not a one-step model label.")


def _topology(state, stations):
    if not np.any(state):
        return dict(event_count=0, current_slot=None, ordered_line_station=[])
    count = int(round(float(state[1])*16))
    current = int(round(float(state[2])*15))
    slots = np.asarray(state[5:]).reshape(16, 14)
    events = []
    for slot in slots[:count]:
        line = int(round(float(slot[1])))
        events.append([line, int(round(float(slot[2])*float(stations[line])))])
    return dict(event_count=count, current_slot=current, ordered_line_station=events)


class ActionForkPrediction:
    """Capture predictions before intervention and score both actual branches later."""
    def __init__(self, learner, state, *, seed):
        before = rng_state()
        try:
            self.device = torch.device(learner.device)
            self.state = torch.as_tensor(np.asarray(state).reshape(1, 229),
                                         dtype=torch.float32, device=self.device).clone()
            world = learner.world
            self.norm = copy.deepcopy(world.norm)
            self.stations = world.stations.detach().clone()
            self.family = world.family
            self.world_version = int(world.version)
            self.evaluator = FrozenBellmanEvaluator(learner, seed=int(seed)+1)
            self.trusts = dict(online=copy.deepcopy(world.trust),
                               frozen_common=copy.deepcopy(world.twin_trust))
            self.action_transform = copy.deepcopy(learner.action_transform)
            self.actions = {name: self.action_transform(self.state, self.state.new_tensor([[u]]))
                            for name, u in (("low", -1.), ("high", 1.))}
            self.candidates = {name: dict(executed_action=float(action.item()),
                holding_seconds=float(self.action_transform.hold_seconds(action).item()))
                for name, action in self.actions.items()}
            gap = self.candidates["high"]["holding_seconds"]-self.candidates["low"]["holding_seconds"]
            self.valid_pair = bool(gap > 1e-5)
            self.outputs, branches = {}, {}
            decoder = SimpleNamespace(norm=self.norm, stations=self.stations, device=self.device,
                                      generator=torch.Generator(device=self.device))
            with torch.no_grad():
                for branch, models in (("online", world.models), ("frozen_common", world.frozen_models)):
                    self.outputs[branch], records = {}, {}
                    for candidate, action in self.actions.items():
                        output = models[self.family](self.state, action).detach().clone()
                        self.outputs[branch][candidate] = output
                        # Each of the four calls starts from the same decoder draw,
                        # including low/high within each model.
                        decoder.generator.manual_seed(int(seed))
                        ns, reward, dt = ModelAugmentation.decode(decoder, self.state, output)
                        batch = dict(observations=self.state, actions=action, next_observations=ns,
                            rewards=reward, delta_t_seconds=dt, discounts=torch.ones(1, device=self.device),
                            terminals=torch.zeros(1, device=self.device))
                        target = float(self.evaluator.target(batch).item())
                        scaled = float(reward.item())*self.evaluator.learner.config.reward_scale
                        global_values = output[0, :4]*self.norm.g_std+self.norm.g_mean
                        raw = global_values.clone()
                        raw[:2] = raw[:2].expm1()
                        decoded, saturated = decoder_global_projection(output[:, :4], self.norm, self.state)
                        trust = self.trusts[branch]
                        records[candidate] = dict(global_raw=raw.cpu().tolist(),
                            global_decoded=decoded.tolist(), standardized_global=output[0, :4].cpu().tolist(),
                            decode_saturated=saturated,
                            decoded_next_topology=_topology(ns[0].cpu().numpy(), self.stations.cpu().numpy()),
                            bellman=dict(scaled_reward=scaled, bootstrap=target-scaled, target=target),
                            trust=dict(weight=float(trust.weight(self.state, action).item()),
                                log_q=float(trust.net(self.state, action).item()), tau=float(trust._tau)))
                    branches[branch] = dict(candidates=records, **paired_effect(records["low"], records["high"]))
                actions = torch.cat([self.actions[name] for name in CANDIDATES])
                q = learner.critic(self.state.expand(2, -1), actions)
                low_q, high_q = learner._reduce_actor_q(q).reshape(-1).cpu().tolist()
                proposal = learner.actor(self.state, deterministic=True)[0]
                executed = self.action_transform(self.state, proposal)
                components = learner.actor.soft_action_components(self.state)
                positive = components["positive_action"].detach()
                zero = self.action_transform(self.state, components["zero_action"])
            with torch.enable_grad():
                positive = positive.clone().requires_grad_(True)
                positive_executed = self.action_transform(self.state, positive)
                derivative = torch.autograd.grad(positive_executed.sum(), positive)[0]
            self.record = dict(state_time_seconds=float(self.state[0, 0])*24000,
                anchor_update=self.evaluator.anchor_update, world_version=self.world_version,
                common_family=self.family, decoder_seed=int(seed), target_seed=int(seed)+1,
                candidate_actions=copy.deepcopy(self.candidates), valid_pair=self.valid_pair,
                executed_hold_gap_seconds=gap, global_fields=list(GLOBAL_FIELDS), branches=branches,
                critic=dict(low=low_q, high=high_q, high_minus_low=high_q-low_q,
                            direction=direction(high_q-low_q), reduction=learner.config.actor_q_reduction),
                policy_geometry=dict(proposed_hold_seconds=float(self.action_transform.hold_seconds(proposal).item()),
                    executed_hold_seconds=float(self.action_transform.hold_seconds(executed).item()),
                    positive_amplitude_derivative=float(derivative.item()),
                    zero_positive_gate_gap_seconds=float((self.action_transform.hold_seconds(positive_executed.detach())
                        -self.action_transform.hold_seconds(zero)).item()),
                    hold_probability=float(components["hold_probability"].item())),
                semantics="Saved predictions before intervention; common live family, decoder draws and frozen Bellman evaluator. Positive gains favor high; costs use low-minus-high, values use high-minus-low.")
        finally:
            restore_rng(before)

    def summary(self):
        return copy.deepcopy(self.record)

    @torch.no_grad()
    def score_transition(self, row, *, now, candidate):
        """Score the next GLOBAL event, including a valid zero-duration event.

        Availability delay governs learning; this separate intervention truth
        may be scored after completion and never enters the active replay.
        """
        if candidate not in CANDIDATES:
            raise ValueError(f"Unknown action candidate: {candidate}")
        if float(row.completed_at) > float(now):
            raise ValueError("Fork outcome scored before completion")
        if not torch.equal(torch.as_tensor(row.s, device=self.device).reshape(1, 229), self.state):
            raise ValueError("Fork outcome did not start from the captured canonical state")
        expected = self.candidates[candidate]["executed_action"]
        if not np.isclose(float(np.asarray(row.a).reshape(-1)[0]), expected, rtol=0, atol=1e-6):
            raise ValueError("Fork outcome did not use its single-transformed executed action")
        before = rng_state()
        try:
            batch = transition_batch([row], self.device)
            actual_target = float(self.evaluator.target(batch).item())
            scaled = float(row.r)*self.evaluator.learner.config.reward_scale
            actual_global = [float(row.dt), -float(row.r),
                             *(((np.asarray(row.ns)-np.asarray(row.s))[3:5])*20000).tolist()]
            y = targets(row_table([row]), self.stations, self.device)
            standardized = ((y["g"]-self.norm.g_mean)/self.norm.g_std)[0].cpu().numpy()
            factual = dict(candidate=candidate, global_decoded=actual_global,
                bellman=dict(scaled_reward=scaled, bootstrap=actual_target-scaled, target=actual_target),
                anchor_update=self.record["anchor_update"], world_version=self.world_version,
                common_family=self.family, target_seed=self.record["target_seed"],
                started_at=float(row.started_at), completed_at=float(row.completed_at),
                scored_at=float(now), terminal=bool(row.terminal), zero_duration=bool(row.dt == 0),
                vehicle_id=str(getattr(row, "vehicle_id", "")),
                next_vehicle_id=str(getattr(row, "next_vehicle_id", "")),
                next_topology=_topology(np.asarray(row.ns), self.stations.cpu().numpy()), branches={},
                fact=dict(dt=float(row.dt), r=float(row.r), terminal=bool(row.terminal),
                    line_id=str(getattr(row, "line_id", "")),
                    next_vehicle_id=str(getattr(row, "next_vehicle_id", "")),
                    sequence=int(row.sequence), started_at=float(row.started_at),
                    completed_at=float(row.completed_at)))
            next_states = batch["next_observations"]
            proposal = self.evaluator.learner.actor(next_states, deterministic=True)[0]
            next_action = self.action_transform(next_states, proposal)
            for branch in BRANCHES:
                predicted = self.record["branches"][branch]["candidates"][candidate]
                detail = dict(global_signed_error=(np.asarray(predicted["global_decoded"])-actual_global).tolist(),
                    standardized_global_signed_error=(np.asarray(predicted["standardized_global"])-standardized).tolist(),
                    bellman_signed_error={key: predicted["bellman"][key]-factual["bellman"][key]
                                          for key in ("scaled_reward", "bootstrap", "target")})
                # Terminal all-zero next states have no nonterminal topology class.
                if row.terminal:
                    detail.update(six_losses=None, trust_outcome=None)
                else:
                    six = losses(self.outputs[branch][candidate], y, self.norm)[0]
                    trust = self.trusts[branch]
                    error = six.mean().reshape(1)
                    log_c = float(trust._c_step(error).clamp_min(1e-8).log().item())
                    bootstrap = float((trust.gamma*trust.target(next_states, next_action)).item())
                    q = predicted["trust"]["log_q"]
                    detail.update(six_losses=six.cpu().tolist(), trust_outcome=dict(
                        six_loss_mean=float(error.item()), tau=float(trust._tau),
                        c_step=float(trust._c_step(error).item()), log_c_step=log_c,
                        bootstrap=bootstrap, target=log_c+bootstrap,
                        prediction=q, prediction_minus_target=q-log_c-bootstrap,
                        anchor_tau_held_fixed=True, no_trust_update=True))
                factual["branches"][branch] = detail
            return factual
        finally:
            restore_rng(before)

    def compare(self, low, high, *, low_cost_300, high_cost_300):
        return compare_records(self.record, low, high, low_cost_300=low_cost_300, high_cost_300=high_cost_300)
