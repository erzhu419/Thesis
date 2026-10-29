"""Passive v8 response predictions scored against resident v7 fork truth."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from experiments.mcwm_closed_loop_action_fork_v7_20261020.predictions import (
    ActionForkPrediction, BRANCHES, CANDIDATES, _topology, compare_records, paired_effect,
)
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import (
    decode_event_step, response_features,
)
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    restore_rng, rng_state,
)
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation


class ResponsePrediction:
    """Save unmodified and response-augmented predictions on one common anchor.

    Readouts were fitted on selected, available prefix facts.  Resident fork
    outcomes are joined only by ``compare`` and never used to fit a readout.
    Trust is copied verbatim from the original prediction, without an update.
    """

    @torch.no_grad()
    def __init__(self, learner, state, memory_features, nonholding_seconds,
                 numeric_lines, readouts, *, seed):
        before = rng_state()
        try:
            self.legacy = ActionForkPrediction(learner, state, seed=seed)
            self.state = self.legacy.state
            self.candidates = copy.deepcopy(self.legacy.candidates)
            self.valid_pair = self.legacy.valid_pair
            self.evaluator = self.legacy.evaluator
            self.record = self.legacy.summary()
            self.record["legacy"] = self.legacy.summary()
            self.record["response_context"] = dict(
                memory_features=torch.as_tensor(memory_features).detach().cpu().reshape(-1).tolist(),
                nonholding_seconds=None if nonholding_seconds is None else float(nonholding_seconds),
                numeric_lines=[int(value) for value in numeric_lines],
                training="selected available prefix facts only; diagnostic truth excluded")
            world = learner.world
            family = self.record["common_family"]
            decoder = SimpleNamespace(norm=self.legacy.norm, stations=self.legacy.stations,
                device=self.legacy.device,
                generator=torch.Generator(device=self.legacy.device))
            branches = {}
            for name, models in (("online", world.models),
                                 ("frozen_common", world.frozen_models)):
                records = {}
                readout = readouts[name][family].to(self.legacy.device)
                for candidate in CANDIDATES:
                    action = self.legacy.actions[candidate]
                    features = response_features(self.state, action, memory_features,
                                                 nonholding_seconds, numeric_lines)
                    correction = features @ readout
                    output = models[family](self.state, action).detach().clone()
                    output[:, :4] += correction
                    decoder.generator.manual_seed(int(seed))
                    next_states, rewards, durations = decode_event_step(
                        self.state, action, output,
                        lambda states, values: ModelAugmentation.decode(decoder, states, values))
                    batch = dict(observations=self.state, actions=action,
                        next_observations=next_states, rewards=rewards,
                        delta_t_seconds=durations,
                        discounts=torch.ones(1, device=self.legacy.device),
                        terminals=torch.zeros(1, device=self.legacy.device))
                    target = float(self.evaluator.target(batch).item())
                    scaled = float(rewards.item())*self.evaluator.learner.config.reward_scale
                    global_values = output[0, :4]*decoder.norm.g_std+decoder.norm.g_mean
                    raw = global_values.clone()
                    raw[:2] = raw[:2].expm1()
                    increments = (next_states[0, 3:5]-self.state[0, 3:5])*20000
                    decoded = [float(durations.item()), -float(rewards.item()),
                               *increments.cpu().tolist()]
                    records[candidate] = dict(
                        executed_action=self.candidates[candidate]["executed_action"],
                        holding_seconds=self.candidates[candidate]["holding_seconds"],
                        global_raw=raw.cpu().tolist(), global_decoded=decoded,
                        standardized_global=output[0, :4].cpu().tolist(),
                        response_correction_standardized=correction[0].cpu().tolist(),
                        decoded_next_topology=_topology(next_states[0].cpu().numpy(),
                                                       decoder.stations.cpu().numpy()),
                        analytic_within_batch=bool(durations.item() == 0),
                        bellman=dict(scaled_reward=scaled, bootstrap=target-scaled, target=target),
                        trust=copy.deepcopy(self.record["legacy"]["branches"][name]
                                            ["candidates"][candidate]["trust"]))
                branches[name] = dict(candidates=records,
                    **paired_effect(records["low"], records["high"]))
            self.record["branches"] = branches
            self.record["semantics"] = (
                "Original action-dependent base plus selected-prefix-fact response residual; "
                "exact analytic same-batch advancement takes precedence. Common family, "
                "decoder draws, Bellman evaluator and unchanged trust; no truth training.")
        finally:
            restore_rng(before)

    def summary(self):
        return copy.deepcopy(self.record)

    def compare(self, truth_low, truth_high, *, low_cost_300, high_cost_300):
        """Join small saved outcomes, retaining anchor and branch-order checks."""
        return compare_records(self.record, truth_low, truth_high,
            low_cost_300=low_cost_300, high_cost_300=high_cost_300)
