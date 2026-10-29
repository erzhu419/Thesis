"""Paired response ablations with one unchanged legacy and value anchor."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from experiments.mcwm_closed_loop_action_fork_v7_20261020.predictions import (
    BRANCHES, CANDIDATES, _topology, compare_records, paired_effect,
)
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import (
    decode_event_step, response_features,
)
from experiments.mcwm_closed_loop_action_response_v8_20261021.predictions import ResponsePrediction
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import restore_rng, rng_state
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.ablation_world import (
    ARM_NAMES, FEATURE_GROUPS, group_contributions,
)
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation


PREDICTION_ARMS = ("base_analytic", *ARM_NAMES,
                   *(f"drop_{name}" for name in FEATURE_GROUPS))


def prediction_readouts(readouts_by_arm):
    """Add zero and fixed-weight deletions to the three separately fitted arms.

    Inputs have shape ``arm -> online/frozen_common -> family -> [66,4]``.
    Deletion arms always start from the complete fitted readout and are never
    fitted again; the supplied training readouts are left untouched.
    """
    result = {arm: {branch: {family: value.detach().clone()
                            for family, value in readouts_by_arm[arm][branch].items()}
                    for branch in BRANCHES} for arm in ARM_NAMES}
    complete = result["memory_interaction"]
    result["base_analytic"] = {
        branch: {family: torch.zeros_like(value) for family, value in complete[branch].items()}
        for branch in BRANCHES}
    for group, columns in FEATURE_GROUPS.items():
        result[f"drop_{group}"] = {
            branch: {family: value.clone() for family, value in complete[branch].items()}
            for branch in BRANCHES}
        for families in result[f"drop_{group}"].values():
            for value in families.values():
                value[list(columns)] = 0
    return {arm: result[arm] for arm in PREDICTION_ARMS}


class AblationPrediction:
    """Evaluate all readouts with the same decoder draws, trust and evaluator.

    Only one v8 legacy capture is constructed.  Every response arm uses that
    exact evaluator, family, executed actions and saved pre-intervention state.
    The complete arm retains v8 arithmetic and decoding, including analytic
    advancement of supported zero-duration same-batch events.
    """

    @torch.no_grad()
    def __init__(self, learner, state, memory_features, nonholding_seconds,
                 numeric_lines, readouts_by_arm, *, seed):
        before = rng_state()
        try:
            readouts = prediction_readouts(readouts_by_arm)
            base = ResponsePrediction(learner, state, memory_features,
                nonholding_seconds, numeric_lines, readouts["base_analytic"], seed=seed)
            self.legacy, self.state = base.legacy, base.state
            self.candidates = copy.deepcopy(base.candidates)
            self.valid_pair, self.evaluator = base.valid_pair, base.evaluator
            self.record = base.summary()
            self.record.pop("branches")
            self.record["arms"] = {"base_analytic": self._arm_record(base.summary())}
            family = self.record["common_family"]
            decoder = SimpleNamespace(norm=self.legacy.norm, stations=self.legacy.stations,
                device=self.legacy.device,
                generator=torch.Generator(device=self.legacy.device))
            features = {candidate: response_features(self.state, self.legacy.actions[candidate],
                memory_features, nonholding_seconds, numeric_lines) for candidate in CANDIDATES}
            for arm in PREDICTION_ARMS[1:]:
                record = self._arm_record(base.summary())
                for branch, models in (("online", learner.world.models),
                                       ("frozen_common", learner.world.frozen_models)):
                    weight = readouts[arm][branch][family].to(self.legacy.device)
                    candidates = {}
                    for candidate in CANDIDATES:
                        action = self.legacy.actions[candidate]
                        correction = features[candidate] @ weight
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
                        values = output[0, :4]*decoder.norm.g_std+decoder.norm.g_mean
                        raw = values.clone()
                        raw[:2] = raw[:2].expm1()
                        increments = (next_states[0, 3:5]-self.state[0, 3:5])*20000
                        candidates[candidate] = dict(
                            executed_action=self.candidates[candidate]["executed_action"],
                            holding_seconds=self.candidates[candidate]["holding_seconds"],
                            global_raw=raw.cpu().tolist(),
                            global_decoded=[float(durations.item()), -float(rewards.item()),
                                            *increments.cpu().tolist()],
                            standardized_global=output[0, :4].cpu().tolist(),
                            response_correction_standardized=correction[0].cpu().tolist(),
                            decoded_next_topology=_topology(next_states[0].cpu().numpy(),
                                                           decoder.stations.cpu().numpy()),
                            analytic_within_batch=bool(durations.item() == 0),
                            bellman=dict(scaled_reward=scaled, bootstrap=target-scaled, target=target),
                            trust=copy.deepcopy(self.record["legacy"]["branches"][branch]
                                                ["candidates"][candidate]["trust"]))
                    record["branches"][branch] = dict(candidates=candidates,
                        **paired_effect(candidates["low"], candidates["high"]))
                self.record["arms"][arm] = record
            self.record["full_group_contributions"] = self._contributions(
                features, readouts["memory_interaction"], family)
            self.record["semantics"] = (
                "Three selected-prefix-only fitted arms; zero base and full-weight group "
                "deletions are never refitted. All arms share one legacy, decoder seed, "
                "Bellman evaluator and unchanged trust. Contributions are standardized "
                "pre-decoder residuals; analytic within-batch advancement overrides them.")
        finally:
            restore_rng(before)

    @staticmethod
    def _arm_record(record):
        """Store large shared context and legacy only once, at the top level."""
        return {key: value for key, value in record.items()
                if key not in ("legacy", "response_context", "semantics")}

    def _contributions(self, features, readouts, family):
        result = {}
        for branch in BRANCHES:
            weight = readouts[branch][family].to(self.legacy.device)
            contributions = {candidate: group_contributions(features[candidate], weight)
                             for candidate in CANDIDATES}
            groups = {}
            for name in FEATURE_GROUPS:
                low, high = (contributions[candidate][name][0] for candidate in CANDIDATES)
                groups[name] = dict(low=low.cpu().tolist(), high=high.cpu().tolist(),
                                   high_minus_low=(high-low).cpu().tolist())
            low, high = (features[candidate] @ weight for candidate in CANDIDATES)
            result[branch] = dict(groups=groups,
                total=dict(low=low[0].cpu().tolist(), high=high[0].cpu().tolist(),
                           high_minus_low=(high-low)[0].cpu().tolist()))
        return result

    def summary(self):
        return copy.deepcopy(self.record)

    def compare(self, truth_low, truth_high, *, low_cost_300, high_cost_300):
        """Join truth once; per-arm comparisons retain the common-anchor checks."""
        result = None
        comparisons = {}
        for arm, prediction in self.record["arms"].items():
            comparison = compare_records(prediction, truth_low, truth_high,
                low_cost_300=low_cost_300, high_cost_300=high_cost_300)
            comparisons[arm] = comparison.pop("predicted_versus_factual")
            if result is None:
                result = comparison
        result["arms"] = comparisons
        return result
