"""Passive, matched-draw removal of each local-to-global correction head.

All variants retain the live generated states/actions, live trust weights,
current critics and policy.  They do not train a shadow policy or claim an
optimizer-mediated action effect; the reported next actions only respond to
the counterfactual predicted next state.
"""

from __future__ import annotations

import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.probes import (
    ProbedTrainer as V3ProbedTrainer,
)
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import (
    TransitH2OTrainerV7,
)
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.world import (
    HeadDiagnosticWorld,
)


HEAD_NAMES = ("logdt", "logcost", "dWaiting", "dOnboard")


def _mean(value):
    return float(value.detach().mean())


def _paired(actual, counterfactual):
    """Signed differences always mean actual calibrated minus removal variant."""
    delta = actual.detach() - counterfactual.detach()
    return dict(actual_mean=_mean(actual), counterfactual_mean=_mean(counterfactual),
                mean_delta=_mean(delta), mean_absolute_delta=_mean(delta.abs()),
                max_absolute_delta=float(delta.abs().max()))


class ProbedWorld(HeadDiagnosticWorld):
    """Extend v3's whole-calibration probe with four one-head removals."""

    capture_counterfactual = False
    counterfactual_batch = None
    head_counterfactual_batches = None
    head_counterfactual_weights = None
    head_counterfactual_outputs = None

    @torch.no_grad()
    def generate(self, actor, size):
        self.counterfactual_batch = None
        self.head_counterfactual_batches = None
        self.head_counterfactual_weights = None
        self.head_counterfactual_outputs = None
        before = self.generator.get_state() if self.capture_counterfactual else None
        result = super().generate(actor, size)
        if result is None or not self.capture_counterfactual:
            return result
        batch, weights = result
        after = self.generator.get_state()
        s, a = batch["observations"], batch["actions"]
        actual_output = self.predict(s, a)
        base_output = self.predict(s, a, calibrated=False)
        variants, outputs = {}, {}
        try:
            self.generator.set_state(before)
            ns, reward, dt = ModelAugmentation.decode(self, s, base_output)
            self.counterfactual_batch = dict(batch, next_observations=ns,
                                            rewards=reward, delta_t_seconds=dt)
            variants["full_base"] = self.counterfactual_batch
            outputs["full_base"] = base_output[:, :4].clone()
            for index, name in enumerate(HEAD_NAMES):
                output = actual_output.clone()
                output[:, index] = base_output[:, index]
                self.generator.set_state(before)
                ns, reward, dt = ModelAugmentation.decode(self, s, output)
                variants["omit_"+name] = dict(batch, next_observations=ns,
                    rewards=reward, delta_t_seconds=dt)
                outputs["omit_"+name] = output[:, :4].clone()
        finally:
            self.generator.set_state(after)
        self.head_counterfactual_batches = variants
        self.head_counterfactual_weights = weights.detach().clone()
        self.head_counterfactual_outputs = dict(actual=actual_output[:, :4].clone(),
                                                variants=outputs)
        return result


class ProbedTrainer(V3ProbedTrainer):
    """Add observation-only per-head targets to v3's unchanged live update."""

    head_counterfactual = None

    def continue_step(self, collect_diagnostics=False):
        self.head_counterfactual = None
        return super().continue_step(collect_diagnostics=collect_diagnostics)

    def _target(self, batch, *, backup):
        probe = (self._collect_td and self._making_model and self.world is not None
                 and self.world.head_counterfactual_batches is not None)
        before = torch.get_rng_state() if probe else None
        actual_target = super()._target(batch, backup=backup)
        if not probe:
            return actual_target
        weights = self.world.head_counterfactual_weights
        output = self.world.head_counterfactual_outputs
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            q = self.critic(batch["observations"], batch["actions"])
            actual_loss = self._elementwise_td_loss(q, actual_target)
            actual_ns = batch["next_observations"]
            actual_u = self.actor(actual_ns, deterministic=True)[0]
            actual_a = self.action_transform(actual_ns, actual_u)
            hold = self.action_transform.hold_seconds
            variants = {}
            for name, counterfactual in self.world.head_counterfactual_batches.items():
                # Restore the state BEFORE the live target for every variant,
                # so target-subset and positive-policy samples match exactly.
                torch.set_rng_state(before)
                cf_target = TransitH2OTrainerV7._target(self, counterfactual, backup=backup)
                cf_loss = self._elementwise_td_loss(q, cf_target)
                cf_ns = counterfactual["next_observations"]
                cf_u = self.actor(cf_ns, deterministic=True)[0]
                cf_a = self.action_transform(cf_ns, cf_u)
                actual_g = output["actual"]*self.world.norm.g_std+self.world.norm.g_mean
                cf_g = output["variants"][name]*self.world.norm.g_std+self.world.norm.g_mean
                dt = _paired(batch["delta_t_seconds"], counterfactual["delta_t_seconds"])
                cost = _paired(-batch["rewards"], -counterfactual["rewards"])
                target = _paired(actual_target, cf_target)
                weighted_loss = _paired(actual_loss*weights[None], cf_loss*weights[None])
                scaled_reward_delta = ((batch["rewards"]-counterfactual["rewards"])
                                       *self.config.reward_scale)
                variants[name] = dict(rows=len(actual_target),
                    applied_rows=int(((output["actual"]-output["variants"][name])
                                      .abs().sum(1)>1e-6).sum()),
                    duration_seconds=dt, cost_passenger_seconds=cost,
                    waiting_increment_persons=_paired(actual_g[:, 2], cf_g[:, 2]),
                    onboard_increment_persons=_paired(actual_g[:, 3], cf_g[:, 3]),
                    target_q=target,
                    scaled_reward_component=_paired(batch["rewards"]*self.config.reward_scale,
                        counterfactual["rewards"]*self.config.reward_scale),
                    bootstrap_component_delta=dict(mean=_mean(actual_target-cf_target-scaled_reward_delta),
                        mean_absolute=_mean((actual_target-cf_target-scaled_reward_delta).abs())),
                    td_loss_unweighted=_paired(actual_loss, cf_loss),
                    td_loss_trust_weighted=weighted_loss,
                    td_loss_actual_model_contribution=_paired(.25*actual_loss*weights[None],
                                                              .25*cf_loss*weights[None]),
                    next_proposed_hold_seconds=_paired(hold(actual_u), hold(cf_u)),
                    next_executed_hold_seconds=_paired(hold(actual_a), hold(cf_a)),
                    decoded_waiting_state_persons=_paired(actual_ns[:, 3]*20000, cf_ns[:, 3]*20000),
                    decoded_onboard_state_persons=_paired(actual_ns[:, 4]*20000, cf_ns[:, 4]*20000))
            self.head_counterfactual = dict(rows=len(actual_target),
                trust_mean=_mean(weights), variants=variants,
                semantics="Actual calibrated minus full-base or single-head removal; identical generated starting states, executed actions, decoder random draws, current critic/policy, target random draws and fixed actual trust weights. Next actions are a predicted-state response without a shadow optimizer update.",
                modeled_duration_enters_bellman_discount=False)
        return actual_target
