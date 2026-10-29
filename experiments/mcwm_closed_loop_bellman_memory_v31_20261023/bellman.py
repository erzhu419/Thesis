"""Read-only H2O+ Bellman probes with the same target draw for every row.

These are frozen common-random probes, not the stochastic targets used by a
historical optimizer update. Observed actions already passed F; only the next
policy's latent actions pass F inside the existing SAC bootstrap definition.
"""

from copy import deepcopy
import random

import numpy as np
import torch

from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    restore_rng, rng_state, _same)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _batch
from experiments.mcwm_closed_loop_v1.learning import TransitH2OTrainerV7


ARMS = ("fresh", "no_fresh")
PRIVATE_RNGS = ("rng", "diagnostic_rng")


def _private_rngs(learners):
    return {arm: {name: deepcopy(getattr(learner, name).bit_generator.state)
                  for name in PRIVATE_RNGS}
            for arm, learner in learners.items()}


class FrozenBellman:
    """Evaluate the retained two learners without changing training state/RNG."""

    def __init__(self, learners, seed=316202):
        if set(learners) != set(ARMS):
            raise ValueError("The fresh and complete-history replacement learners are required")
        if any(learner.config.online_backup != "h2oplus" for learner in learners.values()):
            raise ValueError("v31 diagnoses the unchanged H2O+ target")
        self.learners = learners
        self.seed = int(seed)
        self.calls = 0
        self._verified_targets = set()

    def _common_draw(self, learner):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        common = np.random.default_rng(self.seed).bit_generator.state
        for name in PRIVATE_RNGS:
            getattr(learner, name).bit_generator.state = deepcopy(common)

    @torch.no_grad()
    def _components(self, learner, batch):
        """Split the existing action-consistent hybrid SAC path, including entropy."""
        states = batch["next_observations"]
        parts = learner.actor.soft_action_components(states)
        zero_action = learner.action_transform(states, parts["zero_action"])
        positive_action = learner.action_transform(states, parts["positive_action"])
        actions = torch.cat((zero_action, positive_action))
        chosen = learner._sample_target_subset()
        features = learner.target_critic.encode_states(states)
        values = learner.target_critic.forward_features_subset(
            torch.cat((features, features)), actions, chosen)
        zero_q, positive_q = values.split(len(states), dim=1)
        zero_value, positive_value = zero_q.min(0).values, positive_q.min(0).values
        q_only = (parts["zero_probability"]*zero_value
                  + parts["hold_probability"]*positive_value)
        if learner.config.backup_entropy:
            zero_value -= learner.alpha.detach()*parts["zero_log_prob"]
            positive_value -= learner.alpha.detach()*parts["positive_log_prob"]
        sac = (parts["zero_probability"]*zero_value
               + parts["hold_probability"]*positive_value)
        next_value = learner.value(states)
        mixing = learner.config.mixed_backup_lambda
        mixed = mixing*next_value + (1.-mixing)*sac
        effective_discount = batch["discounts"].reshape(-1)*(1.-batch["terminals"].reshape(-1))
        scaled = learner.config.reward_scale*batch["rewards"].reshape(-1)
        reconstructed = scaled + effective_discount*mixed
        q = learner.critic(batch["observations"], batch["actions"])[:, 0]
        scalar = lambda value: float(value.reshape(-1)[0])
        return dict(q_heads=q.tolist(), q_min=float(q.min()), q_mean=float(q.mean()),
            next_value=scalar(next_value), sac_bootstrap=scalar(sac),
            sac_q_without_entropy=scalar(q_only), sac_entropy_bonus=scalar(sac-q_only),
            mixed_bootstrap=scalar(mixed), effective_discount=scalar(effective_discount),
            value_contribution=scalar(effective_discount*mixing*next_value),
            sac_contribution=scalar(effective_discount*(1.-mixing)*sac),
            bootstrap_contribution=scalar(effective_discount*mixed),
            target_subset=chosen.tolist(),
            next_actions=dict(zero_proposal=scalar(parts["zero_action"]),
                positive_proposal=scalar(parts["positive_action"]),
                zero_executed=scalar(zero_action), positive_executed=scalar(positive_action),
                zero_holding_seconds=scalar(learner.action_transform.hold_seconds(zero_action)),
                positive_holding_seconds=scalar(learner.action_transform.hold_seconds(positive_action)),
                hold_probability=scalar(parts["hold_probability"])),
            _reconstructed=reconstructed, _q=q)

    @torch.no_grad()
    def evaluate(self, transition):
        global_before = rng_state()
        private_before = _private_rngs(self.learners)
        result = dict(dt_seconds=float(transition.dt), raw_reward=float(transition.r),
            terminal=bool(transition.terminal), zero_duration=float(transition.dt) == 0.,
            diagnostic_seed=self.seed, common_draw_mode="fixed same draw for every model and event",
            models={}, diagnostic_budget=dict(optimizer_updates=0, training_labels=0))
        for name in ("started_at", "completed_at", "available_at", "sequence"):
            if hasattr(transition, name):
                value = getattr(transition, name)
                result[name] = int(value) if name == "sequence" else float(value)
        try:
            for arm in ARMS:
                learner = self.learners[arm]
                batch = _batch([transition], learner.device)
                self._common_draw(learner)
                point = self._components(learner, batch)
                reconstructed, q = point.pop("_reconstructed"), point.pop("_q")
                # Check native target parity once per frozen model in each child.
                # Subsequent rows use this same decomposition, with new row input.
                if arm not in self._verified_targets:
                    self._common_draw(learner)
                    target = TransitH2OTrainerV7._target(learner, batch, backup="h2oplus")
                    if not torch.equal(target, reconstructed):
                        raise AssertionError("Bellman decomposition differs from the native H2O+ target")
                    self._verified_targets.add(arm)
                target, exact = reconstructed, arm in self._verified_targets
                point.update(target=float(target[0]), target_path_exact=exact,
                    reward_scaled=float((batch["rewards"]*learner.config.reward_scale)[0]),
                    td_heads=(q-target[0]).tolist(), td_min=float(q.min()-target[0]),
                    mixed_backup_lambda=float(learner.config.mixed_backup_lambda),
                    backup_entropy=bool(learner.config.backup_entropy))
                result["models"][arm] = point
        finally:
            for arm, learner in self.learners.items():
                for name in PRIVATE_RNGS:
                    getattr(learner, name).bit_generator.state = deepcopy(private_before[arm][name])
            restore_rng(global_before)
        result["reward_scaled"] = result["models"]["fresh"]["reward_scaled"]
        result["integrity"] = dict(rng_restored=_same(global_before, rng_state()),
            learner_rng_restored=_same(private_before, _private_rngs(self.learners)),
            target_path_exact=all(point["target_path_exact"] for point in result["models"].values()))
        result["native_target_parity"] = "once per frozen model in each child; same decomposition for all rows"
        self.calls += 1
        return result

    __call__ = evaluate
