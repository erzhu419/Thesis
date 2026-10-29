"""Keep the v70 learner; separate policy covariates from complete TD outcomes."""
from collections import deque

import numpy as np
import torch

from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import learning as original
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import (
    FreshOutcomeReplay as BaseReplay, _batch, _join)

FIELDS = original.FIELDS
ExecutionAction = original.ExecutionAction
ExecutedCQLTrainer = original.ExecutedCQLTrainer
TransitTrainerConfig = original.TransitTrainerConfig
SCHEDULE, FAMILIES = original.SCHEDULE, original.FAMILIES
cql_checks, recency_checks, target_checks = (
    original.cql_checks, original.recency_checks, original.target_checks)


class DecoupledReplay(BaseReplay):
    """285 latest-five s/a rows; private recent30 complete risk-selected outcomes."""

    def __init__(self, *args, outcome_seed, **kwargs):
        super().__init__(*args, window=285, **kwargs)
        self.fresh_outcomes = deque(maxlen=285)
        self.outcome_rng = np.random.default_rng(outcome_seed)
        self.outcome_seed = int(outcome_seed)
        self.outcome_sample_calls = self.maximum_sampled_label_age = 0
        self.last_outcome_sampling = None
        self.covariate_ids, self.outcome_ids, self.selected_outcome_ids = [], [], []

    def add_target(self, covariates, outcomes):
        if len(covariates) != len(outcomes) or len(covariates) != 5:
            raise ValueError('Each packet requires five original covariates and five complete outcomes')
        for row in covariates:
            self.recent.append({key: np.asarray(getattr(row, key)).copy() for key in ('s', 'a')})
            self.covariate_ids.append(int(row.sequence))
            self.target_received += 1
        self.selected_outcome_ids.extend(int(row.sequence) for row in outcomes)
        if self.arm == 'fresh':
            for row in outcomes:
                self.fresh_outcomes.append({key: np.asarray(getattr(row, key)).copy()
                    for key in FIELDS.values()})
                self.outcome_ids.append(int(row.sequence))

    def sample_mixed(self, size, online_fraction, *, rng):
        if online_fraction != .5 or size % 4:
            raise ValueError('Preserve original source/history/fresh quotas2:1:1')
        if not self.recent:
            raise RuntimeError('Continue only after an admitted covariate packet')
        count = size//4
        # Preserve all four native draws, including the historical replacement.
        base_ids = rng.choice(len(self.history['s']), size=count, p=self.history_weights)
        replacement_ids = rng.choice(len(self.history['s']), size=count, p=self.history_weights)
        recent_ids = rng.integers(len(self.recent), size=count)
        source_ids = rng.integers(len(self.source['s']), size=2*count)
        covariates = {key: np.asarray([row[key] for row in self.recent]) for key in ('s', 'a')}
        historical = _batch(self.history, base_ids, self.device)
        self.last_covariates = dict(
            observations=torch.cat((historical['observations'], torch.as_tensor(
                covariates['s'][recent_ids], dtype=torch.float32, device=self.device))),
            actions=torch.cat((historical['actions'], torch.as_tensor(
                covariates['a'][recent_ids], dtype=torch.float32, device=self.device).reshape(-1, 1))))
        self.last_sampling = dict(history=base_ids.tolist(), replacement=replacement_ids.tolist(),
            recent=recent_ids.tolist(), source=source_ids.tolist())
        outcomes = _batch(self.history, replacement_ids, self.device)
        # The private fifth draw advances in both arms, as in v70.
        window = min(30, len(self.recent))
        local_ids = self.outcome_rng.integers(window, size=count)
        self.outcome_sample_calls += 1
        self.last_outcome_sampling = dict(window=window, local_indices=local_ids.tolist())
        if self.arm == 'fresh':
            tail = list(self.fresh_outcomes)[-window:]
            table = {key: np.asarray([row[key] for row in tail]) for key in FIELDS.values()}
            outcomes = _batch(table, local_ids, self.device)
            self.maximum_sampled_label_age = max(self.maximum_sampled_label_age,
                int(np.max(window-1-local_ids)))
        return _join(historical, outcomes), _batch(self.source, source_ids, self.device)

    def summary(self):
        cov, outcome, selected = set(self.covariate_ids), set(self.outcome_ids), set(self.selected_outcome_ids)
        return dict(super().summary(), fresh_outcomes_stored=len(self.fresh_outcomes),
            covariate_window=285, critic_fresh_outcome_window=30 if self.arm == 'fresh' else 0,
            independent_outcome_sampler_seed=self.outcome_seed,
            independent_outcome_sampler_calls=self.outcome_sample_calls,
            maximum_fresh_outcome_age_in_admitted_labels=(
                self.maximum_sampled_label_age if self.arm == 'fresh' else None),
            no_fresh_original_batch_unchanged=self.arm == 'no_fresh',
            covariate_records_received=len(self.covariate_ids),
            outcome_labels_received=len(self.outcome_ids),
            covariate_unique_records=len(cov), outcome_unique_records=len(outcome),
            distinct_target_records_used=len(cov | outcome),
            selected_target_record_union=len(cov | selected),
            outcome_state_record_overlap=len(cov & outcome),
            covariate_rows_only_sa=all(set(row) == {'s', 'a'} for row in self.recent),
            complete_outcome_rows=all(set(row) == set(FIELDS.values()) for row in self.fresh_outcomes),
            fresh_current_s_a_use=['IQL value', 'SAC actor', 'temperature', 'executed CQL'],
            recency_semantics='Independent latest-five covariates; risk-selected complete outcomes; '
                'only the latest30 admitted outcomes enter fresh TD. Original four sampler draws '
                'and the private outcome draw preserve their order.')


FreshOutcomeReplay = DecoupledReplay


def load_learner(seed, *, arm):
    """Restore the exact v70 initializer, Adam and trainer, then connect split replay."""
    learner, old, contract, stations, initializer = original.load_learner(seed, arm=arm)
    history = dict(old.history, sample_weight=old.history_weights.copy())
    replay = DecoupledReplay(history, old.source, contract, stations,
        arm=arm, outcome_seed=seed+600000)
    # Preserve already normalized history weights bit for bit; renormalization
    # could perturb the categorical sampling probabilities on the no_fresh path.
    replay.history_weights = old.history_weights.copy()
    learner.replay_buffer = replay
    return learner, replay, contract, stations, initializer
