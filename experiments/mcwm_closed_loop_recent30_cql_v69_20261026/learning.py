"""Fixed v63 recent30 replay composed with unchanged v68 fresh-V and CQL."""
from dataclasses import replace
import json

import numpy as np
import torch

from experiments.mcwm_closed_loop_control_campaign_v60_20261025.control import (
    RecentOutcomeReplay as FreshOutcomeReplay)
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.learning import (
    FIELDS, ExecutionAction, ExecutedCQLTrainer, TransitTrainerConfig,
    load_table, resident, cql_checks, SCHEDULE, FAMILIES, CQL_COEFFICIENT)


def recency_checks(audit, arm, calls, seed):
    fresh = arm == 'fresh'
    received = min(285, calls//8*5)
    age = audit.get('maximum_fresh_outcome_age_in_admitted_labels')
    return dict(current_covariates_unchanged=audit.get('covariate_window') == 285
            and audit.get('recent_covariates') == received,
        fixed_outcome_window=audit.get('critic_fresh_outcome_window') == (30 if fresh else 0),
        private_sampler_seed=audit.get('independent_outcome_sampler_seed') == seed+600000,
        private_sampler_calls=audit.get('independent_outcome_sampler_calls') == calls,
        eligible_fresh_age=(isinstance(age, int) and 0 <= age <= 29) if fresh else age is None,
        no_fresh_batch_unchanged=fresh or audit.get('no_fresh_original_batch_unchanged') is True,
        outcome_isolation=audit.get('fresh_outcomes_stored') == (received if fresh else 0))


def load_learner(seed, *, arm):
    """Restore v68 initialization before adding the fixed v63 replay sampler."""
    torch.set_num_threads(1)
    inputs = resident.INPUTS
    metadata = json.loads((inputs/'history_metadata.json').read_text())
    with np.load(inputs/'history.npz') as archive:
        history = {key: archive[key] for key in (*FIELDS.values(), 'sample_weight')}
    source = load_table(inputs/'source_9101.npz', 4096)
    contract = resident.formal.TransitMDPConfigV7(**metadata['contract'])
    stations = np.asarray(metadata['station_max_by_line'], np.float32)
    initializer = inputs/f'h2oplus_seed{seed}.pt'
    payload = torch.load(initializer, map_location='cpu', weights_only=False)
    if (payload['total_updates'] != 50000 or payload['online_critic_updates'] != 0
            or payload['extra']['source_training_seed'] != seed):
        raise ValueError('Wrong read-only initializer')
    config = replace(TransitTrainerConfig(**payload['source_config']), use_ratio_correction=False)
    if config.batch_size != 256 or config.cql_n_actions != 10:
        raise ValueError('v69 retains original128/64/64 quotas and ten sampled proposals')
    del payload
    replay = FreshOutcomeReplay(history, source, contract, stations, arm=arm, outcome_seed=seed+600000)
    learner = ExecutedCQLTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    learner.activate_executed_cql()
    return learner, replay, contract, stations, initializer
