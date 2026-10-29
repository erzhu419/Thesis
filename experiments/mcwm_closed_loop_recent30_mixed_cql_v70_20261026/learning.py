"""Keep v69 recent30/CQL; restore the original mixed backup on fresh rows."""
from dataclasses import replace
import json

import numpy as np
import torch

from experiments.mcwm_closed_loop_recent30_cql_v69_20261026.learning import (
    FIELDS, ExecutionAction, FreshOutcomeReplay, TransitTrainerConfig,
    load_table, resident, cql_checks, recency_checks, SCHEDULE, FAMILIES,
    CQL_COEFFICIENT)
from experiments.mcwm_closed_loop_executed_cql_v68_20261026.learning import (
    ExecutedCQLTrainer as OriginalExecutedCQLTrainer)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import (
    FreshOutcomeTrainer)


class ExecutedCQLTrainer(OriginalExecutedCQLTrainer):
    """Bypass only v64's fresh V rewrite; all parent update paths stay intact."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fresh_mixed_target_calls = self.fresh_mixed_target_rows = 0

    @torch.no_grad()
    def _target(self, batch, *, backup):
        # This is the same original mixed target that v69 always evaluated
        # before replacing its fresh quarter. Actor/head sampling occurs once
        # on the complete batch, with the unchanged native random streams.
        target = FreshOutcomeTrainer._target(self, batch, backup=backup)
        if backup == 'h2oplus':
            self.parent_mixed_target_calls += 1
            self.parent_mixed_target_rows += len(target)
            if self.outcome_arm == 'no_fresh':
                self.no_fresh_unchanged_returns += 1
            else:
                self.fresh_mixed_target_calls += 1
                self.fresh_mixed_target_rows += self.config.batch_size//4
        return target

    def target_audit(self):
        values = super().target_audit()
        if self.outcome_arm == 'no_fresh':
            # The full no_fresh receipt remains exactly comparable with v68.
            return values
        return dict(values,
            fresh_mixed_target_calls=self.fresh_mixed_target_calls,
            fresh_mixed_target_rows=self.fresh_mixed_target_rows,
            mixed_backup_lambda=self.config.mixed_backup_lambda,
            mixed_target_generated_before_replacement=False,
            rule='All rows retain the original reward_scale*r+discounts*(1-terminal)'
                '*(0.1*V(ns)+0.9*SAC(ns)); no fresh V-only replacement.')


def target_checks(audit, arm, calls, batch_size=256):
    """Validate the one changed target scope in native and summary admission."""
    fresh = arm == 'fresh'
    quarter = batch_size//4
    return dict(original_full_mixed_target=(audit.get('parent_mixed_target_calls') == calls
            and audit.get('parent_mixed_target_rows') == calls*batch_size),
        no_value_only_replacement=(audit.get('fresh_value_replacement_calls') == 0
            and audit.get('fresh_value_replacement_rows') == 0),
        fresh_mixed_scope=(audit.get('fresh_mixed_target_calls') == calls
            and audit.get('fresh_mixed_target_rows') == calls*quarter
            and audit.get('mixed_backup_lambda') == .1
            and audit.get('mixed_target_generated_before_replacement') is False) if fresh
            else 'fresh_mixed_target_calls' not in audit,
        unchanged_no_fresh_returns=audit.get('no_fresh_unchanged_returns') == (0 if fresh else calls),
        original_row_partition=(audit.get('fresh_rows_start') == quarter
            and audit.get('fresh_rows_stop') == 2*quarter),
        original_replay_draw_count=audit.get('original_sampler_draws') == 4)


def load_learner(seed, *, arm):
    """Restore the original archive and Adam before enabling unchanged CQL."""
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
    if (config.batch_size != 256 or config.cql_n_actions != 10
            or config.online_backup != 'h2oplus' or config.mixed_backup_lambda != .1):
        raise ValueError('v70 requires original128/64/64 quotas and original .1V+.9SAC backup')
    del payload
    replay = FreshOutcomeReplay(history, source, contract, stations, arm=arm, outcome_seed=seed+600000)
    learner = ExecutedCQLTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    learner.activate_executed_cql()
    return learner, replay, contract, stations, initializer
