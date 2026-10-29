"""Only paid fresh critic rows use V(next_s); original full target draws remain."""
from dataclasses import replace
import json

import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import (
    FIELDS, ExecutionAction, FreshOutcomeReplay, FreshOutcomeTrainer,
    TransitTrainerConfig, load_table, resident)
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import distribution


class FreshValueTrainer(FreshOutcomeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_mixed_target_calls = self.parent_mixed_target_rows = 0
        self.fresh_value_replacement_calls = self.fresh_value_replacement_rows = 0
        self.no_fresh_unchanged_returns = 0

    @torch.no_grad()
    def _target(self, batch, *, backup):
        # Always evaluate the original full target first, including SAC's
        # positive-action and target-head draws on every original batch row.
        target = super()._target(batch, backup=backup)
        if backup != 'h2oplus':
            return target
        self.parent_mixed_target_calls += 1
        self.parent_mixed_target_rows += len(target)
        if self.outcome_arm == 'no_fresh':
            self.no_fresh_unchanged_returns += 1
            return target
        if len(target) != self.config.batch_size:
            raise ValueError('Fresh V target requires the original combined history/fresh/source batch')
        quarter = self.config.batch_size//4
        # FreshOutcomeReplay+native update concatenate [history,fresh,source].
        selected = slice(quarter, 2*quarter)
        values = self.value(batch['next_observations'])
        effective_discount = batch['discounts'].reshape(-1)*(1.-batch['terminals'].reshape(-1))
        result = target.clone()
        result[selected] = (self.config.reward_scale*batch['rewards'].reshape(-1)[selected]
            + effective_discount[selected]*values[selected])
        self.fresh_value_replacement_calls += 1
        self.fresh_value_replacement_rows += quarter
        if self._collect_td:
            # The parent recorded its original target; report the final target
            # that actually enters this critic update instead.
            self._real_batches[-1]['target_q'] = distribution(result)
        return result

    def target_audit(self):
        quarter = self.config.batch_size//4
        return dict(parent_mixed_target_calls=self.parent_mixed_target_calls,
            parent_mixed_target_rows=self.parent_mixed_target_rows,
            fresh_value_replacement_calls=self.fresh_value_replacement_calls,
            fresh_value_replacement_rows=self.fresh_value_replacement_rows,
            no_fresh_unchanged_returns=self.no_fresh_unchanged_returns,
            fresh_rows_start=quarter, fresh_rows_stop=2*quarter,
            mixed_target_generated_before_replacement=True, original_sampler_draws=4,
            rule='Only fresh outcome rows use reward_scale*r+discounts*(1-terminal)*V(ns); '
                'historical/source retain original mixed target; no_fresh returns parent exactly.')


def load_learner(seed, *, arm):
    """Restore the exact seed archive and Adam, with the original285-row replay."""
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
    if config.batch_size != 256:
        raise ValueError('Actual control requires the original128/64/64 quotas')
    del payload
    replay = FreshOutcomeReplay(history, source, contract, stations, arm=arm, window=285)
    learner = FreshValueTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    return learner, replay, contract, stations, initializer
