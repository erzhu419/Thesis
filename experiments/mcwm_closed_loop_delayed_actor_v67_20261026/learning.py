"""v64 factual targets; eight value/critic steps precede one actor/alpha step."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_value_v64_20261025.learning import (
    FIELDS, ExecutionAction, FreshOutcomeReplay, FreshValueTrainer,
    TransitTrainerConfig, load_table, resident)
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.learning import hurdle_kl


SCHEDULE = dict(online_actor_update_interval=8, online_actor_batch_mode='latest',
                online_update_order='critic-before-actor')
FAMILIES = ('value', 'critic', 'actor', 'alpha')


class DelayedActorTrainer(FreshValueTrainer):
    """Use the existing scheduler; hooks observe actual optimizer steps only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_counts = dict.fromkeys(FAMILIES, 0)
        self._packet_counts = []
        self._packet_events = []
        self._actor_after_critic_steps = []
        self._optimizer_order_exact = True
        self._packet_actor = None
        self._optimizer_hooks = []
        for family in FAMILIES:
            def record(optimizer, args, kwargs, family=family):
                self._record_optimizer_step(family)
            self._optimizer_hooks.append(
                getattr(self, family+'_optimizer').register_step_post_hook(record))

    def activate_delayed_schedule(self):
        # Loading first preserves the source initializer's exact compatibility
        # contract, Adam moments, counters and RNG without an adapter.
        self.config = replace(self.config, **SCHEDULE)

    def begin_actor_packet(self):
        if self._packet_counts and not self._packet_complete(self._packet_counts[-1]):
            raise RuntimeError('Complete the preceding eight-critic/one-actor packet')
        if any(getattr(self.config, key) != value for key, value in SCHEDULE.items()):
            raise RuntimeError('Activate the fixed delayed schedule after loading the initializer')
        self._packet_actor = deepcopy(self.actor).requires_grad_(False)
        self._packet_counts.append(dict.fromkeys(FAMILIES, 0))
        self._packet_events = []

    @staticmethod
    def _packet_complete(counts):
        return counts == dict(value=8, critic=8, actor=1, alpha=1)

    def _record_optimizer_step(self, family):
        if not self._packet_counts:
            raise RuntimeError('An optimizer stepped outside a paid feedback packet')
        counts = self._packet_counts[-1]
        counts[family] += 1
        self._step_counts[family] += 1
        self._packet_events.append(family)
        if family == 'actor':
            self._actor_after_critic_steps.append(counts['critic'])
        expected = ['value', 'critic']*8+['actor', 'alpha']
        position = len(self._packet_events)
        self._optimizer_order_exact &= self._packet_events == expected[:position]
        if not self._optimizer_order_exact:
            raise RuntimeError('The paid packet optimizer order changed')

    def continue_step(self, collect_diagnostics=False):
        if not self._packet_counts or self._packet_counts[-1]['critic'] >= 8:
            raise RuntimeError('Begin one paid packet before its eight critic updates')
        return super().continue_step(collect_diagnostics=collect_diagnostics)

    @torch.no_grad()
    def actor_comparison(self, states):
        """Same-state diagnostics only: no KL term enters any training loss."""
        if self._packet_actor is None:
            raise RuntimeError('Begin the paid packet before comparing its actor')
        current = self.actor.distribution_parameters(states)
        reference = self._packet_actor.distribution_parameters(states)
        result = hurdle_kl(current, reference)
        for name, parameters in (('reference', reference), ('current', current)):
            gate, mean, log_std = parameters
            components = self.actor.soft_action_components_from_parameters(
                *parameters, deterministic_positive=True)
            proposal = torch.where((components['hold_probability'] >= .5)[:, None],
                components['positive_action'], components['zero_action'])
            result.update({f'hold_probability_{name}': torch.sigmoid(gate),
                f'positive_mean_{name}': mean, f'positive_log_std_{name}': log_std,
                f'{name}_hold_seconds': self.action_transform.hold_seconds(
                    self.action_transform(states, proposal)).reshape(-1)})
        result['holding_delta_seconds'] = result['current_hold_seconds']-result['reference_hold_seconds']
        result['gate_flip'] = ((result['hold_probability_current'] >= .5)
                              != (result['hold_probability_reference'] >= .5))
        return result

    def packet_actor_audit(self):
        counts = self._packet_counts[-1]
        return dict(packet=len(self._packet_counts),
            **{family+'_updates': counts[family] for family in FAMILIES},
            actor_update_after_critic_steps=(self._actor_after_critic_steps[-1]
                if counts['actor'] else None),
            optimizer_order_exact=self._optimizer_order_exact,
            complete=self._packet_complete(counts))

    def actor_schedule_audit(self):
        return dict(packets=len(self._packet_counts),
            **{family+'_updates': self._step_counts[family] for family in FAMILIES},
            **{'packet_'+family+'_steps': [row[family] for row in self._packet_counts]
               for family in FAMILIES},
            actor_update_after_critic_steps=list(self._actor_after_critic_steps),
            optimizer_order_exact=self._optimizer_order_exact,
            schedule={key: getattr(self.config, key) for key in SCHEDULE},
            kl_penalty=False, reference_for_diagnostics_only=True,
            additional_random_draws=0)


def load_learner(seed, *, arm):
    """Load the original initializer and all Adam state before changing schedule."""
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
    learner = DelayedActorTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    learner.activate_delayed_schedule()
    return learner, replay, contract, stations, initializer
