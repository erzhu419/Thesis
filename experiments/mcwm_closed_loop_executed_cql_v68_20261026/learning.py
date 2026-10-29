"""Executed-action finite CQL on the original v64 factual update schedule."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from time import perf_counter

import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_value_v64_20261025.learning import (
    FIELDS, ExecutionAction, FreshOutcomeReplay, FreshValueTrainer,
    TransitTrainerConfig, load_table, resident)
from experiments.mcwm_closed_loop_actor_constraint_v66_20261026.learning import hurdle_kl


SCHEDULE = dict(online_actor_update_interval=1, online_actor_batch_mode='latest',
                online_update_order='actor-before-critic')
FAMILIES = ('value', 'actor', 'alpha', 'critic')
CQL_COEFFICIENT = .2
PROPOSALS_PER_STATE = 22


def cql_checks(audit, expected_calls, rows_per_call=128):
    """Shared native/summary admission of the executed-action regularizer."""
    rows = expected_calls*rows_per_call
    return dict(coefficient=audit['coefficient'] == CQL_COEFFICIENT,
        calls=audit['penalty_calls'] == expected_calls,
        factual_rows=audit['factual_rows'] == rows,
        candidate_count=(audit['proposals_per_state'] == PROPOSALS_PER_STATE
            and audit['candidate_rows_before_dedup'] == rows*PROPOSALS_PER_STATE
            and rows <= audit['candidate_rows_after_dedup'] <= rows*PROPOSALS_PER_STATE),
        executed_candidates=audit['candidate_actions_transformed'],
        factual_actions_not_retransformed=not audit['factual_actions_transformed'],
        covariates_only=not audit['outcome_fields_read'],
        isolated_rng=(audit['main_rng_preserved'] and audit['numpy_draws'] == 0),
        exact_alias_deduplication=audit['exact_deduplication'],
        all_heads=audit['all_heads'], finite_penalty=audit['finite_penalty'])


class ExecutedCQLTrainer(FreshValueTrainer):
    """Keep factual TD/actor paths; CQL reads only common current-state s/a."""

    def __init__(self, *args, **kwargs):
        seed = int(kwargs.get('seed', 0))
        super().__init__(*args, **kwargs)
        self._cql_seed = seed+68_000_000
        self._step_counts = dict.fromkeys(FAMILIES, 0)
        self._packet_counts, self._packet_events = [], []
        self._optimizer_order_exact = True
        self._packet_actor = None
        self._optimizer_hooks = []
        for family in FAMILIES:
            def record(optimizer, args, kwargs, family=family):
                self._record_optimizer_step(family)
            self._optimizer_hooks.append(
                getattr(self, family+'_optimizer').register_step_post_hook(record))
        self._cql_calls = self._cql_rows = self._cql_raw = self._cql_unique = 0
        self._cql_unique_min = PROPOSALS_PER_STATE
        self._cql_unique_max = 0
        self._cql_rng_preserved = self._cql_finite = True
        self._cql_penalty_sum = 0.
        self._cql_wall_seconds = 0.
        self._cql_penalty_min, self._cql_penalty_max = math.inf, -math.inf
        self._cql_head_sums = {}

    def activate_executed_cql(self, coefficient=CQL_COEFFICIENT):
        # The source compatibility check and all Adam restoration happen first.
        if any(getattr(self.config, key) != value for key, value in SCHEDULE.items()):
            raise ValueError('v68 retains the original per-step actor-before-critic schedule')
        self.config = replace(self.config, online_cql_alpha=float(coefficient))

    def begin_actor_packet(self):
        if self._packet_counts and not self._packet_complete(self._packet_counts[-1]):
            raise RuntimeError('Complete the preceding eight-update packet')
        self._packet_actor = deepcopy(self.actor).requires_grad_(False)
        self._packet_counts.append(dict.fromkeys(FAMILIES, 0))
        self._packet_events = []

    @staticmethod
    def _packet_complete(counts):
        return counts == dict.fromkeys(FAMILIES, 8)

    def _record_optimizer_step(self, family):
        if not self._packet_counts:
            raise RuntimeError('An optimizer stepped outside a paid feedback packet')
        self._packet_counts[-1][family] += 1
        self._step_counts[family] += 1
        self._packet_events.append(family)
        expected = list(FAMILIES)*8
        self._optimizer_order_exact &= self._packet_events == expected[:len(self._packet_events)]
        if not self._optimizer_order_exact:
            raise RuntimeError('The original value/actor/alpha/critic order changed')

    def continue_step(self, collect_diagnostics=False):
        if not self._packet_counts or self._packet_counts[-1]['critic'] >= 8:
            raise RuntimeError('Begin one paid packet before its eight updates')
        result = super().continue_step(collect_diagnostics=collect_diagnostics)
        if collect_diagnostics:
            diagnostic = result['td_diagnostic']
            diagnostic['critic_change_cosine_negative_td_gradient'] = diagnostic.pop(
                'critic_change_cosine_negative_total_gradient')
            diagnostic['gradient_scope'] = 'TD only; CQL excluded'
        return result

    @torch.no_grad()
    def _executed_candidates(self, states):
        """22 current-state proposals, isolated from every native random stream."""
        before = torch.get_rng_state().clone()
        numpy_before = deepcopy(self.rng.bit_generator.state)
        # Production is CPU-only. fork_rng restores the caller's main Torch state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._cql_seed+self.online_critic_updates)
            parameters = self.actor.distribution_parameters(states)
            expanded = tuple(value[:, None].expand(-1, 10).reshape(-1)
                             for value in parameters)
            uniform = torch.empty(len(states), 10, 1, device=states.device).uniform_(-1., 1.)
            positive = self.actor.soft_action_components_from_parameters(*expanded)[
                'positive_action'].reshape(len(states), 10, 1)
            deterministic = self.actor.soft_action_components_from_parameters(
                *parameters, deterministic_positive=True)['positive_action'][:, None]
            proposals = torch.cat((torch.full_like(deterministic, -1.), uniform,
                                   positive, deterministic), dim=1)
            repeated = states[:, None].expand(-1, PROPOSALS_PER_STATE, -1).reshape(
                -1, states.shape[-1])
            executed = self.action_transform(repeated, proposals.reshape(-1, 1)).reshape(
                len(states), PROPOSALS_PER_STATE)
        self._cql_rng_preserved &= (torch.equal(before, torch.get_rng_state())
            and numpy_before == self.rng.bit_generator.state)
        return executed

    def _cql_penalty(self, batch, data_q):
        started = perf_counter()
        # The parent's batch/data_q can contain replacement outcomes in no_fresh.
        # Use the shared factual covariates instead; observed a is already executed.
        covariates = self.replay_buffer.last_covariates
        states, factual_actions = covariates['observations'], covariates['actions']
        executed = self._executed_candidates(states)
        sorted_actions, _ = executed.sort(dim=1)
        unique = torch.ones_like(sorted_actions, dtype=torch.bool)
        unique[:, 1:] = sorted_actions[:, 1:] != sorted_actions[:, :-1]
        row_ids, columns = unique.nonzero(as_tuple=True)
        counts = unique.sum(dim=1)
        features = self.critic.encode_states(states)
        factual_q = self.critic.forward_features(features, factual_actions)
        candidate_q = self.critic.forward_features(features.index_select(0, row_ids),
            sorted_actions[row_ids, columns, None])
        head_count, size = factual_q.shape
        positions = row_ids*PROPOSALS_PER_STATE+columns
        padded = candidate_q.new_full((head_count, size*PROPOSALS_PER_STATE), -math.inf)
        padded = padded.index_copy(1, positions, candidate_q).reshape(
            head_count, size, PROPOSALS_PER_STATE)
        log_partition = torch.logsumexp(padded, dim=2)-counts.log()[None]
        penalty_by_head_state = log_partition-factual_q
        penalty = penalty_by_head_state.mean()
        if not torch.isfinite(penalty):
            self._cql_finite = False
            raise RuntimeError('Nonfinite executed-action CQL penalty')
        with torch.no_grad():
            candidate_means = torch.zeros_like(factual_q).index_add_(
                1, row_ids, candidate_q)/counts[None]
            differences = candidate_q-factual_q.index_select(1, row_ids)
            absolute_advantage = torch.zeros_like(factual_q).index_add_(
                1, row_ids, differences.abs())/counts[None]
            zero_columns = (sorted_actions == executed[:, :1]).long().argmax(1)
            positive_columns = (sorted_actions == executed[:, -1:]).long().argmax(1)
            rows = torch.arange(size, device=states.device)
            positive_minus_zero = (padded[:, rows, positive_columns]
                                   - padded[:, rows, zero_columns])
            head_values = dict(factual_q_mean=factual_q.mean(1),
                candidate_q_mean=candidate_means.mean(1),
                logmeanexp_mean=log_partition.mean(1),
                penalty_mean=penalty_by_head_state.mean(1),
                candidate_advantage_mean=(candidate_means-factual_q).mean(1),
                candidate_advantage_abs_mean=absolute_advantage.mean(1),
                deterministic_positive_minus_zero_mean=positive_minus_zero.mean(1))
            if not all(bool(torch.isfinite(value).all()) for value in head_values.values()):
                self._cql_finite = False
                raise RuntimeError('Nonfinite executed-action CQL head diagnostic')
            for key, value in head_values.items():
                self._cql_head_sums[key] = self._cql_head_sums.get(key, np.zeros(head_count))+value.cpu().numpy()*size
            value = float(penalty)
            self._cql_calls += 1
            self._cql_rows += size
            self._cql_raw += size*PROPOSALS_PER_STATE
            self._cql_unique += int(counts.sum())
            self._cql_unique_min = min(self._cql_unique_min, int(counts.min()))
            self._cql_unique_max = max(self._cql_unique_max, int(counts.max()))
            self._cql_penalty_sum += value*size
            self._cql_penalty_min = min(self._cql_penalty_min, value)
            self._cql_penalty_max = max(self._cql_penalty_max, value)
        self._cql_wall_seconds += perf_counter()-started
        return penalty

    @torch.no_grad()
    def actor_comparison(self, states):
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
            optimizer_order_exact=self._optimizer_order_exact,
            complete=self._packet_complete(counts))

    def actor_schedule_audit(self):
        return dict(packets=len(self._packet_counts),
            **{family+'_updates': self._step_counts[family] for family in FAMILIES},
            **{'packet_'+family+'_steps': [row[family] for row in self._packet_counts]
               for family in FAMILIES}, optimizer_order_exact=self._optimizer_order_exact,
            schedule={key: getattr(self.config, key) for key in SCHEDULE},
            kl_penalty=False, reference_for_diagnostics_only=True,
            additional_random_draws=0, additional_draws_scope='main native streams',
            isolated_uniform_proposals=self._cql_rows*10,
            isolated_positive_proposals=self._cql_rows*10)

    def cql_audit(self):
        coefficient = self.config.online_cql_alpha
        penalty = dict(mean=self._cql_penalty_sum/self._cql_rows,
            min=self._cql_penalty_min, max=self._cql_penalty_max) if self._cql_rows else dict.fromkeys(('mean', 'min', 'max'), 0.)
        return dict(coefficient=coefficient, temperature=1., penalty_calls=self._cql_calls,
            factual_rows=self._cql_rows, proposals_per_state=PROPOSALS_PER_STATE,
            candidate_rows_before_dedup=self._cql_raw,
            candidate_rows_after_dedup=self._cql_unique,
            isolated_uniform_proposals=self._cql_rows*10,
            isolated_positive_proposals=self._cql_rows*10,
            regularizer_wall_seconds=self._cql_wall_seconds,
            unique_candidates_per_state=dict(min=self._cql_unique_min if self._cql_rows else 0,
                max=self._cql_unique_max,
                mean=self._cql_unique/self._cql_rows if self._cql_rows else 0.),
            main_rng_preserved=self._cql_rng_preserved, numpy_draws=0,
            all_heads=True, head_count=self.config.ensemble_size,
            outcome_fields_read=False, factual_actions_transformed=False,
            candidate_actions_transformed=True, exact_deduplication=True,
            finite_penalty=self._cql_finite, penalty=penalty,
            weighted_penalty={key: coefficient*value for key, value in penalty.items()},
            by_head=[dict(head=head, **{key: float(value[head]/self._cql_rows)
                for key, value in self._cql_head_sums.items()})
                for head in range(self.config.ensemble_size)] if self._cql_rows else [])


def load_learner(seed, *, arm):
    """Read the original archive and Adam before enabling only online CQL."""
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
        raise ValueError('v68 retains original128/64/64 quotas and ten sampled proposals')
    del payload
    replay = FreshOutcomeReplay(history, source, contract, stations, arm=arm, window=285)
    learner = ExecutedCQLTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    learner.activate_executed_cql()
    return learner, replay, contract, stations, initializer
