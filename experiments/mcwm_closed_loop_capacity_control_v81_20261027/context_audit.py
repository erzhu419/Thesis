"""Causal context receipts and the reused first paid eight optimizer updates."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from time import perf_counter

import torch

from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    OPTIMIZERS, _equal, _rng)
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.learning import row_context
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.preflight import (
    native_preflight as original_preflight)


class ContextAudit:
    def __init__(self, out, arm):
        self.arm, self.out = arm, out
        self.issue_log = (out/'context_issues.jsonl').open('x')
        self.selected_log = (out/'context_selected.jsonl').open('x')
        self.issues, self.selected = Counter(), Counter()
        self.by_line = defaultdict(Counter)
        self.reasons = Counter()
        self.native_read = Counter()
        self.native_read_wall_seconds = 0.
        self.checks = dict(atomic_issue_context=True, causal_issue_clock=True,
            issue_identity_exact=True, covariate_context_exact=True,
            outcome_context_exact=True, next_endpoint_exact=True,
            no_fresh_complete_context_retained=True)
        self.first_packet = None

    def read_context(self, reader, issue):
        started = perf_counter()
        try:
            context = reader(issue)
        finally:
            self.native_read['reader_calls'] += 1
            self.native_read_wall_seconds += perf_counter()-started
        self.native_read['simulation_getTime'] += 2
        if context['next_stop'] is not None and 'next_stop_missing' not in context['reasons']:
            for name in ('vehicle_getPersonCapacity','vehicle_getLine',
                         'vehicle_getPersonIDList','busstop_getPersonIDs'):
                self.native_read[name] += 1
        self.native_read['person_getStage0'] += sum(
            values['read'] for values in context['stage_counts'].values())
        return context

    def observe_issue(self, issue):
        context = issue['ctx_s']
        binding, clocks = context['binding'], context['clocks']
        self.checks['atomic_issue_context'] &= bool(clocks['obs_start'] == clocks['obs_end'] ==
            clocks['issue'] == issue['started_at'])
        self.checks['causal_issue_clock'] &= clocks['event'] <= clocks['issue']
        self.checks['issue_identity_exact'] &= all(binding[key] == issue[key] for key in
            ('episode_id','sequence','trip_id','vehicle_id','line_id','station_id'))
        line = self.by_line[str(issue['line_id'])]
        for target in (self.issues, line):
            target['total'] += 1
            target['observed' if context['observed'] else 'missing'] += 1
            if context['observed']:
                counts = context['counts']
                target['onboard_positive'] += counts['onboard'] > 0
                target['next_queue_positive'] += counts['next_queue'] > 0
                target['release_positive'] += counts['release_next'] > 0
        self.reasons.update(context['reasons'])
        self.issue_log.write(json.dumps(dict(sequence=issue['sequence'],
            ctx_s=context), allow_nan=False)+'\n')

    def record_packet(self, covariates, outcomes, *, packet, processed_at, replay):
        record = dict(packet=packet, processed_at=processed_at, covariates=[], outcomes=[])
        for row in covariates:
            context = row.ctx_s
            self.selected['covariates'] += 1
            self.selected['covariates_observed'] += context['observed']
            self.checks['covariate_context_exact'] &= (
                context['binding']['sequence'] == row.sequence and
                context['binding']['issue_clock'] == row.started_at)
            record['covariates'].append(dict(sequence=row.sequence, ctx_s=context))
        if self.arm == 'fresh':
            for row in outcomes:
                context, next_context = row.ctx_s, row.ctx_ns
                self.selected['complete_outcomes'] += 1
                self.selected['outcome_s_observed'] += context['observed']
                self.selected['outcome_ns_observed'] += next_context['observed']
                self.selected['zero_duration_outcomes'] += row.dt == 0
                self.checks['outcome_context_exact'] &= (
                    context['binding']['sequence'] == row.sequence and
                    context['binding']['issue_clock'] == row.started_at)
                binding = next_context['binding']
                self.checks['next_endpoint_exact'] &= binding['issue_clock'] == row.completed_at
                if not row.terminal:
                    self.checks['next_endpoint_exact'] &= (
                        binding['sequence'] == row.sequence+1 and
                        binding['vehicle_id'] == row.next_vehicle_id and
                        binding['episode_id'] == row.episode_id)
                record['outcomes'].append(dict(sequence=row.sequence,
                    ctx_s=context, ctx_ns=next_context))
        else:
            self.checks['no_fresh_complete_context_retained'] &= not replay.context_outcomes
        self.selected_log.write(json.dumps(record, allow_nan=False)+'\n')
        self.selected_log.flush()

    def summary(self, replay, issued_count):
        checks = dict(self.checks,
            all_native_issues_recorded=self.issues['total'] == issued_count,
            native_read_count_exact=self.native_read['reader_calls'] == issued_count,
            covariate_context_budget=self.selected['covariates'] == len(replay.context_covariates) == 285,
            complete_context_budget=self.selected['complete_outcomes'] ==
                len(replay.context_outcomes) == (285 if self.arm == 'fresh' else 0),
            first_paid_packet_pass=self.first_packet is not None and self.first_packet['all_pass'])
        return dict(integrity=dict(all_pass=all(checks.values()), checks=checks),
            issues=dict(counts=dict(self.issues)), selected=dict(counts=dict(self.selected)),
            by_line={key:dict(value) for key,value in sorted(self.by_line.items())},
            missing_reasons=dict(self.reasons), first_packet=self.first_packet,
            replay_context=replay.context_summary(),
            native_observation=dict(calls=dict(self.native_read),
                total_getter_calls=sum(value for key,value in self.native_read.items()
                    if key != 'reader_calls'), wall_seconds=self.native_read_wall_seconds,
                scope='All issue captures are passive observations; only selected285 covariates '
                    'and285 fresh complete outcomes enter learner pools.'),
            issue_context_trace=str(self.out/'context_issues.jsonl'),
            selected_context_trace=str(self.out/'context_selected.jsonl'),
            diagnostic_v79_data_used=False,
            archive_context='history/source NaN with mask0; no native backfill')

    def close(self):
        self.issue_log.close()
        self.selected_log.close()


def _optimizer_steps(optimizer):
    return {id(parameter): int(state['step']) for parameter,state in optimizer.state.items()
            if 'step' in state}


class FirstPacketAudit:
    """Observe the main packet, never execute another update or draw."""
    def __init__(self, learner):
        self.learner = learner
        self.start_update = learner.total_updates
        self.optimizers = {name:getattr(learner,name) for name in OPTIMIZERS}
        self.start_steps = {name:_optimizer_steps(value) for name,value in self.optimizers.items()}
        self.adam_objects = {name:{id(p):id(s) for p,s in value.state.items()}
            for name,value in self.optimizers.items()}
        self.rng_start = self._capture_rng()
        self.update_numbers, self.theta_gradients = [], []

    def _capture_rng(self):
        return dict(global_state=_rng(),
            sampler=deepcopy(self.learner.rng.bit_generator.state),
            outcome=deepcopy(self.learner.replay_buffer.outcome_rng.bit_generator.state))

    def record_update(self):
        self.update_numbers.append(self.learner.total_updates-self.start_update)
        gradient = self.learner.critic.action_weight.grad
        self.theta_gradients.append(dict(present=gradient is not None,
            finite=gradient is not None and bool(torch.isfinite(gradient).all()),
            nonzero=gradient is not None and bool(gradient.any())))

    def finish(self, out):
        before = self._capture_rng()
        learner = self.learner
        end_steps = {name:_optimizer_steps(value) for name,value in self.optimizers.items()}
        checks = dict(exact_eight_main_updates=self.update_numbers == list(range(1,9)),
            original_optimizer_objects=all(getattr(learner,name) is value
                for name,value in self.optimizers.items()),
            existing_Adam_objects_retained=all(
                all(id(value.state[p]) == self.adam_objects[name][id(p)]
                    for p in value.state if id(p) in self.adam_objects[name])
                for name,value in self.optimizers.items()),
            old_Adam_steps_advance_eight=all(end_steps[name].get(key) == step+8
                for name,rows in self.start_steps.items() for key,step in rows.items()),
            new_theta_Adam_step_eight=int(learner.critic_optimizer.state[
                learner.critic.action_weight]['step']) == 8,
            theta_gradient_present_and_finite=all(row['present'] and row['finite']
                for row in self.theta_gradients),
            theta_finite=bool(torch.isfinite(learner.critic.action_weight).all()),
            target_theta_finite=bool(torch.isfinite(learner.target_critic.action_weight).all()),
            replay_calls_eight=learner.replay_buffer.outcome_sample_calls == 8,
            CQL_preserves_main_rng=learner.cql_audit()['main_rng_preserved'])
        # Save only RNG states, no model weights, for a later exact replay audit.
        torch.save(dict(before=self.rng_start, after=before), out/'first_packet_rng.pt')
        checks['audit_adds_no_random_draws'] = _equal(before,self._capture_rng())
        return dict(all_pass=all(checks.values()), checks=checks, actual_updates=8,
            reused_main_day=True, no_extra_steps=True,
            theta=[float(value) for value in learner.critic.action_weight.detach()],
            target_theta=[float(value) for value in learner.target_critic.action_weight.detach()],
            theta_nonzero=bool(learner.critic.action_weight.detach().any()),
            theta_gradient_nonzero_steps=sum(row['nonzero'] for row in self.theta_gradients),
            RNG_receipt=str(out/'first_packet_rng.pt'),
            RNG_scope='Main first8 states recorded; no old-v75 updated-trajectory equality claim',
            prior_zero_init_parity='v80 frozen integration208 checks; no extra updates here')


def native_preflight(*args, context, first_packet, **kwargs):
    result = original_preflight(*args, **kwargs)
    result['checks'].update({'context/'+key:value for key,value in context.checks.items()})
    result['checks'].update({'first_packet/'+key:value for key,value in first_packet['checks'].items()})
    result['context_first_packet'] = first_packet
    result['technical_gate_pass'] = all(result['checks'].values())
    return result
