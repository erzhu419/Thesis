"""Spend the original 24+432 updates; credit probes never affect SGD."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .credit import prepare_chain, score_chain


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(plain(value), handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def steps(learner):
    result = {}
    for name in ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer'):
        counts = [float(row['step']) for row in getattr(learner, name).state.values() if 'step' in row]
        result[name] = dict(min=min(counts, default=None), max=max(counts, default=None))
    return result


def objects(learner):
    return {name: (id(getattr(learner, name)), tuple(id(parameter) for group in
        getattr(learner, name).param_groups for parameter in group['params']))
        for name in ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer')}


def records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PilotUpdates:
    def __init__(self, out, protocol):
        self.out, self.protocol = Path(out), protocol
        self.packets, self.curve = [], []
        self.identities = self.last_steps = None
        self.chains = []
        self.admission = None

    def _start(self, learner):
        if self.identities is None:
            if learner.total_updates != 50000:
                raise RuntimeError('Pilot must retain the original50000 initializer')
            self.out.mkdir(exist_ok=False)
            self.identities, self.last_steps = objects(learner), steps(learner)
        if objects(learner) != self.identities or steps(learner) != self.last_steps:
            raise RuntimeError('Optimizer objects or Adam state changed between packets')

    def _eight(self, learner, replay, receipt):
        self._start(learner)
        index, started = len(self.packets)+1, time.time()
        draws, metrics = [], None
        for step in range(8):
            value = learner.continue_step(collect_diagnostics=step == 0)
            draws.append(deepcopy(replay.last_sampling))
            if step == 0:
                metrics = value
        if learner.total_updates != 50000+8*index:
            raise RuntimeError('The original8-step update budget changed')
        self.last_steps = steps(learner)
        record = dict(packet=index, receipt=receipt, updates=8*index,
            feedback_slots=replay.target_received, metrics=metrics, sampling=draws,
            optimizer_steps=self.last_steps, optimizer_objects_retained=objects(learner) == self.identities,
            update_started_wall=started, update_finished_wall=time.time(),
            rng=dict(torch=torch.get_rng_state(), sampler=learner.rng.bit_generator.state,
                diagnostic=learner.diagnostic_rng.bit_generator.state))
        self.packets.append(plain(record))
        with (self.out/'packets.jsonl').open('a') as handle:
            handle.write(json.dumps(plain(record), separators=(',', ':'), allow_nan=False)+'\n')
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='native_one_step_updates',
            current=index, total=57, updates=8*index, feedback_slots=replay.target_received)), flush=True)

    def warmup(self, learner, replay, rows, receipt):
        self._start(learner)
        index = len(self.packets)+1
        if not 1 <= index <= 3 or len(rows) != 5 or replay.target_received != 5*(index-1):
            raise RuntimeError('Warmup requires exactly three new five-slot packets')
        if any(row.available_at > receipt['processed_at'] for row in rows):
            raise RuntimeError('Warmup feedback was not physically released')
        replay.add_target(rows)
        self._eight(learner, replay, receipt)

    def _score(self, learner):
        before = torch.get_rng_state().clone()
        private = {name: plain(getattr(learner, name).bit_generator.state)
                   for name in ('rng', 'diagnostic_rng')}
        results = []
        for chain in self.chains:
            prefixes = {} if chain['first_reward_event'] is None else {'first_reward': chain['first_reward_event']}
            value = score_chain(learner, chain['prepared'], prefixes=prefixes)
            results.append(dict(line_id=chain['line_id'], arm=chain['arm'],
                first_reward_event=chain['first_reward_event'], **value))
        if not torch.equal(before, torch.get_rng_state()) or any(
                private[name] != plain(getattr(learner, name).bit_generator.state) for name in private):
            raise RuntimeError('Read-only credit probes changed a training RNG')
        update = learner.total_updates-50000
        with (self.out/'credit_packets.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(updates=update, chains=results), separators=(',', ':'), allow_nan=False)+'\n')
        compact = []
        for value in results:
            compact.append(dict(line_id=value['line_id'], arm=value['arm'],
                first_reward_event=value['first_reward_event'], prefixes=value['prefixes'],
                drift=dict(behavior_p_hold_mean=math.fsum(row['behavior_parameters']['positive_probability']
                    for row in value['rows'])/44,
                    target_p_hold_mean=math.fsum(row['target_parameters']['positive_probability']
                    for row in value['rows'])/44,
                    latent_mean_shift=math.fsum(row['target_parameters']['mean']-row['behavior_parameters']['mean']
                    for row in value['rows'])/44,
                    log_std_shift=math.fsum(row['target_parameters']['log_std']-row['behavior_parameters']['log_std']
                    for row in value['rows'])/44)))
        # Every packet is scored and stored resident; selected predeclared
        # boundaries alone enter the small result transferred to the operator.
        if update in self.protocol['credit_score_after_updates']:
            self.curve.append(dict(updates=update, chains=compact))
        return results

    def response(self, learner, replay, rows, receipt, reference):
        self._start(learner)
        if (learner.total_updates != 50024 or replay.target_received != 15 or len(rows) != 270
                or receipt['processed_at'] != 9630.):
            raise RuntimeError('Response admission requires actual9630,15 prior slots and24 updates')
        identifiers = [row.feedback_id for row in rows]
        if len(set(identifiers)) != 270 or any(row.available_at > 9630. for row in rows):
            raise RuntimeError('Response slots are duplicated or physically unavailable')
        for pair in receipt['pairs']:
            arms = {arm: records(branch['events_path']) for arm, branch in pair['branches'].items()}
            reward_event = next((i for i, (low, high) in enumerate(zip(arms['low'], arms['high']))
                if low['transition']['r'] != high['transition']['r']), None)
            for arm in ('low', 'high'):
                self.chains.append(dict(line_id=pair['line_id'], arm=arm,
                    first_reward_event=reward_event,
                    prepared=prepare_chain(reference, arms[arm], pair['selected'])))
        if len(self.chains) != 6:
            raise RuntimeError('Credit scoring needs all three paired native chains')
        # All270 outcomes precede the FIRST response gradient. The previous
        # per-five-row add/update interleaving is deliberately not used here.
        replay.add_target(rows)
        if replay.target_received != 285 or len(replay.recent) != 285:
            raise RuntimeError('15 ordinary plus270 response slots must remain in replay')
        self.admission = dict(clock=9630., response_slots=270, total_slots=285,
            ordinary_retained=15, updates_before=24, new_truth_only=True,
            unique_feedback_ids=270, max_available_at=max(row.available_at for row in rows),
            zero_duration_response=sum(row.dt == 0. for row in rows))
        write(self.out/'admission.json', self.admission)
        self._score(learner)
        for _ in range(54):
            self._eight(learner, replay, dict(processed_at=9630., phase='response_update',
                all285_previously_admitted=True))
            self._score(learner)
        if learner.total_updates != 50456 or len(self.packets) != 57:
            raise RuntimeError('Pilot must finish at the original456 update budget')
        # No trace loss/target is installed. Freeze only after all original
        # one-step updates have finished, before resuming the real native day.
        for name in ('actor', 'critic', 'target_critic', 'value', 'ratio'):
            getattr(learner, name).eval()
            for parameter in getattr(learner, name).parameters():
                parameter.requires_grad_(False)
        with (self.out/'post456_model.pt').open('xb') as handle:
            torch.save(dict(learner=learner, protocol=self.protocol,
                source='new_v43_native_feedback', updates=456), handle)
        write(self.out/'update_summary.json', self.summary(learner, replay))

    def summary(self, learner, replay):
        checks = dict(packets57=len(self.packets) == 57, slots285=replay.target_received == 285,
            updates456=learner.total_updates == 50456,
            response_all_admitted_before_updates=self.admission is not None,
            optimizer_objects_retained=objects(learner) == self.identities,
            frozen_steps_unchanged=steps(learner) == self.last_steps,
            no_trace_target=True, no_diagnostic_truth=True)
        return dict(integrity=dict(all_pass=all(checks.values()), checks=checks),
            budget=dict(ordinary=15, response=270, received=replay.target_received,
                warmup_updates=24, response_updates=432, actual_updates=learner.total_updates-50000),
            admission=self.admission, credit_curve=self.curve, final_optimizer_steps=steps(learner),
            curve_boundaries=self.protocol['credit_score_after_updates'],
            first_reward_definition='first ordinal with differing actual raw rewards; real event clocks can differ, so this is not a clock-matched causal delay; no selection/admission effect',
            metric_scope='L_geom and L_arith condition on observed states; not trajectory-level expectations')
