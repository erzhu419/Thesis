"""Frozen stochastic-policy holding summaries from the existing v47 models.

The actual v43 controller samples its hurdle gate. Deterministic mode alone
therefore does not identify its executed holding distribution. Here the gate
is integrated analytically and positive Gaussian draws use the declared v46
128 common seeds. No critic, learner target, replay update or SUMO call occurs.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def compact(value):
    if isinstance(value, float):
        return float(f'{value:.9g}')
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return value


def _extra_evidence(learner):
    """Cheap candidate-only evidence, without copying models or Adam arrays."""
    import torch
    if not hasattr(learner, 'mean8_extra_rng_state'):
        return None
    last = learner.mean8_last
    versions = None if last is None else {
        name: (id(value), value._version) if isinstance(value, torch.Tensor) else value
        for name, value in last.items()}
    return dict(rng=learner.mean8_extra_rng_state.clone(),
        compute=deepcopy(learner.mean8_compute), last_versions=versions)


def actor_scores(learner, row, blocks):
    """One fixed issue state, exact Bernoulli gate and conditional positive CRNs."""
    import math
    import numpy as np
    import torch
    states = torch.as_tensor(np.asarray(row.s), dtype=torch.float32,
                             device=learner.device).reshape(1, -1)
    parameters = learner.actor.distribution_parameters(states)
    _, mean, log_std = parameters
    zero_action = torch.full((1, 1), -1., dtype=states.dtype, device=learner.device)
    zero = float(learner.action_transform.hold_seconds(
        learner.action_transform(states, zero_action)).reshape(-1)[0])
    proposal = learner.actor(states, deterministic=True)[0]
    deterministic = float(learner.action_transform.hold_seconds(
        learner.action_transform(states, proposal)).reshape(-1)[0])
    block_means, positive_samples, p_hold = [], [], None
    for block in blocks:
        expectations = []
        for seed in block['draw_seeds']:
            torch.manual_seed(int(seed))
            parts = learner.actor.soft_action_components_from_parameters(*parameters)
            probability = float(parts['hold_probability'].reshape(-1)[0])
            if p_hold is None:
                p_hold = probability
            elif probability != p_hold:
                raise RuntimeError('The analytic gate changed inside a fixed issue state')
            positive = float(learner.action_transform.hold_seconds(
                learner.action_transform(states, parts['positive_action'])).reshape(-1)[0])
            # actor.forward(False) draws Bernoulli(p_hold), so its zero mass
            # is 1-p_hold. This also closes the constant-F case exactly.
            expectations.append(zero+probability*(positive-zero))
            positive_samples.append(positive)
        block_means.append(math.fsum(expectations)/8.)
    return dict(expected_holding=math.fsum(block_means)/16., block_means=block_means,
        p_hold=p_hold, latent_mean=float(mean.reshape(-1)[0]),
        latent_log_std=float(log_std.reshape(-1)[0]), deterministic_F_holding=deterministic,
        zero_F_holding=zero, positive_F_min=min(positive_samples),
        positive_F_max=max(positive_samples), issue_clock=float(row.started_at),
        memory_observed_at=float(row.memory_s['observed_at']))


def main():
    import numpy as np
    import torch
    from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.score import load_bank
    from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import frozen_state, unchanged
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import _loaded, optimizer_versions
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _restore

    spec, protocol = read(HERE/'policy_summary_spec.json'), read(HERE/'protocol.json')
    blocks = spec['blocks']
    if (len(blocks) != 16 or any(len(block['draw_seeds']) != 8 for block in blocks)
            or len({seed for block in blocks for seed in block['draw_seeds']}) != 128):
        raise RuntimeError('The declared fixed16 by8 positive CRNs are incomplete')
    torch.set_num_threads(1)
    started, global_rng = time.time(), _rng()
    output = HERE/'analysis/policy_summary.json'
    output.parent.mkdir(exist_ok=True)
    rows = []
    counts = dict(resident_checkpoints_loaded=0, cached_parameter_forwards=0,
        total_actor_network_forwards=0,
        deterministic_actor_forwards=0, positive_conditional_draws=0,
        analytic_gate_integrations=0, F_calls=0, critic_forwards=0,
        target_calls=0, optimizer_updates=0, new_feedback_slots=0,
        new_native_branches=0, checkpoints_downloaded=0, actor_seconds=0., checkpoint_load_seconds=0.)
    columns = ['scenario', 'updates', 'line', 'M1_Eholding', 'M8_Eholding',
        'paired_delta_mean', 'paired_delta_sample_sd', 'paired_delta_min', 'paired_delta_max',
        'M1_p_hold', 'M8_p_hold', 'M1_latent_mean', 'M8_latent_mean',
        'M1_latent_logstd', 'M8_latent_logstd', 'M1_deterministic_Fhold',
        'M8_deterministic_Fhold', 'M1_zero_Fhold', 'M8_zero_Fhold']
    try:
        with torch.no_grad():
            for scenario in spec['scenarios']:
                bank = load_bank(HERE.parent/protocol['source_version']/'runs'/scenario/'response_receipt.json')
                selected = {item['line_id']: item['row'] for item in bank if item['arm'] == 'low'}
                if set(selected) != set(spec['lines']):
                    raise RuntimeError('The three existing low-arm issue states are missing')
                for update in spec['checkpoints']:
                    scores = {}
                    for arm in spec['arms']:
                        load_started = time.perf_counter()
                        learner, payload = _loaded(HERE/'runs'/scenario/arm/f'post{update}_model.pt')
                        counts['checkpoint_load_seconds'] += time.perf_counter()-load_started
                        if (payload['scenario'] != scenario or payload['arm'] != arm
                                or payload['updates'] != update
                                or learner.total_updates != 50000+update
                                or learner.replay_buffer.target_received != (15 if update == 24 else 285)):
                            raise RuntimeError('A saved v47 checkpoint has a different identity or budget')
                        before, rng = frozen_state(learner), _rng()
                        adam, extra = optimizer_versions(learner), _extra_evidence(learner)
                        actor_started = time.perf_counter()
                        scores[arm] = {line: actor_scores(learner, selected[line], blocks) for line in spec['lines']}
                        counts['actor_seconds'] += time.perf_counter()-actor_started
                        _restore(rng)
                        if (not unchanged(learner, before, rng)['all_pass']
                                or not _equal(adam, optimizer_versions(learner))
                                or not _equal(extra, _extra_evidence(learner))):
                            raise RuntimeError('Frozen actor-only scoring changed model, Adam, RNG or extra stream')
                        counts['resident_checkpoints_loaded'] += 1
                        counts['cached_parameter_forwards'] += 3
                        counts['total_actor_network_forwards'] += 6
                        counts['deterministic_actor_forwards'] += 3
                        counts['positive_conditional_draws'] += 3*128
                        counts['analytic_gate_integrations'] += 3*128
                        counts['F_calls'] += 3*(128+2)
                        del learner, payload
                    for line in spec['lines']:
                        low, high = scores['M1'][line], scores['M8'][line]
                        differences = np.asarray(high['block_means'])-np.asarray(low['block_means'])
                        rows.append([scenario, update, line, low['expected_holding'], high['expected_holding'],
                            float(differences.mean()), float(differences.std(ddof=1)),
                            float(differences.min()), float(differences.max()),
                            low['p_hold'], high['p_hold'], low['latent_mean'], high['latent_mean'],
                            low['latent_log_std'], high['latent_log_std'],
                            low['deterministic_F_holding'], high['deterministic_F_holding'],
                            low['zero_F_holding'], high['zero_F_holding']])
        if len(rows) != 12 or not _equal(global_rng, _rng()):
            raise RuntimeError('The unified12-row scope or global RNG was changed')
        result = compact(dict(status='complete', columns=columns, rows=rows,
            units='seconds', paired_direction='M8 minus M1', blocks=16, draws_per_block=8,
            proof=dict(all_pass=True, models_Adam_RNG_extra_stream_unchanged=True,
                actor_only=True, saved_low_issue_representatives=True,
                same128_positive_Gaussian_CRNs=True, gate='native Bernoulli p_hold; p_zero=1-p_hold'),
            inference=dict(**counts, wall_seconds=time.time()-started),
            scope='Conditional expected F holding at3 existing issue states per scenario; gate analytically integrated. Block dispersion describes128 fixed CRNs, not independent states or control-cost benefit.'))
        encoded = json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n'
        if len(encoded.encode()) > spec['max_bytes']:
            raise RuntimeError('The policy-only small summary exceeds its5000-byte limit')
        with output.open('x') as handle:
            handle.write(encoded)
        print(json.dumps(dict(status='complete', bytes=len(encoded.encode()), rows=12,
            inference=result['inference'])), flush=True)
        return result
    finally:
        _restore(global_rng)


if __name__ == '__main__':
    main()
