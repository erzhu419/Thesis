"""Frozen post456 action coverage; six existing issues, no learning or truth."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def compact(value):
    if isinstance(value, float):
        return float(f'{value:.7g}')
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return value


def versions(learner):
    parameters = [parameter for name in ('actor', 'critic', 'target_critic', 'value', 'ratio')
                  for parameter in getattr(learner, name).parameters()]
    return [(id(parameter), parameter._version) for parameter in parameters+[learner.log_alpha]]


def main():
    import numpy as np
    import torch
    from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
    from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.mean_target import mean8_state
    from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.score import load_bank
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import optimizer_versions
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng, _restore, _equal

    torch.set_num_threads(1)
    protocol = json.loads((HERE/'protocol.json').read_text())
    started, global_rng = time.perf_counter(), _rng()
    rows, critic_rows = [], 0
    checks = dict(frozen_parameters=True, Adam=True, training_private_rng=True,
                  extra_stream=True, global_rng=True, same_F=True, original285_456=True)
    try:
        for scenario_index, scenario in enumerate(protocol['scenarios']):
            learners = [torch.load(HERE/'runs'/scenario/arm/'post456_model.pt',
                        map_location='cpu', weights_only=False)['learner'] for arm in ('M1', 'M8')]
            before = [dict(parameters=versions(model), adam=optimizer_versions(model),
                private=[deepcopy(getattr(model, name).bit_generator.state)
                         for name in ('rng', 'diagnostic_rng')],
                extra=mean8_state(model) if arm == 'M8' else None)
                for arm, model in zip(('M1', 'M8'), learners)]
            transform = learners[0].action_transform
            checks['same_F'] &= (type(transform) is type(learners[1].action_transform)
                and transform.__dict__ == learners[1].action_transform.__dict__)
            checks['original285_456'] &= all(model.total_updates == 50456
                and model.replay_buffer.target_received == 285 for model in learners)
            if not checks['same_F'] or not checks['original285_456']:
                raise RuntimeError('Frozen endpoint F or285/456 source budget differs')
            bank = load_bank(HERE.parent/protocol['source_version']/'runs'/scenario/'response_receipt.json')
            with torch.no_grad():
                for line_index, line in enumerate(protocol['focus_lines']):
                    row = next(item['row'] for item in bank if item['line_id'] == line and item['arm'] == 'low')
                    state = torch.as_tensor(np.asarray(row.s), dtype=torch.float32).reshape(1, -1)
                    memory = torch.as_tensor(row.memory_s['vector'], dtype=state.dtype).reshape(1, -1)
                    actor_actions = [model.actor(state, deterministic=True)[0].reshape(1, 1)
                                     for model in learners]
                    fixed = state.new_tensor([0., 15., 30., 60.]).reshape(-1, 1)
                    proposals = torch.cat((2.*fixed/transform.max_hold-1., *actor_actions))
                    executed = transform(state.repeat(len(proposals), 1), proposals)
                    action_values = executed.reshape(-1).tolist()
                    representatives, aliases = [], []
                    for index, value in enumerate(action_values):
                        match = next((group for group, previous in enumerate(representatives)
                                      if action_values[previous] == value), None)
                        if match is None:
                            match = len(representatives)
                            representatives.append(index)
                        aliases.append(match)
                    unique = executed[representatives]
                    minimum = []
                    for model in learners:
                        explicit = critic_states(model, state, memory)
                        q = model.critic(explicit.repeat(len(unique), 1), unique).min(0).values
                        minimum.append(q[aliases])
                    advantages = [(q-q[0]).tolist() for q in minimum]
                    best = [int(q.argmax()) for q in minimum]
                    seconds = transform.hold_seconds(executed).reshape(-1).tolist()
                    actor_delta = [float(q[5]-q[4]) for q in minimum]
                    actor_order = [int(delta > 0)-int(delta < 0) for delta in actor_delta]
                    rows.append([scenario_index, line_index,
                        transform.hold_seconds(proposals).reshape(-1).tolist(), seconds, aliases,
                        *advantages, [action_values[index] for index in best],
                        [seconds[index] for index in best], actor_delta, actor_order])
                    critic_rows += 2*len(unique)
            for arm, model, saved in zip(('M1', 'M8'), learners, before):
                checks['frozen_parameters'] &= versions(model) == saved['parameters']
                checks['Adam'] &= optimizer_versions(model) == saved['adam']
                checks['training_private_rng'] &= all(_equal(saved['private'][index],
                    getattr(model, name).bit_generator.state)
                    for index, name in enumerate(('rng', 'diagnostic_rng')))
                checks['extra_stream'] &= _equal(saved['extra'], mean8_state(model) if arm == 'M8' else None)
            del learners
        checks['global_rng'] = _equal(global_rng, _rng())
        if not all(checks.values()):
            raise RuntimeError('Frozen coverage changed '+str([name for name, value in checks.items() if not value]))
    finally:
        _restore(global_rng)
    value = compact(dict(status='complete', scenarios=protocol['scenarios'], lines=protocol['focus_lines'],
        columns=['scenario', 'line', 'proposal_seconds', 'executed_seconds', 'F_alias_groups',
                 'adv_M1', 'adv_M8', 'best_executed_action_M1_M8', 'best_seconds_M1_M8',
                 'Q_actorM8_minus_actorM1_each_critic', 'actor_order_each_critic'],
        rows=rows, actor_order='-1 prefersM1;0 tie;+1 prefersM8',
        proof=dict(all_pass=True, **checks),
        compute=dict(models_read=4, actor_forwards=12, critic_forwards=12,
            critic_rows=critic_rows, target_forwards=0, optimizer_steps=0,
            new_feedback=0, native_branches=0, seconds=time.perf_counter()-started),
        scope='post456_only;3existing_issues_per_scenario;deduplicated_executed_actions;no58packet_extended_grid_curve;no_truth_or_control_gain'))
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > 3500:
        raise RuntimeError(f'Coverage summary exceeds3500B: {len(encoded.encode())}')
    path = HERE/'analysis/coverage_summary.json'
    with path.open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status='complete', bytes=len(encoded.encode()), proof=value['proof'])), flush=True)


if __name__ == '__main__':
    main()
