"""Whole-day gate accepts exact execution and rejects scientific/stream drift."""
from copy import deepcopy
import json

import pytest

from experiments.mcwm_closed_loop_fast_branch_v24r1_20261023 import day_compare as c


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    source, fast = root/'source', root/'fast'
    records = []
    for index in range(9):
        issue = dict(id=f'train/{index}', index=index, role='train',
            selected=dict(started_at=1559.+index*100, sequence=index))
        records.append(issue)
        children = {}
        for arm, hold, pid, shift in (('low', 0., 100+index*2, 0.), ('high', 7.5, 101+index*2, 1.)):
            child = dict(status='complete', integrity=dict(all_pass=True), selected=issue['selected'],
                prefix=dict(counters=dict(target_labels=index*5), issued_count=index+1),
                prediction=dict(route=dict(wait=3.)), service_prediction=dict(history=dict(arrival=7.5)),
                service_history=dict(status='supported'),
                execution=dict(requested_hold_seconds=hold, executed_hold_seconds=hold),
                truth=dict(fixed300=dict(dt=300., wait=123.+hold)), service_shift=dict(arrival=[100.+hold]),
                global_transition=dict(dt=3., s=[1., 0.]),
                suffix=dict(seconds=300, training_delta=dict(policy=0, world=0)),
                native=dict(other_batch_actions=[], prefix_matched_issues=index+1),
                clock_audit=dict(actor_real_clock=True),
                budget=dict(prefix_sim_seconds=issue['selected']['started_at'], suffix_sim_seconds=300,
                    replay_target_labels=index*5, replay_policy_updates=index*8, truth_labels_enter_training=0),
                execution_wall=dict(pid=pid, native_started_wall=1000.+index*100+shift,
                    native_finished_wall=1010.+index*100+shift), wall_seconds=100.-20.*shift)
            write(source/'training_branches'/str(index)/arm/'summary.json', child)
            child = deepcopy(child)
            child['budget'].update(prefix_sim_seconds=0, replay_target_labels=0, replay_policy_updates=0)
            child['fork_reuse'] = dict(prefix_execution='inherited_not_independently_replayed',
                parent_verified_prefix_seconds=issue['selected']['started_at'], parent_recorded_issues=index+1,
                inherited_target_labels=index*5)
            child['execution_wall'].update(native_started_wall=10.+index*100+shift,
                native_finished_wall=12.+index*100+shift)
            child['wall_seconds'] = 3.+shift
            write(fast/'training_branches'/str(index)/arm/'summary.json', child)
            children[arm] = child
        live = dict(native_before=issue['selected']['started_at'], native_after=issue['selected']['started_at'],
            counters_before=dict(policy=index*8), counters_after=dict(policy=index*8),
            parent_rng_unchanged=True, parent_trace_cookie_unchanged=True,
            children={arm:child['execution_wall'] for arm, child in children.items()}, overlap_seconds=1.,
            route_streams=dict(count=1, parent_offsets_before={'12':32768},
                parent_offsets_after={'12':32768}, parent_offsets_unchanged=True))
        write(fast/'training_branches'/str(index)/'live_fork.json', live)
    main = {field:{} for field in c.MAIN_FIELDS}
    main.update(status='complete', integrity=dict(all_pass=True), initializer=dict(initial_updates=50000),
        budget=dict(target_labels=285, ordinary_labels=267, paired_action_labels=18,
            training_pairs=9, paired_outcomes_per_pair=2, packets=57,
            policy_updates=456, trust_updates=456, native_seconds=24000), acquisitions=records,
        training_branch_budget=dict(prefix_sim_seconds=0, replay_target_labels=0,
            replay_policy_updates=0, suffix_sim_seconds=5400), wall_seconds=10000.)
    for directory in (source, fast):
        write(directory/'summary.json', main)
        write(directory/'acquisitions.json', dict(records=records))
        (directory/'prefix_trace.jsonl').write_text(''.join(json.dumps(dict(sequence=i, s=[float(i)]))+'\n'
            for i in range(9)))
    main['wall_seconds'] = 1000.
    write(fast/'summary.json', main)
    return source, fast


def mutate(path, fn):
    value = c.read(path)
    fn(value)
    write(path, value)


def test_full_day_equivalence_reports_measured_parallel_occupancy(tmp_path):
    source, fast = fixture(tmp_path)
    result = c.compare(fast, source)
    assert result['integrity']['all_pass'] and result['decision']['ready_for_next_experiment']
    assert result['compute']['observed_wall_speedup'] == 10.
    parallel = result['compute']['parallel']
    assert parallel['fast_suffix']['weighted_overlap_factor'] == pytest.approx(4./3.)
    assert parallel['original_replay']['weighted_overlap_factor'] == pytest.approx(20./11.)
    assert parallel['fast_suffix']['arm_wall_seconds_sum'] == 63.
    assert parallel['original_replay']['arm_wall_seconds_sum'] == 1620.
    assert result['paired_suffixes'][0]['parallel']['fast_suffix']['pair_span_seconds'] == 3.
    assert len(json.dumps(result).encode()) < 20000


@pytest.mark.parametrize('drift', ('trace', 'suffix_training', 'route_offset'))
def test_full_day_gate_blocks_scientific_or_stream_drift(tmp_path, drift):
    source, fast = fixture(tmp_path)
    if drift == 'trace':
        (fast/'prefix_trace.jsonl').write_text('{"sequence":0,"s":[0.00000000001]}\n')
    elif drift == 'suffix_training':
        mutate(fast/'training_branches/0/low/summary.json',
            lambda row:row['suffix']['training_delta'].update(policy=1))
    else:
        mutate(fast/'training_branches/0/live_fork.json',
            lambda row:row['route_streams']['parent_offsets_after'].update({'12':65536}))
    result = c.compare(fast, source)
    assert not result['integrity']['all_pass'] and not result['decision']['ready_for_next_experiment']
    if drift == 'trace':
        assert not result['integrity']['checks']['complete_trace_exact']
    else:
        failed = result['paired_suffixes'][0]['failed_checks']
        assert ('zero_suffix_training' if drift == 'suffix_training' else 'parent_route_offsets_unchanged') in failed
