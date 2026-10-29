"""Reject native drift, nonparallel execution and changed training semantics."""
from copy import deepcopy
import json

from experiments.mcwm_closed_loop_fast_branch_v24_20261023 import compare as c


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    source, probe = root/'source', root/'probe'
    issue = dict(id='train/7X/bootstrap', index=4, role='train',
        selected=dict(started_at=1559., sequence=20))
    write(source/'acquisitions.json', dict(records=[issue]))
    rows = {}
    for arm, hold, pid, start in (('low', 0., 101, 1.), ('high', 7.5, 102, 2.)):
        value = dict(id=issue['id'], status='complete', integrity=dict(all_pass=True),
            selected=issue['selected'], prefix=dict(counters=dict(policy=16), receipts=[dict(at=1200)]),
            prediction=dict(route=dict(wait=3.)), service_prediction=dict(shared=dict(arrival=[7.5]*3)),
            service_history=dict(status='supported'), execution=dict(requested_hold_seconds=hold, executed_hold_seconds=hold),
            truth=dict(fixed300=dict(dt=300., wait=123.+hold)), service_shift=dict(arrival=[100.+hold]),
            global_transition=dict(dt=3., s=[1., 0.]), suffix=dict(seconds=300, training_delta=dict(policy=0, world=0)),
            native=dict(other_batch_actions=[], prefix_matched_issues=21), clock_audit=dict(actor_real_clock=True),
            budget=dict(prefix_sim_seconds=1559., suffix_sim_seconds=300, truth_labels_enter_training=0),
            execution_wall=dict(pid=pid, native_started_wall=start, native_finished_wall=start+10.), wall_seconds=100.)
        write(source/'training_branches/4'/arm/'summary.json', value)
        child = deepcopy(value)
        child['budget']['prefix_sim_seconds'] = 0
        child['wall_seconds'] = 5.
        write(probe/arm/'summary.json', child)
        rows[arm] = child
    parent = deepcopy(rows['low'])
    parent['budget']['prefix_sim_seconds'] = 1559.
    parent['wall_seconds'] = 120.
    write(probe/'summary.json', parent)
    live = dict(native_before=1559., native_after=1559., counters_before=dict(policy=16), counters_after=dict(policy=16),
        parent_rng_unchanged=True, parent_trace_cookie_unchanged=True,
        children={arm:value['execution_wall'] for arm, value in rows.items()}, overlap_seconds=9.)
    write(probe/'live_fork.json', live)
    return source, probe


def mutate(path, fn):
    value = c.read(path)
    fn(value)
    write(path, value)


def test_native_equivalence_has_small_gate_and_separate_compute(tmp_path):
    source, probe = fixture(tmp_path)
    result = c.compare(probe, source, 'train/7X/bootstrap')
    assert result['status'] == 'complete' and result['integrity']['all_pass']
    assert result['target']['index'] == 4
    assert result['diagnostic_budget'] == dict(prefix_sim_seconds=1559., suffixes=3,
        suffix_sim_seconds=900, new_training_labels=0, new_training_updates=0)
    assert not result['decision']['full_day_acceleration_deployed']
    assert result['decision']['stage_b'] == 'closed'
    assert len(json.dumps(result).encode()) < 20000


def test_truth_drift_is_exactly_rejected_without_tolerance(tmp_path):
    source, probe = fixture(tmp_path)
    mutate(probe/'high/summary.json', lambda v:v['truth']['fixed300'].update(wait=130.50000000001))
    result = c.compare(probe, source, 'train/7X/bootstrap')
    assert not result['integrity']['checks']['old_branches_exact']
    assert result['exact_replay']['children_to_original']['high']['mismatches']['truth'][0]['path'] == 'truth.fixed300.wait'


def test_parent_continuation_drift_is_rejected(tmp_path):
    source, probe = fixture(tmp_path)
    mutate(probe/'summary.json', lambda v:v['global_transition'].update(dt=4.))
    result = c.compare(probe, source, 'train/7X/bootstrap')
    assert not result['integrity']['checks']['parent_continuation_exact']


def test_parent_state_drift_is_rejected_even_with_exact_truth(tmp_path):
    source, probe = fixture(tmp_path)
    mutate(probe/'live_fork.json', lambda v:v.update(native_after=1560., parent_rng_unchanged=False))
    checks = c.compare(probe, source, 'train/7X/bootstrap')['integrity']['checks']
    assert not checks['parent_clock_fixed'] and not checks['parent_rng_fixed']


def test_serial_children_are_rejected(tmp_path):
    source, probe = fixture(tmp_path)
    mutate(probe/'live_fork.json', lambda v:v['children']['high'].update(native_started_wall=12., native_finished_wall=20.))
    assert not c.compare(probe, source, 'train/7X/bootstrap')['integrity']['checks']['children_actually_parallel']


def test_suffix_training_and_added_labels_are_rejected(tmp_path):
    source, probe = fixture(tmp_path)
    mutate(probe/'low/summary.json', lambda v:(v['suffix']['training_delta'].update(policy=1),
        v['budget'].update(truth_labels_enter_training=1)))
    checks = c.compare(probe, source, 'train/7X/bootstrap')['integrity']['checks']
    assert not checks['all_suffixes_zero_training'] and not checks['diagnostic_truth_not_fitted']


def test_children_replaying_prefix_is_rejected(tmp_path):
    source, probe = fixture(tmp_path)
    mutate(probe/'low/summary.json', lambda v:v['budget'].update(prefix_sim_seconds=1559.))
    assert not c.compare(probe, source, 'train/7X/bootstrap')['integrity']['checks']['children_no_prefix_replay']


def test_missing_required_field_is_not_mistaken_for_equal_none(tmp_path):
    source, probe = fixture(tmp_path)
    for path in (probe/'low/summary.json', source/'training_branches/4/low/summary.json'):
        mutate(path, lambda v:v.pop('prediction'))
    assert not c.compare(probe, source, 'train/7X/bootstrap')['integrity']['checks']['old_branches_exact']


def test_pending_and_native_failure_are_separate(tmp_path):
    source, probe = fixture(tmp_path)
    (probe/'high/summary.json').unlink()
    assert c.compare(probe, source, 'train/7X/bootstrap')['status'] == 'pending'
    write(probe/'high/failure.json', dict(message='native exception'))
    assert c.compare(probe, source, 'train/7X/bootstrap')['status'] == 'native_failed'
