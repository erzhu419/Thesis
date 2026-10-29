from experiments.mcwm_closed_loop_objective_alignment_v58_20261025.support import (
    inventory, validate_chain, validate_pair,
)


def row(seq=2, start=10., stop=12., action=0.):
    return dict(sequence=seq, started_at=start, completed_at=stop, dt=stop-start,
                s=[start], a=[action], ns=[stop], r=-2*(stop-start), terminal=False,
                memory_s=dict(observed_at=start, vector=[start]),
                memory_ns=dict(observed_at=stop, vector=[stop]))


def meta(first):
    return dict(selected={key: first[key] for key in ('sequence', 'started_at', 's', 'memory_s')},
                external_seed=106255, counters=dict(before=dict(updates=456), after=dict(updates=456)),
                native=dict(other_batch_actions=[]))


def test_actual_endpoint_not_manufactured_at_300():
    records = [row(2, 10., 310.5)]
    value = validate_chain(records)
    assert value['legal'] and value['reaches_300']
    assert not value['actual_300_state'] and value['endpoint_at_300'] is None
    assert value['first_endpoint_at_or_after_300']['clock'] == 310.5


def test_zero_dt_is_legal_but_does_not_extend_credit():
    records = [row(2, 10., 10.), row(3, 10., 12.)]
    value = validate_chain(records)
    assert value['legal'] and value['zero_duration_rows'] == 1
    assert value['span_seconds'] == 2 and not value['reaches_300']


def test_s_ns_and_memory_links_are_exact():
    records = [row(2, 10., 12.), row(3, 12., 15.)]
    assert validate_chain(records)['legal']
    records[1]['s'] = [12.01]
    assert not validate_chain(records)['checks']['exact_state_links']
    records[1]['s'] = [12.]
    records[1]['memory_s']['vector'] = [12.01]
    assert not validate_chain(records)['checks']['exact_memory_links']


def test_missing_writer_and_missing_expected_artifact_are_distinct():
    assert validate_chain(None, writer_expected=False)['status'] == 'not_expected'
    assert validate_chain(None)['status'] == 'missing'
    assert validate_chain([row()], source_kind='issue_trace')['status'] == 'unsupported_artifact'
    assert validate_chain([dict(s=[0], a=[0])])['status'] == 'invalid_schema'


def test_v55_raw_td_did_not_serialize_dynamic_memory():
    native = row()
    del native['memory_s'], native['memory_ns']
    result = validate_chain([native], source_kind='summary_first_transition')
    assert result['status'] == 'memory_not_saved'
    assert result['raw_physical_transition_available'] and not result['legal']
    assert not result['memory_writer_expected']


def test_ledger_reward_closure_and_clocks():
    event = dict(transition=row(),
                 ledger_s=dict(time_seconds=10., accrued_wait_seconds=5., accrued_in_vehicle_seconds=7.),
                 ledger_ns=dict(time_seconds=12., accrued_wait_seconds=6., accrued_in_vehicle_seconds=9.))
    assert validate_chain([event])['ledger']['verified']
    event['transition']['r'] = -3.
    assert not validate_chain([event])['legal']


def test_pair_support_is_root_not_fake_long_return():
    lo, hi = [row(action=0.)], [row(action=1.)]
    value = validate_pair(lo, hi, meta(lo[0]), meta(hi[0]))
    assert value['paired_root_supported'] and value['action_distinct']
    assert not value['both_chains_reach_300']
    assert not value['matched_long_horizon_soft_Q_return']
    high_meta = meta(hi[0]); high_meta['external_seed'] += 1
    assert not validate_pair(lo, hi, meta(lo[0]), high_meta)['paired_root_supported']


def test_training_changes_invalidate_paired_support():
    lo, hi = [row(action=0.)], [row(action=1.)]
    hm = meta(hi[0]); hm['counters']['after']['updates'] = 457
    assert not validate_pair(lo, hi, meta(lo[0]), hm)['paired_root_supported']


def test_inventory_distinguishes_unwritten_v55_chain(tmp_path):
    output = inventory(tmp_path, versions=('v55',))['v55']['stable']
    assert any(x['status'] == 'not_expected' and x['role'] == 'unwritten_event_chain'
               for x in output['artifacts'])
    assert any(x['status'] == 'missing' and x['writer_expected'] for x in output['artifacts'])
