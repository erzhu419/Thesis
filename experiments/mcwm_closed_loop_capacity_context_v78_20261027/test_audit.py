import json
from pathlib import Path

from audit import PASSENGER_FILES, V75, V76, _ride_source, audit_sources


def test_archived_clock_difference_does_not_manufacture_native_context(tmp_path):
    runtime = tmp_path/'sumo_eval/runtime/H2Oplus/bus_h2o'
    runtime.mkdir(parents=True)
    (runtime/'mdp_contract_v4.py').write_text(
        "RAW_EVENT_STATE_NAMES=('line_id','station_index','sim_time_seconds')\n")
    for seed in (6208, 6207):
        for arm in ('fresh', 'no_fresh'):
            folder = tmp_path/V75/f'runs/executed_cql/validation/seed{seed}/stable'/arm
            folder.mkdir(parents=True)
            rows = [dict(started_at=t, event=[0, 0, t-1], s=[.5], a=[0],
                         issue_ledger={'time_seconds': t}) for t in (11999, 12000, 16199, 16200)]
            (folder/'prefix_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            receipt = tmp_path/V76/f'runs/seed{seed}'/arm/'prefix_summary.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps(dict(status='complete',
                integrity=dict(all_pass=True, checks=dict(full_prefix_exact=True)))))
    out = audit_sources(tmp_path)
    assert out['window_issue_rows'] == 8
    assert out['missing_joint_context'] == 8
    assert out['joint_context_observations'] == 0
    assert out['native_joint_coverage_verified'] is False
    assert out['historical_joint_available'] is False
    assert out['service_timestamp_joins'] == 0
    assert out['final_state_reads'] == 0
    for row in out['archived_prefixes']:
        assert row['started_minus_event_clock']['values'] == [dict(seconds=1., count=2)]
        assert row['field_coverage']['context_z'] == 0
    assert all(row['checks']['full_prefix_exact'] for row in out['v76_restoration_receipts'])
    assert out['issue_provenance']['same_timestamp_proves_preaction'] is False


def test_static_route_attributes_are_counted_without_becoming_native_stage_evidence(tmp_path):
    path = tmp_path/PASSENGER_FILES[0]
    path.write_text('<routes><person id="private"><ride busStop="s" lines="7X 705X"/>'
                    '<ride busStop="t" lines="ANY"/><ride lines="7X"/></person></routes>')
    out = _ride_source(path)
    assert out['rides'] == 3
    assert out['rides_with_busStop'] == 2
    assert out['rides_with_explicit_lines'] == 2
    assert out['rides_with_both'] == 1
    assert out['every_ride_explicit'] is False
    assert 'private' not in json.dumps(out)
