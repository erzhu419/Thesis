"""Commands align to native trip visits, rather than reused physical fleet IDs."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.mcwm_closed_loop_service_segments_v74_20261027.analyze import holding_comparison


def visit(trip, start, end):
    return (trip, '7X', 'dwell', 'A', 'A', 0), dict(kind='dwell',
        start={'clock_seconds': start}, end={'clock_seconds': end})


def issue(trip, clock, hold):
    return dict(vehicle_id='7X_fleet_0', trip_id=trip, line_id='7X', station_id='A',
        started_at=clock, a=[hold/30-1])


def write(tmp_path, name, rows):
    p = tmp_path/name
    p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return p


def test_reused_fleet_missing_control_trip_does_not_shift_next_trip_pair(tmp_path):
    segments = dict([visit('7X_0', 100, 120), visit('7X_1', 200, 220)])
    fresh = write(tmp_path, 'fresh.jsonl', [issue('7X_0', 110, 1), issue('7X_1', 210, 2)])
    baseline = write(tmp_path, 'baseline.jsonl', [issue('7X_1', 210, 3)])
    result = holding_comparison(fresh, baseline, segments, segments)
    assert result['matched'] == 1
    assert result['unmatched_fresh'] == 1 and result['unmatched_baseline'] == 0
    assert result['issue_clock_delta']['min'] == result['issue_clock_delta']['max'] == 0
    assert abs(result['executed_seconds_delta']['sum']+1) < 1e-12
    assert result['all_commands']['fresh']['count'] == 2
    assert abs(result['all_commands']['sum_difference_seconds']) < 1e-12


def test_same_trip_issue_outside_physical_visit_is_not_matched(tmp_path):
    segments = dict([visit('7X_0', 100, 120)])
    fresh = write(tmp_path, 'fresh.jsonl', [issue('7X_0', 130, 2)])
    baseline = write(tmp_path, 'baseline.jsonl', [issue('7X_0', 110, 2)])
    result = holding_comparison(fresh, baseline, segments, segments)
    assert result['matched'] == 0
    assert result['alignment_audit']['fresh']['outside_physical_dwell_issues'] == 1
    assert result['alignment_audit']['fresh']['physical_dwell_visits_without_issue'] == 1
    assert result['executed_seconds_delta']['sum'] is None
    assert result['all_commands']['fresh']['count'] == 1


def test_multiple_commands_in_visit_are_unscorable_instead_of_overwritten(tmp_path):
    segments = dict([visit('7X_0', 100, 120)])
    fresh = write(tmp_path, 'fresh.jsonl', [issue('7X_0', 110, 1), issue('7X_0', 111, 2)])
    baseline = write(tmp_path, 'baseline.jsonl', [issue('7X_0', 110, 2)])
    result = holding_comparison(fresh, baseline, segments, segments)
    assert result['matched'] == 0 and result['unmatched_baseline'] == 1
    audit = result['alignment_audit']['fresh']
    assert audit['duplicate_issue_visits'] == 1 and audit['issues_in_duplicate_visits'] == 2
    assert audit['single_issue_anchored_visits'] == 0
    assert result['all_commands']['fresh']['count'] == 2
    assert abs(result['all_commands']['fresh']['sum_seconds']-3) < 1e-12
