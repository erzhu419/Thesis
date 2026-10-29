import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_first import policy_source_audit, trace_audit


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def rows():
    return [dict(sequence=i, raw_s=[t], s=[float((np.asarray([t], np.float32)/24000)[0])],
        event=[0, 0, t], started_at=t+1, issue_ledger=dict(time_seconds=t+1),
        a=[-.5], proposed_a=[.3]) for i, t in enumerate((16863., 16864.))]


def test_original_detection_and_delivery_clocks_are_preserved(tmp_path):
    original = rows()
    actual = rows()
    actual[-1]['a'] = [-1.]
    source, branch = tmp_path/'source', tmp_path/'branch'
    write_rows(source, original)
    write_rows(branch, actual)
    result = trace_audit(branch, source, 1, [-.5], 24000.)
    assert all(result['checks'].values())
    assert result['delivery_minus_event_seconds'] == dict(min=1., max=1.)
    assert result['issue_count'] == 2


def test_changed_actor_clock_or_proposal_fails_exact_audit(tmp_path):
    source, branch = tmp_path/'source', tmp_path/'branch'
    write_rows(source, rows())
    changed = rows()
    changed[0]['s'][0] = float(np.float32(16864/24000))
    changed[1]['proposed_a'] = [.4]
    write_rows(branch, changed)
    result = trace_audit(branch, source, 1, [-.5], 24000.)
    assert result['checks']['full_prefix_exact'] is False
    assert result['checks']['native_event_clock_exact'] is False


def test_original_actor_clock_check_is_observational_only():
    result = policy_source_audit(HERE.parent/'mcwm_closed_loop_execution_phase_v19_20261022/phase_stream.py')
    assert result['actor_receives_original_state']
    assert result['no_state_clock_assignment']
