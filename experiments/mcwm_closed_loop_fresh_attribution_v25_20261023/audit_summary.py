"""Extract four passive factual issue audits into one small, lossless summary."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

LINES = ('7X', '102X', '705X')
SCENARIOS = ('stable', 'local_speed_drop')
ARMS = ('fresh', 'no_fresh')
JOINT_KEYS = tuple(f'{sign}|{bucket}' for sign in (-1, 0, 1)
    for bucket in ('below_minus_hold', 'within_hold', 'above_hold'))
WINDOW_COLUMNS = ('line', 'name', 'start', 'end', 'issues', 'action_supported', 'action_unsupported',
    'history_supported', 'history_missing', 'history_noncausal', 'history_feature_missing',
    'joint_eligible', 'bins_agree', 'third_joint_counts', 'mean_joint_counts',
    'raw_headway_range', 'history_mean_seconds_range')


def summarize(root):
    root = Path(root)
    rows, columns = [], None
    for scenario in SCENARIOS:
        for arm in ARMS:
            path = root/'runs'/scenario/'seed6201'/arm/'summary.json'
            key = dict(scenario=scenario, seed=6201, arm=arm)
            if not path.exists():
                rows.append(dict(**key, status='missing'))
                continue
            day = json.loads(path.read_text())
            audit = day.get('issue_audit')
            if audit is None:
                rows.append(dict(**key, status='missing_issue_audit', run_status=day.get('status')))
                continue
            if columns is None:
                columns = audit['mapping_columns']
            elif audit['mapping_columns'] != columns:
                raise ValueError('Four arms use different execution-audit columns')
            windows = []
            for window in audit['windows']:
                if window['line'] not in LINES:
                    continue
                encoded = []
                for field in WINDOW_COLUMNS:
                    value = window[field]
                    if field in ('third_joint_counts', 'mean_joint_counts'):
                        if not set(value) <= set(JOINT_KEYS):
                            raise ValueError('Unexpected joint stratum in factual audit')
                        value = [value.get(name, 0) for name in JOINT_KEYS]
                    encoded.append(value)
                windows.append(encoded)
            eligible = sum(window[WINDOW_COLUMNS.index('joint_eligible')] for window in windows)
            agree = sum(window[WINDOW_COLUMNS.index('bins_agree')] for window in windows)
            rows.append(dict(**key, status=audit['status'], run_status=day.get('status'),
                training_rows=audit['training_rows'], environment_queries=audit['environment_queries'],
                retained_full_states=audit['retained_full_states'],
                mapping=[row for row in audit['mapping'] if row['line'] in LINES], windows=windows,
                bin_comparison=dict(joint_eligible_issues=eligible, same_bin_issues=agree,
                    different_bin_issues=eligible-agree)))
    present = [row for row in rows if 'mapping' in row]
    result = dict(status='complete' if len(present) == 4 else 'incomplete',
        mapping_columns=columns, window_columns=list(WINDOW_COLUMNS), joint_keys=list(JOINT_KEYS),
        audits=rows, checks=dict(four_audits_present=len(present) == 4,
            four_runs_complete=len(present) == 4 and all(row['run_status'] == 'complete' for row in present),
            diagnostic_not_fitted=len(present) == 4 and all(row['training_rows'] == 0 for row in present),
            no_environment_queries=len(present) == 4 and all(row['environment_queries'] == 0 for row in present)),
        semantics=dict(window_encoding='Rows follow window_columns. The two joint count arrays follow joint_keys; all counters and ranges are preserved.',
            coverage='Complete factual-window census; no stop after first selection. Counts include explicit unsupported actions and missing/noncausal histories.',
            strata='Third segment residual reproduces the old sampling definition; three-segment mean is the actual history5 input, with identical +/-7.5-second bins. Same/different-bin counts compare the same eligible issues.',
            use='Passive diagnostic only, neither label feedback nor a service-reversal admission gate. No old trace, CSV, checkpoint or counterfactual query is read.'))
    if len(json.dumps(result, separators=(',', ':'), allow_nan=False).encode()) > 30000:
        raise ValueError('Combined factual audit exceeds30KB small-summary limit')
    return result


def _self_test():
    """Synthetic complete nine-window audits; no native process or existing run."""
    mapping = [dict(line=line, phase=phase, values=[123456, 0, 1.2345678901234567])
        for line in (*LINES, 'ignored_line') for phase in ('pre_floor', 'reserve_floor', 'tail')]
    windows = [dict(line=line, name=name, start=1200., end=17670., issues=123, action_supported=122,
        action_unsupported=1, history_supported=120, history_missing=2, history_noncausal=1,
        history_feature_missing=0, joint_eligible=119, bins_agree=100,
        third_joint_counts={key:i+1 for i, key in enumerate(JOINT_KEYS)},
        mean_joint_counts={key:i+2 for i, key in enumerate(JOINT_KEYS)},
        raw_headway_range=[-1.2345678901234567, 9.876543210987654],
        history_mean_seconds_range=[-111.23456789012345, 222.34567890123456])
        for line in LINES for name in ('bootstrap', 'response', 'late')]
    audit = dict(status='complete', training_rows=0, environment_queries=0, retained_full_states=0,
        mapping_columns=['issues', 'floor_condition', 'example_seconds'], mapping=mapping,
        windows=windows, semantics=dict(large_repeated_text='not exported'))
    with tempfile.TemporaryDirectory(prefix='v25-audit-summary-') as temp:
        root = Path(temp)
        for scenario in SCENARIOS:
            for arm in ARMS:
                target = root/'runs'/scenario/'seed6201'/arm/'summary.json'
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps(dict(status='complete', issue_audit=audit)))
        result = summarize(root)
        assert all(result['checks'].values()) and len(result['audits']) == 4
        assert all(len(row['mapping']) == 9 and len(row['windows']) == 9 for row in result['audits'])
        decoded = dict(zip(result['window_columns'], result['audits'][0]['windows'][0]))
        for field in ('third_joint_counts', 'mean_joint_counts'):
            decoded[field] = {key:value for key, value in zip(result['joint_keys'], decoded[field]) if value}
        assert decoded == windows[0]
        assert 'large_repeated_text' not in json.dumps(result)
        damaged = deepcopy(audit)
        damaged['training_rows'] = 1
        (root/'runs/stable/seed6201/fresh/summary.json').write_text(json.dumps(dict(status='complete', issue_audit=damaged)))
        assert summarize(root)['checks']['diagnostic_not_fitted'] is False
        return dict(status='passed', bytes=len(json.dumps(result, separators=(',', ':')).encode()),
            audits=4, windows=36, mapping_rows=36)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(_self_test() if args.self_test else summarize(args.root), separators=(',', ':'), allow_nan=False))
