"""Read-only complete-day endpoint parity of the three reference suffixes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')


def audit_scene(root, scenario):
    rows, references = [], []
    for line in LINES:
        directory = root/'runs'/scenario/'pairs'/line
        summary_path = directory/'low'/'summary.json'
        root_path = directory/'root.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        declaration = json.loads(root_path.read_text()) if root_path.exists() else None
        row = dict(line_id=line, root_time=None, full_endpoint=None,
            executed_reference_action=None, declared_reference_action=None,
            missing_files=[str(path.relative_to(root)) for path in (summary_path, root_path)
                           if not path.exists()])
        if summary is None:
            row['checks'] = dict(reference_summary_present=False,
                                 declared_action_matches=False, scenario_metadata=False)
            rows.append(row)
            continue
        issue = summary['root']['issue_ledger']
        remaining = summary['remaining']
        wait = issue['accrued_wait_seconds']+remaining['wait']
        ride = issue['accrued_in_vehicle_seconds']+remaining['ride']
        reference = summary['reference']
        references.append(reference)
        expected_action = declaration['choice']['reference_action'] if declaration is not None else None
        row.update(root_time=summary['root']['started_at'],
            full_endpoint=dict(wait=wait, ride=ride, cost=2*wait+ride),
            executed_reference_action=summary['executed_action'],
            declared_reference_action=expected_action,
            reference_metadata=dict(scenario=reference['scenario'], seed=reference['seed'],
                source=reference['source'], source_arm=reference['source_arm'],
                packets=reference['packets'], published_clock=reference['published_clock']),
            checks=dict(reference_summary_present=True,
                declared_action_matches=declaration is not None
                    and summary['executed_action'] == expected_action
                    and summary['root'] == declaration['root'],
                scenario_metadata=reference['scenario'] == scenario
                    and reference['seed'] == 6207 and reference['source'] == 'v75'
                    and reference['source_arm'] == 'fresh' and summary['role'] == 'reference'
                    and summary['external_seed'] == 106207
                    and summary['root']['line_id'] == line))
        rows.append(row)
    complete = all(row['full_endpoint'] is not None for row in rows)
    spreads = {key:(max(row['full_endpoint'][key] for row in rows)
                   -min(row['full_endpoint'][key] for row in rows)) if complete else None
               for key in ('wait', 'ride', 'cost')}
    checks = dict(all_reference_endpoints_present=complete,
        all_declarations_present=all(not row['missing_files'] for row in rows),
        all_reference_actions_exact=all(row['checks']['declared_action_matches'] for row in rows),
        all_scenario_metadata_exact=all(row['checks']['scenario_metadata'] for row in rows),
        same_reference_snapshot=len(references) == 3
            and all(reference == references[0] for reference in references),
        complete_day_wait_ride_cost_equal=complete and all(value == 0 for value in spreads.values()))
    return dict(all_pass=all(checks.values()), checks=checks, rows=rows,
                max_minus_min=spreads, all_equal=checks['complete_day_wait_ride_cost_equal'])


def audit(root):
    root = Path(root)
    scenes = {scenario:audit_scene(root, scenario) for scenario in SCENARIOS}
    passed = all(value['all_pass'] for value in scenes.values())
    return dict(all_pass=passed, status='pass' if passed else 'hold', scenes=scenes,
        training_must_remain_on_hold=not passed,
        interpretation='All three low arms follow the same frozen reference trajectory; '
            'issue cumulative W/R plus remaining W/R must recover the identical full-day endpoint. '
            'A failure blocks using the action-difference labels for training readiness.',
        native_simulation_seconds=0, training_updates=0, weights_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.root)
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 8000:
        raise ValueError('Reference audit exceeds8KB; retain full branch data on server')
    with args.out.open('x') as output:
        output.write(raw)
    print(json.dumps(dict(all_pass=value['all_pass'], status=value['status'],
                          bytes=len(raw.encode()))))


if __name__ == '__main__':
    main()
