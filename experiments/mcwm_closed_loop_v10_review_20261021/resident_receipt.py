"""Read resident v10 summaries/logs; return compact evidence, without simulation."""
import argparse
from collections import Counter
import json
from pathlib import Path


def bounds(values):
    return dict(n=len(values), minimum=min(values) if values else None,
                maximum=max(values) if values else None)


def inspect(root):
    cells, values = [], {}
    for seed in (6201, 6202):
        for scenario in ('stable', 'local_speed_drop'):
            task = {(6201, 'stable'): 't208468', (6201, 'local_speed_drop'): 't208469',
                    (6202, 'stable'): 't208470', (6202, 'local_speed_drop'): 't208471'}[seed, scenario]
            directory = root/'runs'/scenario/f'seed{seed}'
            v = json.loads((directory/'summary.json').read_text())
            protocol = json.loads((directory/'protocol.json').read_text())
            values[seed, scenario] = v
            records, evaluations = v['controlled_records'], v['evaluations']
            last = v['receipts'][-1]
            freeze = v['budget']['training_stopped_at']
            heldout = [r for r in records if r['role'] == 'heldout']
            assignments = {r['sequence']: r for r in v['calibration']['assignments']}
            log = Path('/tmp')/f'sched_{task}.log'
            lines = log.read_text(errors='replace').splitlines() if log.is_file() else []
            failures = [line for line in lines if 'WARNING: Failed to apply holding' in line]
            controlled_trips = {r['trip_id'] for r in records}
            controlled_failures = [line for line in failures if any(trip in line for trip in controlled_trips)]
            prediction_ranges = {}
            for line in ('7X', '102X', '705X'):
                rows = [r for r in evaluations if r['role'] == 'heldout' and r['line_id'] == line]
                by_endpoint = {}
                for endpoint in ('global_next', 'same_vehicle_next', 'fixed300'):
                    predicted = []
                    for row in rows:
                        chosen = 'low' if abs(row['hold_seconds']) < 1e-5 else 'high'
                        branch = row['endpoint_prediction'].get(chosen, {})
                        outcome = branch.get('endpoints', {}).get(endpoint)
                        if outcome is not None:
                            predicted.append(outcome)
                    by_endpoint[endpoint] = dict(
                        duration=bounds([r['dt'] for r in predicted]),
                        cost=bounds([r['cost'] for r in predicted]),
                        negative_costs=sum(r['cost'] < 0 for r in predicted),
                        nonpositive_durations=sum(r['dt'] <= 0 for r in predicted))
                prediction_ranges[line] = by_endpoint
            final_contrasts = {}
            for endpoint, weights in v['endpoint_model']['coefficients'].items():
                final_contrasts[endpoint] = {}
                for i, line in enumerate(('7X', '102X', '705X')):
                    weight = weights[3+i]
                    final_contrasts[endpoint][line] = dict(
                        duration_gain_low_minus_high=-2*300*weight[0],
                        cost_gain_low_minus_high=-2*10000*(2*weight[1]+weight[2]))
            all_ids = [seq for receipt in v['receipts'] for seq in receipt['sequence_ids']]
            cells.append(dict(seed=seed, scenario=scenario, task=task,
                external_seed=seed+protocol['protocol']['target_seed_offset'],
                initializer=protocol['initializer'],
                last_packet={k: last[k] for k in ('deadline', 'processed_at', 'max_available_at', 'records')},
                packets_processed_after_18000=sum(r['processed_at'] > 18000 for r in v['receipts']),
                last_training_clock=freeze,
                heldout_issue_times=[r['started_at'] for r in heldout],
                heldout_before_last_update=[r['sequence'] for r in heldout if r['started_at'] < freeze],
                heldout_at_last_update=[r['sequence'] for r in heldout if r['started_at'] == freeze],
                unique_training_slots=len(set(all_ids)), total_training_slots=len(all_ids),
                selected_controlled=len([r for r in records if r['selected_for_training']]),
                target_to_record_hold_max_error=max(abs(r['hold_seconds']-r['assignment']['target_hold_seconds']) for r in records),
                command_to_record_hold_max_error=max(abs(30*(assignments[r['sequence']]['action'][0]+1)-r['hold_seconds']) for r in records),
                all_jointly_executable=all(r['jointly_executable'] for r in assignments.values()),
                hold_values=sorted({r['hold_seconds'] for r in records}),
                native_log_exists=log.is_file(), native_hold_failures=len(failures),
                controlled_trip_hold_failures=len(controlled_failures), failure_examples=controlled_failures[:3],
                issue_delivery_lag=bounds([r['issue_delivery_lag_seconds'] for r in records]),
                fixed300_true_duration=bounds([r['endpoints']['fixed300']['dt'] for r in records]),
                heldout_prediction_ranges=prediction_ranges, final_contrasts=final_contrasts))
    prefix_checks = []
    for seed in (6201, 6202):
        stable, shock = (values[seed, s] for s in ('stable', 'local_speed_drop'))
        receipts = [[r for r in v['receipts'] if r['deadline'] < 9000] for v in (stable, shock)]
        def issue_fields(v):
            return [{k: r[k] for k in ('sequence', 'line_id', 'vehicle_id', 'trip_id', 'station_id',
                    'started_at', 'hold_seconds', 'role', 'assignment')}
                    for r in v['controlled_records'] if r['started_at'] < 9000]
        def prediction_fields(v):
            return [{k: r[k] for k in ('sequence', 'issued_at', 'line_id', 'role',
                    'hold_seconds', 'endpoint_prediction')}
                    for r in v['evaluations'] if r['issued_at'] < 9000]
        prefix_checks.append(dict(seed=seed,
            before_shock_feedback_packets=len(receipts[0]),
            before_shock_feedback_slots=sum(r['records'] for r in receipts[0]),
            feedback_receipts_exact=receipts[0] == receipts[1],
            controlled_issues_exact=issue_fields(stable) == issue_fields(shock),
            issue_predictions_exact=prediction_fields(stable) == prediction_fields(shock),
            before_shock_controlled_issues=len(issue_fields(stable))))
    return dict(cells=cells, stable_shock_before9000=prefix_checks,
                evidence='Resident summary/log reads only; no simulator, training, checkpoints or raw trajectories')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), ensure_ascii=False, separators=(',', ':'), allow_nan=False))
