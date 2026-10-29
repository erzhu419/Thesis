"""Read resident context logs; emit only timing and physical-coverage statistics."""
from pathlib import Path
import json

ROOT = Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_closed_loop_critic_memory_v32_20261023')
FOCUS = ('7X', '102X', '705X')


def audit():
    cases = []
    for scenario in ('stable', 'local_speed_drop'):
        rows = [json.loads(line) for line in (ROOT/'runs'/'seed6202'/scenario/'memory_training_rows.jsonl').read_text().splitlines()]
        checks = dict(s_clock_exact=0, ns_clock_exact=0, all_observations_past_or_present=0,
                      fixed_schema=0, command_excluded=0)
        cold, windows = [], {}
        for row in rows:
            left, right = row['memory_s'], row['memory_ns']
            checks['s_clock_exact'] += left['observed_at'] == row['issue']
            checks['ns_clock_exact'] += right['observed_at'] == row['completed']
            causal = True
            for snap in (left, right):
                for key in ('latest_native_observation_at', 'latest_recent_apply_at', 'latest_recent_physical_observation_at'):
                    value = snap['metadata'][key]
                    causal &= value is None or value <= snap['observed_at']
            checks['all_observations_past_or_present'] += causal
            checks['fixed_schema'] += left['line_ids']==right['line_ids'] and len(left['vector'])==len(right['vector'])==84
            checks['command_excluded'] += all(s['metadata']['command_is_feature'] is False and s['metadata']['selected_diagnostic_vehicle'] is False for s in (left,right))
            own = next(line for line in left['lines'] if line['line_id']==row['line_id'])
            if not own['available']:
                cold.append(dict(line_id=row['line_id'], sequence=row['sequence'], issue=row['issue'],
                                 completed=row['completed'], ns_available=next(line for line in right['lines'] if line['line_id']==row['line_id'])['available']))
            if row['line_id'] not in FOCUS:
                continue
            window = 'before9000' if row['issue']<9000 else '9000-12000' if row['issue']<12000 else '12000-18000'
            key = row['line_id']+'/'+window
            values = own['vector']
            selected = [values[i] for i in (3,4,5,6)]
            if key not in windows:
                windows[key] = dict(rows=0, stopped_present=0, departed_present=0, physical_observed=0,
                                    min=selected.copy(), max=selected.copy())
            v = windows[key]
            v['rows'] += 1
            v['stopped_present'] += own['stopped_count']>0
            v['departed_present'] += own['departure_observed_count']>0
            v['physical_observed'] += own['phase_observed_count']>0
            v['min'] = [min(a,b) for a,b in zip(v['min'], selected)]
            v['max'] = [max(a,b) for a,b in zip(v['max'], selected)]
        cases.append(dict(scenario=scenario, rows=len(rows), timing_checks=checks,
            all_pass=len(rows)==285 and all(n==285 for n in checks.values()),
            zero_dt_admitted=sum(r['dt']==0 for r in rows), cold_start=cold, windows=windows))
    return dict(all_pass=all(c['all_pass'] for c in cases), cases=cases,
        range_columns=['stopped_fraction','departure_observed_fraction','apply_age_over300','observed_departure_age_over300'])


if __name__ == '__main__':
    print(json.dumps(audit(),separators=(',',':'),allow_nan=False))
