"""Weight original compact strata by their actual matched record counts."""
from collections import defaultdict
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_controlled_calibration_v10_20261021/calibration_summary.json'


def errors(groups, path, width):
    def value(group):
        for key in path:
            group = group[key]
        return group
    n = sum(value(g)['n'] for g in groups)
    return dict(n=n, **{metric: [sum(value(g)['n']*value(g)[metric][i]
        for g in groups if value(g)['n'])/n for i in range(width)] if n else None
        for metric in ('mae', 'mse')})


def gains(groups, path):
    values = []
    for group in groups:
        value = group
        for key in path:
            value = value.get(key, {})
        if value:
            values.append(value)
    n = sum(v[0] for v in values)
    return [n, *[sum(v[0]*v[i] for v in values if v[0])/n if n else None for i in (1, 2)]]


def group_metrics(groups):
    global_errors = {arm: errors(groups, ('fair_global_errors', arm), 2)
                     for arm in ('base_analytic', 'global58', 'endpoint_candidate')}
    if len({v['n'] for v in global_errors.values()}) != 1:
        raise ValueError('Global arms do not have the same matched sample counts')
    return dict(total_n=sum(g['n'] for g in groups), global_errors=global_errors,
        global_gain={arm: gains(groups, ('fair_global_gains', arm))
            for arm in ('global58_raw', 'endpoint_candidate_raw', 'global58_decoded', 'endpoint_candidate_decoded')},
        endpoint_errors={endpoint: errors(groups, ('endpoint_errors', endpoint), 4)
            for endpoint in ('global_next', 'same_vehicle_next', 'fixed300')},
        endpoint_gains={endpoint: gains(groups, ('predicted_gains', 'endpoint_'+endpoint))
            for endpoint in ('same_vehicle_next', 'fixed300')})


def main():
    source = json.loads(SOURCE.read_text())
    strata = source['strata']
    grouped = defaultdict(list)
    for row in strata:
        grouped[row['scenario'], row['phase'], row['duration'], 'all'].append(row)
        grouped[row['scenario'], row['phase'], row['duration'], row['line']].append(row)
    metrics = [dict(scenario=s, phase=p, duration=d, line=line, **group_metrics(rows))
               for (s, p, d, line), rows in sorted(grouped.items())]
    heldout = [g for g in strata if g['phase'] == 'heldout_after' and g['duration'] == 'positive_dt']
    zeros = [g for g in strata if g['duration'] == 'zero_dt']
    overall = dict(heldout_positive=group_metrics(heldout), zero_duration=group_metrics(zeros))
    coverage = []
    for cell in source['cells']:
        for line in ('7X', '102X', '705X'):
            coverage.append(dict(seed=cell['seed'], scenario=cell['scenario'], line=line,
                train_low=cell['selected_train_coverage'][line]['low'],
                train_high=cell['selected_train_coverage'][line]['high'],
                positive_global_low=cell['selected_positive_global_coverage'][line]['low'],
                positive_global_high=cell['selected_positive_global_coverage'][line]['high'],
                heldout_low=cell['complete_heldout_coverage'][line]['low'],
                heldout_high=cell['complete_heldout_coverage'][line]['high']))
    result = dict(source=str(SOURCE), status=source['status'], integrity=source['integrity'],
        original_gate=source['independent_confirmation'], overall=overall, groups=metrics,
        decision='Do not advance to independent confirmation; retain trust and rebuild the response readout in development.',
        aggregation='Record-weighted original matched MAE/MSE; each raw/decoded gain uses its own matched n. No missing outcomes imputed.')
    (HERE/'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (HERE/'coverage.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    flat = []
    for group in metrics:
        record = {k: group[k] for k in ('scenario', 'phase', 'duration', 'line', 'total_n')}
        for arm, value in group['global_errors'].items():
            for key, metric in (('dt_mae', ('mae', 0)), ('cost_mae', ('mae', 1)), ('cost_mse', ('mse', 1))):
                record[arm+'_'+key] = value[metric[0]][metric[1]] if value['n'] else None
            record[arm+'_n'] = value['n']
        for name, value in group['global_gain'].items():
            record[name+'_n'], record[name+'_mean_abs'] = value[0], value[2]
        for endpoint, value in group['endpoint_errors'].items():
            record[endpoint+'_n'] = value['n']
            record[endpoint+'_dt_mae'] = value['mae'][0] if value['n'] else None
            record[endpoint+'_cost_mae'] = value['mae'][3] if value['n'] else None
        flat.append(record)
    with (HERE/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
