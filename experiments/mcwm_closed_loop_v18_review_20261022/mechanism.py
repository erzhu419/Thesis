"""Recompute v18 mechanism diagnostics from the existing small summary only."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_state_response_v18_20261022/learning_summary.json'
value = json.loads(SOURCE.read_text())
rows, days = value['heldout_pairs'], value['days']
assert len(rows) == 24 and all(row['selection_status'] == 'selected' for row in rows)
headway = 'forward_minus_backward_headway_deviation'
by_day = {(day['seed'], day['scenario']): day for day in days}
coefficients, flips, windows, oracle = [], [], [], []
for day in days:
    shared = day['readouts']['shared']
    candidate = day['readouts']['candidate']
    old, new = shared['coefficients'], candidate['coefficients']
    coefficients.append(dict(seed=day['seed'], scenario=day['scenario'],
        wait_coefficients_exact=all(new['wait'][key] == old[key]['wait'] for key in new['wait']),
        shared_ride={key: item['ride'] for key, item in old.items()},
        candidate_ride=new['ride'], candidate_ride_design=candidate['design']['ride'],
        training_load_by_line=day['issue_load_training_by_line']))
for row in rows:
    shared, candidate = row['effects'][2:4]
    if (shared[2] > 0) == (candidate[2] > 0):
        continue
    models = by_day[(row['seed'], row['scenario'])]['readouts']
    old, new = models['shared']['coefficients'], models['candidate']['coefficients']['ride']
    line, features = row['line_id']+'_intercept', row['features']
    terms = dict(line_intercept_refit=new[line]-old[line]['ride'],
        headway_refit=(new[headway]-old[headway]['ride'])*features[3],
        added_load=new['onboard_capacity_ratio']*features[4])
    flips.append(dict(seed=row['seed'], scenario=row['scenario'], line_id=row['line_id'],
        window=row['window_name'], truth_wait_ride_cost=row['truth'][2][2][1:],
        shared_wait_ride_cost=shared[:3], candidate_wait_ride_cost=candidate[:3],
        cost_change=candidate[2]-shared[2], cost_change_components=terms,
        reconstruction_error=sum(terms.values())-(candidate[2]-shared[2]),
        regret_change=candidate[8]-shared[8], load_coverage=row['issue_load_coverage']))
for seed, scene, line in sorted({(r['seed'], r['scenario'], r['line_id']) for r in rows}):
    pair = sorted([r for r in rows if (r['seed'], r['scenario'], r['line_id']) == (seed, scene, line)], key=lambda r: r['window_name'])
    gains = dict(truth=[r['truth'][2][2][3] for r in pair],
        shared=[r['effects'][2][2] for r in pair], candidate=[r['effects'][3][2] for r in pair])
    windows.append(dict(seed=seed, scenario=scene, line_id=line, gains=gains,
        sign_flips={key: numbers[0]*numbers[1] < 0 for key, numbers in gains.items()}))
for scene in ('all', 'stable', 'local_speed_drop'):
    selected = [r for r in rows if scene == 'all' or r['scenario'] == scene]
    for name, index in (('shared', 2), ('candidate', 3)):
        for mode in ('native', 'truth_wait_only', 'truth_ride_only'):
            correct, regret = 0, 0.
            for row in selected:
                pred, truth = row['effects'][index], row['truth'][2][2]
                gain = pred[2] if mode == 'native' else 2*truth[1]+pred[1] if mode == 'truth_wait_only' else 2*pred[0]+truth[2]
                good = (gain > 0) == (truth[3] > 0)
                correct += good
                regret += 0. if good else abs(truth[3])
            oracle.append(dict(scenario=scene, model=name, mode=mode, n=len(selected),
                action_correct=correct, regret=regret))
output = dict(source=str(SOURCE.relative_to(HERE.parent)), input_significant_digits=8,
    wait_prediction_exact_count=sum(r['effects'][2][0] == r['effects'][3][0] for r in rows),
    cost_prediction_decreased=sum(r['effects'][3][2] < r['effects'][2][2] for r in rows),
    coefficients=coefficients, action_flips=flips, cross_window=windows,
    cross_window_flip_counts={key: sum(r['sign_flips'][key] for r in windows) for key in ('truth', 'shared', 'candidate')},
    single_head_truth_replacement=oracle,
    interpretation='Posthoc accounting/sensitivity diagnostics only: no fit, hyperparameter search, additional label collection, deployed arm or performance upper bound. Both heads can have cancelling errors; replacing one with truth can increase cost regret. Coefficient reconstruction uses the rounded existing summary.',
    sign='G=C(0)-C(7.5), C=2*wait+ride; choose7.5 iff G>0, else0.')
(HERE/'mechanism.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(output='mechanism.json', bytes=(HERE/'mechanism.json').stat().st_size,
    wait_exact=output['wait_prediction_exact_count'], action_flips=len(flips),
    cross_window_flip_counts=output['cross_window_flip_counts'])))
