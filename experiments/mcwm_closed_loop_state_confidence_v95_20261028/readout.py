"""One exposed-data feasibility readout; no deployable confidence model is fitted."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_state_confidence_v95_20261028.confidence import fit_bias, predict, seed_score


def write(name, value):
    with (HERE/name).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def run():
    protocol = json.loads((HERE/'protocol.json').read_text())
    if (HERE/'readout_attempt.json').exists() or (HERE/'readout.json').exists():
        raise RuntimeError('Inspect the saved readout; never repeat the three diagnostic fits')
    artifact = json.loads(Path(protocol['source_model']).read_text())
    probe = json.loads(Path(protocol['source_probe']).read_text())
    name = protocol['primary_model']
    rows = [dict(id=source['id'], line=source['line_id'], scenario=source['scenario'],
        q=source['features'][2], m=artifact['models'][name]['prediction'][index],
        prediction=artifact['models'][name]['prediction'][index], delta=source['target'],
        active=True, issue_clock=source['root']['started_at'], sequence=source['root']['sequence'])
        for index, source in enumerate(artifact['training_rows'])]
    if len(rows) != 12 or protocol['current_exposed_check']['fits'] != 3:
        raise RuntimeError('Only the frozen twelve-row, three-fold diagnostic is authorized')
    write('readout_attempt.json', dict(fits=3, native_seconds=0, independent_model=False))
    folds, evaluated = {}, []
    for number, line in enumerate(('7X', '102X', '705X'), 1):
        training = [row for row in rows if row['line'] != line]
        held = [row for row in rows if row['line'] == line]
        model = fit_bias(training, protocol['q_range_for_scaling_only'])
        pressure = seed_score(training, model)
        folds[line] = dict(exposed_bias=model, training_pressure_not_independent_C=pressure,
            training_ids=[row['id'] for row in training], held_ids=[row['id'] for row in held])
        for row in held:
            parts = predict(model, row['q'], row['m'])
            upper = parts['center']+pressure*parts['support_scale']
            evaluated.append(dict(row, b=parts['bias'], C=pressure, u=parts['support_scale'],
                support_inflation=pressure*(parts['support_scale']-1.), U=upper,
                eligible_after_bound=row['m'] < 0. and upper < 0.))
        print(json.dumps(dict(progress=f'{number}/3', held_line=line, rank=model['rank'])), flush=True)
    by_id = {row['id']:row for row in evaluated}
    scenes, selected, selected_beneficial = {}, [], 0
    for scene, saved in probe['scenarios'].items():
        first = sorted([by_id[item['id']] for item in saved['opportunities']],
                       key=lambda row:(row['issue_clock'], row['sequence']))
        chosen = next((row for row in first if row['eligible_after_bound']), None)
        cost = chosen['delta'] if chosen else 0.
        selected.append(cost)
        selected_beneficial += bool(chosen and cost < 0.)
        scenes[scene] = dict(opportunities=[row['id'] for row in first],
            cross_fold_diagnostic_selected=chosen['id'] if chosen else None,
            selected_delta_per_passenger=cost,
            raw_history6_selected=saved['models'][name]['selected_root'],
            raw_history6_delta=saved['models'][name]['delta_cost_per_passenger'])
    first_ids = {value for scene in scenes.values() for value in scene['opportunities']}
    accepted = [row for row in evaluated if row['eligible_after_bound']]
    checks = dict(three_rank2_fits=len(folds) == 3 and all(item['exposed_bias']['rank'] == 2 for item in folds.values()),
        useful_first_query_selected=selected_beneficial > 0,
        selected_mean_below_reference=sum(selected)/len(selected) < 0.,
        no_positive_scene_delta=all(value <= 0. for value in selected),
        nonzero_intervention=any(scene['cross_fold_diagnostic_selected'] is not None for scene in scenes.values()))
    passed = all(checks.values())
    result = dict(status='exposed_feasibility_pass' if passed else 'close_fixed_candidate',
        scope=protocol['current_exposed_check'], checks=checks, folds=folds, rows=evaluated,
        exposed_point_counts=dict(roots=len(rows), accepted=len(accepted),
            accepted_beneficial=sum(row['delta'] < 0. for row in accepted),
            accepted_harmful=sum(row['delta'] > 0. for row in accepted),
            accepted_first_query=sum(row['id'] in first_ids for row in accepted)),
        cross_fold_diagnostic=dict(scenes=scenes, selected_mean_delta=sum(selected)/len(selected),
            raw_history6_mean_delta=sum(scene['raw_history6_delta'] for scene in scenes.values())/len(scenes),
            deployable_single_model=False, independent_calibration=False, actual_policy_replay=False),
        actual_budget=dict(native_seconds=0, new_labels=0, independent_bias_fits=0,
            exposed_diagnostic_fits=3, optimizer_updates=0, initialization_steps=0, remote_bytes_pulled=0),
        decision='Consider only the frozen two-seed T collection after review; no collection submitted'
            if passed else 'Close this fixed design; no native collection, threshold/ridge search or extra diagnostic fits')
    write('readout.json', result)
    write('decision.json', dict(status=result['status'], checks=checks, decision=result['decision'],
        actual_budget=result['actual_budget'], deployable_model=None, calibrated_margin=None,
        v87_truth_used=0, reserved816207_used=0))
    print(json.dumps(dict(status=result['status'], actual_budget=result['actual_budget'])), flush=True)


if __name__ == '__main__':
    run()
