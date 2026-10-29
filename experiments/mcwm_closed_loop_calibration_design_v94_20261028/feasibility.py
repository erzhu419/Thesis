"""Exposed-training stress test; never estimate a deployable calibration margin."""
from __future__ import annotations

import json
from pathlib import Path

from .calibration import seed_score, select_first


def analyze(model, probe):
    """Use saved fits only; first opportunities here are the old query subset."""
    first_ids = {row['id'] for scene in probe['scenarios'].values()
                 for row in scene['opportunities']}
    rows = []
    for index, source in enumerate(model['training_rows']):
        rows.append(dict(id=source['id'], scenario=source['scenario'],
            issue_clock=source['root']['started_at'],
            sequence=source['root']['sequence'], active=True,
            predictions={key: value['prediction'][index]
                         for key, value in model['models'].items()},
            delta=source['target']))
    first = [row for row in rows if row['id'] in first_ids]
    result = {}
    for name in model['models']:
        pressure = seed_score(first, name)
        negative = [row for row in first if row['predictions'][name] < 0]
        worst = max(negative, key=lambda row: row['delta']-row['predictions'][name])
        by_scene = {}
        for scene in probe['scenarios']:
            chosen = select_first([row for row in first if row['scenario'] == scene], name, pressure)
            by_scene[scene] = chosen['id'] if chosen else None
        result[name] = dict(
            first_query_roots=len(first), negative_first_query_roots=len(negative),
            largest_positive_residual=pressure, residual_source=worst['id'],
            minimum_saved_prediction=min(row['predictions'][name] for row in rows),
            original_first_query_choices={scene: data['models'][name]['selected_root']
                                          for scene, data in probe['scenarios'].items()},
            stress_accepted_roots=sum(row['predictions'][name]+pressure < 0 for row in rows),
            stress_first_query_choices=by_scene)
    return dict(
        scope='Exposed716207 training-query subset only; no independent calibration or full native policy replay',
        deployable_margin=None, independent_calibration_seeds=0,
        readouts=result,
        decision='Do not request the proposed global additive calibration budget',
        reason='The observed first-opportunity residual pressure suppresses every saved training proposal; it supplies no positive action-acceptance feasibility evidence',
        limitations='Future independent margins need not equal these training residuals. This resource decision does not prove independent failure or exclude other calibration designs.',
        actual_budget=dict(new_native_seconds=0, new_labels=0, model_solves=0,
                           optimizer_updates=0, remote_bytes_pulled=0))


def main():
    here = Path(__file__).resolve().parent
    source = here.parent/'mcwm_closed_loop_history_fit_v93_20261028'
    result = analyze(json.loads((source/'model.json').read_text()),
                     json.loads((source/'training_policy_probe.json').read_text()))
    with (here/'feasibility.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(decision=result['decision'], actual_budget=result['actual_budget'])))


if __name__ == '__main__':
    main()
