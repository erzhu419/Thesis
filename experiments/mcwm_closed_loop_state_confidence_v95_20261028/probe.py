"""Apply each saved diagnostic rule separately, without choosing a winning fold."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_state_confidence_v95_20261028.confidence import select_first


def analyze(report):
    rows = {row['id']:row for row in report['rows']}
    results = {}
    for line, fold in report['folds'].items():
        scenes = {}
        for scene, entry in report['cross_fold_diagnostic']['scenes'].items():
            selected = select_first([rows[key] for key in entry['opportunities']],
                fold['exposed_bias'], fold['training_pressure_not_independent_C'])
            scenes[scene] = dict(selected=selected['id'] if selected else None,
                                 selected_delta=selected['delta'] if selected else 0.)
        results[line] = dict(scenes=scenes,
            mean_delta=sum(value['selected_delta'] for value in scenes.values())/len(scenes))
    return dict(scope='Apply each saved fold rule separately to the old first-query subset. '
        'No choosing the best fold; every result is reported. These rules used exposed old '
        'labels and are not the new independently trained model.', rules=results,
        additional_fits=0, new_native_seconds=0, new_labels=0)


if __name__ == '__main__':
    result = analyze(json.loads((HERE/'readout.json').read_text()))
    with (HERE/'single_rule_probe.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
