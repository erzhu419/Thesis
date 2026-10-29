"""Read-only frozen-response and actor/F audit before v96 native collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import torch

from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference, V84


def audit():
    protocol = json.loads((HERE/'protocol.json').read_text())
    source = json.loads((HERE/'source_protocol.json').read_text())
    model = json.loads((HERE/'source_model.json').read_text())
    seeds = [309177151, 1476154684]
    scenes = ['stable', 'local_speed_drop']
    checks = dict(
        initializer_6207=protocol['source_initializer'] == source['frozen_reference']['initializer_seed']
            == model['reference_contract']['initializer_seed'] == 6207,
        frozen_T_seeds=protocol['external_seeds'] == source['independent_cohorts']['T'] == seeds,
        frozen_scenes=protocol['scenarios'] == source['frozen_reference']['scenarios'] == scenes,
        frozen_reference=protocol['frozen_reference'] == source['frozen_reference'] == model['reference_contract'],
        frozen_common_support=protocol['frozen_common_support'] == source['frozen_common_support'] == model['common_support'],
        frozen_q_scaling=protocol['q_range_for_scaling_only'] == source['q_range_for_scaling_only'],
        complete_v95_T_contract=protocol['analysis']['T_integrity_gate'] == source['model']['T_integrity_gate'],
        frozen_T_budget=protocol['budget']['total_cap'] == source['proposed_budget']['T']['max_native_seconds'] == 338400
            and protocol['budget']['expected_pairs'] == source['proposed_budget']['T']['max_pairs'] == 12
            and protocol['budget']['expected_physical_labels'] == source['proposed_budget']['T']['max_physical_labels'] == 24,
        frozen_native_endpoint=protocol['native_contract']['true_endpoint'] == 24000
            and protocol['native_contract']['label_available_at'] == 24030,
        frozen_collection_only=protocol['budget']['fit_count'] == protocol['budget']['calibration_validation_queries'] == 0
            and protocol['native_contract']['suffix_training_updates'] == protocol['native_contract']['admitted_labels'] == 0,
        zero_update_budget=all(protocol['budget'][key] == 0 for key in
            ('initializer_updates', 'actor_updates', 'critic_updates', 'V_updates',
             'alpha_updates', 'world_updates', 'trust_updates')),
        frozen_execution_contract=protocol['native_contract']['F'] == 'unchanged/applied once'
            and protocol['native_contract']['trust'] == 'unchanged'
            and protocol['native_contract']['actor_observation'] == protocol['native_contract']['V_observation'] == 229,
        source_model_gate=model['integrity']['all_pass'] and all(model['integrity']['checks'].values()),
        original_three_fits=model['budget']['model_solves'] == 3,
        fixed_response_units=model['ridge'] == 1. and model['denominator'] == 15079. and model['hold_scale'] == 7.5,
        source_model_version=model['name'] == 'mcwm_closed_loop_history_fit_v93_20261028')
    errors = {}
    for name, width in (('line3', 3), ('service5', 5), ('service_history6', 6)):
        matrix = np.asarray(model['design']['matrices'][name], dtype=np.float64)
        coef = np.asarray(model['models'][name]['coef'], dtype=np.float64)
        saved = np.asarray(model['models'][name]['prediction'], dtype=np.float64)
        shapes = matrix.shape == (12, width) and coef.shape == (width,) and saved.shape == (12,)
        finite = bool(np.isfinite(matrix).all() and np.isfinite(coef).all() and np.isfinite(saved).all())
        error = float(np.max(np.abs(matrix@coef-saved))) if shapes and finite else None
        errors[name] = error
        checks[name+'/frozen_forward'] = bool(error is not None and error <= 1e-12)

    paths, roots = {}, 0
    torch.set_num_threads(1)
    v85 = HERE.parent/'mcwm_closed_loop_cost_advantage_v85_20261027'
    for scene in scenes:
        controller, _mdp, _stations, metadata = load_reference(scene)
        paths[scene] = metadata['source_actor_path']
        checks[scene+'/reference_actor_source'] = paths[scene] == model['reference_source_paths'][scene]
        checks[scene+'/saved_initialization'] = metadata['source_training_seed'] == 6207 and metadata['source_offline_updates'] == 50000
        for cohort, directory in (('v84', V84), ('v85', v85)):
            for line in ('7X', '102X', '705X'):
                row = json.loads((directory/'runs'/scene/'pairs'/line/'root.json').read_text())
                action = controller(np.asarray(row['root']['s'], dtype=np.float32))
                checks[cohort+'/'+scene+'/'+line] = bool(
                    np.array_equal(action, np.asarray(row['choice']['reference_action'], dtype=np.float32))
                    and np.array_equal(controller.last_proposal,
                        np.asarray(row['root']['reference_proposal'], dtype=np.float32)))
                roots += 1
    checks['twelve_original_issue_inputs'] = roots == 12
    return dict(all_pass=all(checks.values()), checks=checks,
        source_initializer=6207, external_seeds=seeds, scenarios=scenes,
        source_reference_paths=paths, reference_source_paths=paths, reference_roots=roots,
        source_model_path=str(HERE/'source_model.json'), source_protocol_path=str(HERE/'source_protocol.json'),
        original_training_fits=3, forward_max_absolute_errors=errors,
        new_fits=0, new_neural_updates=0, new_suffixes=0, new_labels=0,
        v87_truth_used=0, native_seconds=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    raw = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 16384:
        raise ValueError('Freeze audit exceeds the declared16KB limit')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(raw)
    print(raw, end='', flush=True)
    if not result['all_pass']:
        raise RuntimeError('Frozen response/actor/F audit failed; no native collection')


if __name__ == '__main__':
    main()
