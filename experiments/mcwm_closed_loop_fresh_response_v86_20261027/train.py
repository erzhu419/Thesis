"""Fit fixed old/new six-pair cohorts before opening the new heldout episodes."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_fresh_response_v86_20261027.model import (
    COHORTS, DENOMINATOR, LINES, SCENARIOS, state_feature, train,
)

V84 = HERE.parent/'mcwm_closed_loop_reference_collection_v84_20261027'
V85 = HERE.parent/'mcwm_closed_loop_cost_advantage_v85_20261027'
SOURCES = dict(stale=V84, fresh=V85)
ROLES = dict(low='reference', high='candidate')
ROOT_KEYS = ('started_at', 'sequence', 'trip_id', 'station_id')
COST_KEYS = ('wait', 'ride', 'cost')


def _read(path):
    return json.loads(path.read_text())


def _load_cohort(arm):
    """Read the shared native pair schema, never the heldout evaluation reader."""
    source = SOURCES[arm]
    scope, external_seed = COHORTS[arm]
    records, references = [], {}
    for scenario in SCENARIOS:
        for line in LINES:
            directory = source/'runs'/scenario/'pairs'/line
            declaration = _read(directory/'root.json')
            pair = _read(directory/'pair_summary.json')
            root, choice = declaration['root'], declaration['choice']
            if not (pair['status'] == 'complete' and pair['all_pass']
                    and all(pair['checks'].values()) and pair['line_id'] == line
                    and pair['root'] == {key: root[key] for key in ROOT_KEYS}):
                raise ValueError(f'{arm}/{scenario}/{line}: source pair or root failed')
            reference = declaration['reference']
            if scenario in references and references[scenario] != reference:
                raise ValueError(f'{arm}/{scenario}: source reference changed between roots')
            references[scenario] = reference
            outcomes = {}
            for branch, role in ROLES.items():
                child = _read(directory/branch/'summary.json')
                remaining = child['remaining']
                if not (child['status'] == 'complete' and child['integrity']['all_pass']
                        and all(child['integrity']['checks'].values())
                        and child['root'] == root and child['reference'] == reference
                        and child['role'] == role and child['external_seed'] == external_seed
                        and child['executed_action'] == choice[role+'_action']
                        and child['executed_hold_seconds'] == choice[role+'_hold_seconds']
                        and child['final_clock'] == 24000
                        and child['suffix_updates'] == child['suffix_admitted_labels'] == 0
                        and remaining['observed_at'] == 24000 and remaining['available_at'] == 24030
                        and remaining['dt'] == 24000-root['started_at'] > 0
                        and remaining == pair['remaining'][role]
                        and remaining['cost'] == 2*remaining['wait']+remaining['ride']):
                    raise ValueError(f'{arm}/{scenario}/{line}/{role}: source outcome failed')
                outcomes[role] = remaining
            delta = {key: outcomes['candidate'][key]-outcomes['reference'][key] for key in COST_KEYS}
            if not (delta == pair['delta'] and delta['cost'] == 2*delta['wait']+delta['ride']
                    and pair['reference_hold_seconds'] == choice['reference_hold_seconds']
                    and pair['candidate_hold_seconds'] == choice['candidate_hold_seconds']):
                raise ValueError(f'{arm}/{scenario}/{line}: paired label or action mismatch')
            records.append(dict(source_scope=scope, external_seed=external_seed,
                scenario=scenario, line_id=line, state=root['s'],
                ref_hold=pair['reference_hold_seconds'], cand_hold=pair['candidate_hold_seconds'],
                target_delta_person_seconds=delta['cost'], label_available_at=24030))
    return records, references


def run():
    started = time.time()
    if any((HERE/name).exists() for name in ('model.json', 'training_records.json', 'train_summary.json')):
        raise FileExistsError('The fixed v86 fit already has an artifact; no repeat fit')
    cohorts, references = {}, {}
    for arm in SOURCES:
        cohorts[arm], references[arm] = _load_cohort(arm)
    same_references = all(references['stale'][scenario]
        == references['fresh'][scenario]['source_snapshot_metadata'] for scenario in SCENARIOS)
    if not same_references:
        raise ValueError('Old and fresh labels must continue the same prior-run reference actor')
    old = _read(V85/'model.json')
    if not old['gate']['all_pass']:
        raise ValueError('The frozen v85 stale coefficients lack their original training gate')
    artifact = train(cohorts)
    exact = {name: artifact['arms']['stale']['models'][name]['coef'] == old['models'][name]['coef']
             for name in ('line', 'state')}
    artifact['gate']['checks'].update(same_reference_snapshots=same_references,
        stale_line_coefficients_exact=exact['line'], stale_state_coefficients_exact=exact['state'])
    artifact['gate']['all_pass'] = all(artifact['gate']['checks'].values())
    artifact.update(fitted_at_wall=time.time(),
        source_reference_paths={scenario: str(V84/'runs'/scenario/'reference_actor.pt') for scenario in SCENARIOS},
        source_native_roots={arm: str(source/'runs') for arm, source in SOURCES.items()},
        training_phase='After both prior cohorts and their24030 label maturity; before the new heldout episodes',
        heldout_external_seed=316207, heldout_labels_used=0,
        initialization_updates=0, prefix_training_updates=0, suffix_training_updates=0,
        source_audit=dict(native_pairs_per_arm=6, outcome_labels_per_arm=12,
            source_pair_checks_passed=True, same_reference_snapshots=same_references,
            stale_coefficients_exact=exact, old_model_path=str(V85/'model.json')))
    receipt = dict(status='complete' if artifact['gate']['all_pass'] else 'hold',
        gate=artifact['gate'], fits=artifact['fits'],
        arms={arm: dict(training_scope=value['training_scope'],
            coefficients={name: model['coef'] for name, model in value['models'].items()},
            design_ranks={name: model['design_rank'] for name, model in value['models'].items()},
            train_fit={name: model['train_fit'] for name, model in value['models'].items()},
            training_covariates=[dict(case=row['scenario']+'/'+row['line_id'],
                z=state_feature(row['state']), holding_difference=row['cand_hold']-row['ref_hold'],
                target_delta_per_passenger=row['target_delta_person_seconds']/DENOMINATOR)
                for row in cohorts[arm]]) for arm, value in artifact['arms'].items()},
        source_audit=artifact['source_audit'], native_simulation_seconds=0,
        new_initialization_updates=0, original_network_updates=0, new_gradient_steps=0,
        heldout_labels_used=0, new_heldout_started=False, heldout_external_seed=316207,
        wall_seconds=time.time()-started,
        interpretation='Complete newer supervision package versus older labels; source prefix distributions differ',
        next_stage='New native heldout evaluation only if the fixed fit and stale reproduction gates pass')
    for name, value in (('model.json', artifact), ('training_records.json', cohorts), ('train_summary.json', receipt)):
        with (HERE/name).open('x') as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            handle.write('\n')
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='matched_D_fit_complete',
        current=artifact['fits'], total=4, unit='closed_form_fits', gate=artifact['gate']['all_pass'])), flush=True)


if __name__ == '__main__':
    run()
