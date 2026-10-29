"""Fit only the six completed v84 contrasts; no heldout files or old learners."""
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.model import train, state_feature

SOURCE = HERE.parent/'mcwm_closed_loop_reference_collection_v84_20261027'


def run():
    started = time.time()
    if (HERE/'model.json').exists():
        raise FileExistsError('Frozen model already exists; no repeat fit')
    source_summary = json.loads((SOURCE/'summary.json').read_text())
    reference_audit = json.loads((SOURCE/'reference_audit.json').read_text())
    if not source_summary['integrity']['all_pass'] or not reference_audit['all_pass']:
        raise RuntimeError('The fixed training labels lack their completed native audits')
    records = []
    for pair in source_summary['pairs']:
        scene, line = pair['scenario'], pair['line_id']
        declaration = json.loads((SOURCE/'runs'/scene/'pairs'/line/'root.json').read_text())
        if (declaration['root']['sequence'] != pair['root']['sequence']
                or declaration['root']['started_at'] != pair['root']['started_at']
                or not pair['valid_for_signal']):
            raise RuntimeError('Training state and paired native outcome do not match')
        if any(value['available_at'] != 24030 for value in pair['remaining'].values()):
            raise RuntimeError('Unreleased source label')
        records.append(dict(source_scope='v84_train', scenario=scene, line_id=line,
            state=declaration['root']['s'], ref_hold=pair['actions']['reference_hold_seconds'],
            cand_hold=pair['actions']['candidate_hold_seconds'],
            target_delta_person_seconds=pair['candidate_minus_reference']['cost'], label_available_at=24030))
    artifact = train(records)
    artifact.update(fitted_at_wall=time.time(), source_summary_path=str(SOURCE/'summary.json'),
        training_phase='After prior v84 episodes and24030 label release; outside their live loops',
        source_reference_paths={scene:str(SOURCE/'runs'/scene/'reference_actor.pt')
                                for scene in ('stable','local_speed_drop')},
        heldout_external_seed=216207, heldout_labels_used=0,
        initialization_updates=0, prefix_training_updates=0, suffix_training_updates=0)
    with (HERE/'model.json').open('x') as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    with (HERE/'training_records.json').open('x') as handle:
        json.dump(records, handle, separators=(',', ':'), allow_nan=False)
    receipt = dict(status='complete' if artifact['gate']['all_pass'] else 'hold',
        gate=artifact['gate'], fits=artifact['fits'], parameters={'line':3,'state':4},
        design_ranks={key:value['design_rank'] for key,value in artifact['models'].items()},
        train_fit={key:value['train_fit'] for key,value in artifact['models'].items()},
        coefficients={key:value['coef'] for key,value in artifact['models'].items()},
        training_covariates=[dict(case=row['scenario']+'/'+row['line_id'],z=state_feature(row['state']),
            holding_difference=row['cand_hold']-row['ref_hold'],
            target_delta_per_passenger=row['target_delta_person_seconds']/15079) for row in records],
        training_pairs=6, label_scope='v84_only', consumer_phase='after_prior_episode24030_release',
        heldout_labels_used=0, native_simulation_seconds=0, old_network_updates=0,
        new_gradient_steps=0, wall_seconds=time.time()-started,
        next_stage='New heldout pairs only if the fixed design gate passes; no control takeover')
    with (HERE/'train_summary.json').open('x') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='D_fit_complete', current=artifact['fits'],
        total=2, unit='closed_form_fits', gate=artifact['gate']['all_pass'])), flush=True)


if __name__ == '__main__':
    run()
