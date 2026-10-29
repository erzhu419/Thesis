"""Run archived-coverage audit and frozen parity; never start a native episode."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def write(path, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
    if len(raw.encode()) > 48000:
        raise ValueError('Small JSON exceeds48KB: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        output.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    args = parser.parse_args()
    from audit import audit_sources
    visibility = audit_sources(HERE.parent)
    write(args.out/'visibility.json', visibility)
    print(json.dumps(dict(phase='archived_visibility_complete', source='existing files only')), flush=True)
    from probe import run_parity
    rows = []
    for seed in (6208, 6207):
        row = run_parity(seed)
        rows.append(row)
        print(json.dumps(dict(phase='frozen_parity_complete', seed=seed,
            all_pass=row['all_pass'])), flush=True)
    parity_pass = all(row['all_pass'] for row in rows)
    write(args.out/'model_parity.json', dict(status='complete', all_pass=parity_pass,
        entries=rows, new_simulated_seconds=0, new_training_updates=0,
        new_feedback_slots=0, arrays_and_initializers_remain_resident=True))
    decision = dict(status='complete', zero_initialization_pass=parity_pass,
        source_api_visibility_possible=visibility['source_api_visibility_possible'],
        native_joint_coverage_verified=visibility['native_joint_coverage_verified'],
        joint_context_observations=visibility['joint_context_observations'],
        collection_started=False, response_training_started=False,
        ready_for_response_training=False,
        native_observation_preflight_may_be_planned=bool(parity_pass and visibility['source_api_visibility_possible']),
        next=('Plan a separately budgeted pre-action native observation-only preflight; do not start it in v78.'
            if parity_pass and visibility['source_api_visibility_possible'] else
            'Resolve failed source/API or zero-initialization checks; do not collect or learn.'),
        unchanged=['actor/V observations', 'F', 'trust', 'sampling', '285 covariate/outcome contract',
            '456 update schedule', 'original initialization'],
        context_information_dimension_expands_if_later_collected=True,
        new_simulated_seconds=0, new_training_updates=0, new_feedback_slots=0,
        old_code_and_results_modified=False,
        limits=['API/source visibility and fake-reader tests are not empirical native joint coverage.',
            'Synthetic nonzero context checks prove wiring only.',
            'No original50000 or online408/456 training updates were rerun.',
            'The critic wrapper is not installed in the original trainer or deployed policy.'])
    write(args.out/'decision.json', decision)
    if not parity_pass:
        raise RuntimeError('Frozen parity failed; no collection or training is allowed')


if __name__ == '__main__':
    main()
