"""Bounded resident result; no trace or checkpoint export."""
import argparse
import json
from pathlib import Path


def analyze(root):
    p = json.loads((root/'protocol.json').read_text())
    cases, pending, failures = [], [], []
    for seed in p['training_seeds']:
        for scenario in p['scenarios']:
            cell = root/'runs'/f'seed{seed}'/scenario
            if (cell/'failure.json').exists():
                failures.append(dict(seed=seed, scenario=scenario,
                    failure=json.loads((cell/'failure.json').read_text())))
            elif not (cell/'summary.json').exists():
                pending.append(dict(seed=seed, scenario=scenario))
            else:
                s = json.loads((cell/'summary.json').read_text())
                context = {k:v for k,v in s['context'].items() if k != 'feature_names'}
                coverage = dict(s['coverage'])
                coverage['own_line_s_feature_ranges'] = {k:v for k,v in
                    coverage['own_line_s_feature_ranges'].items() if k in ('7X', '102X', '705X')}
                cases.append(dict(seed=seed, scenario=scenario, status=s['status'],
                    integrity=s['integrity'], replay_exact=s['replay_exact'],
                    budget=s['budget'], coverage=coverage,
                    zero_init_probes=s['zero_init_probes'], context=context,
                    execution_wall=s['execution_wall']))
    passed = not pending and not failures and len(cases)==2 and all(c['integrity']['all_pass'] for c in cases)
    overlap = None
    if len(cases)==2:
        starts = [c['execution_wall']['native_started_wall'] for c in cases]
        ends = [c['execution_wall']['native_finished_wall'] for c in cases]
        overlap = max(0., min(ends)-max(starts))
    return dict(status='complete' if passed else 'pending' if pending else 'integrity_failed',
        integrity=dict(all_pass=passed), cases=cases, pending=pending, failures=failures,
        native_days_overlap_wall_seconds=overlap, diagnostic_budget=p['compute_budget'],
        next_decision='Inspect admitted per-line context/physical coverage and variation before budget-matched cumulative memory/zero-memory shadow learning. No automatic policy launch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        value = analyze(args.root)
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        limit = json.loads((args.root/'protocol.json').read_text())['summary_max_bytes']
        if len(text.encode()) > limit:
            raise RuntimeError('Coverage summary exceeds its declared small-transfer limit')
        with args.out.open('x') as handle:
            handle.write(text+'\n')
        if not value['integrity']['all_pass']:
            raise RuntimeError('Ordinary replay or context precheck failed')
    except Exception as error:
        with (args.root/'coverage_summary_failure.json').open('x') as handle:
            json.dump(dict(type=type(error).__name__, message=str(error)), handle)
        raise
