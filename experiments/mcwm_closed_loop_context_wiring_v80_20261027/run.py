"""Frozen resident context integration; no simulator or optimizer steps."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def write(path, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
    if len(raw.encode()) > 48000:
        raise ValueError('Planned small summary exceeds48KB')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        output.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    args = parser.parse_args()
    from experiments.mcwm_closed_loop_context_wiring_v80_20261027.probe import run_parity
    protocol = json.loads((HERE/'protocol.json').read_text())
    started, rows = time.monotonic(), []
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            row = run_parity(seed, arm)
            rows.append(row)
            print(json.dumps(dict(seed=seed, arm=arm, all_pass=row['all_pass'],
                new_optimizer_steps=0, new_simulated_seconds=0)), flush=True)
    passed = all(row['all_pass'] for row in rows)
    write(args.out/'integration.json', dict(status='complete', all_pass=passed, entries=rows,
        elapsed_seconds=time.monotonic()-started, actual_budget=protocol['budget_this_stage'],
        resident_archives_used_for_temporary_test_rows=True,
        diagnostic_v79_context_used=False, arrays_and_initializers_remain_resident=True))
    write(args.out/'decision.json', dict(status='complete', frozen_integration_pass=passed,
        collection_started=False, response_training_started=False,
        next=('Prepare one separately declared budget-matched actual-control batch; '
              'reuse its first8 steps for optimizer parity and run paired arms concurrently.'
              if passed else 'Correct the failed wiring before any collection or response updates.'),
        old_code_results_and_gates_modified=False,
        performance_improvement_tested=False,
        actual_budget=protocol['budget_this_stage']))
    if not passed:
        raise RuntimeError('Frozen integration failed; see small summary')


if __name__ == '__main__':
    main()
