"""Finish the bounded summary from completed v42 workers without replaying them."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_sync_v42_20261024.analyze import analyze


def read(path):
    return json.loads(path.read_text())


if __name__ == '__main__':
    failure = read(HERE / 'failure.json')
    if failure['message'] != 'Small-result summary exceeds24000bytes':
        raise RuntimeError('This finalizer only consumes the completed size-limited run')
    runs = HERE / 'runs'
    for case in ('stable', 'local_speed_drop'):
        result = read(runs / case / 'summary.json')
        if result['status'] != 'complete' or not result['integrity']['all_pass']:
            raise RuntimeError('Both existing workers must have completed successfully')
    value = analyze(runs, HERE / 'analysis', read(runs / 'snapshot_inventory.json'),
                    {'stable': 0, 'local_speed_drop': 0}, read(HERE / 'sync_plan.json'))
    with (HERE / 'analysis/finalization.json').open('x') as handle:
        json.dump(dict(previous_failure=failure, worker_replays=0,
            new_optimizer_updates=0, new_native_seconds=0,
            summary_bytes=(HERE / 'analysis/summary.json').stat().st_size), handle)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'],
        collection_recommended=value['decision']['collection_recommended'])))
