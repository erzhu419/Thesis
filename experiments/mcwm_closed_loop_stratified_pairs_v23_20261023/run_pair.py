"""One reserved two-CPU task executes both frozen service suffixes concurrently."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def branch_command(args, arm):
    command = [sys.executable, '-u', str(HERE/'run.py'), '--mode', 'branch',
        '--seed', str(args.seed), '--scenario', args.scenario,
        '--source', str(args.source), '--target-id', args.target_id,
        '--branch', arm, '--out', str(args.out/arm)]
    if args.preflight:
        command.append('--preflight')
    return command


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    children, launches, values = {}, {}, {}
    try:
        with ExitStack() as logs:
            try:
                for arm in ('low', 'high'):
                    log = logs.enter_context((args.out/(arm+'.log')).open('x'))
                    launched = time.time()
                    children[arm] = subprocess.Popen(branch_command(args, arm),
                        stdout=log, stderr=subprocess.STDOUT)
                    launches[arm] = dict(pid=children[arm].pid, started_wall=launched)
                pending, last_progress = set(children), time.monotonic()
                while pending:
                    for arm in ('low', 'high'):
                        if arm not in pending:
                            continue
                        child = children[arm]
                        try:
                            result = child.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            continue
                        if result:
                            raise subprocess.CalledProcessError(result, child.args)
                        launches[arm]['ended_wall'] = time.time()
                        pending.remove(arm)
                    if time.monotonic()-last_progress >= 45:
                        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='parallel_service_pair',
                            pair_id=args.target_id, running=sorted(pending),
                            child_pids={arm:children[arm].pid for arm in sorted(pending)})), flush=True)
                        last_progress = time.monotonic()
                for arm in ('low', 'high'):
                    values[arm] = json.loads((args.out/arm/'summary.json').read_text())
                    if values[arm].get('status') != 'complete' or values[arm].get('integrity', {}).get('all_pass') is not True:
                        raise RuntimeError('Native service branch failed integrity: '+arm)
            finally:
                running = [child for child in children.values() if child.poll() is None]
                for child in running:
                    child.terminate()
                for child in running:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        low, high = values['low'], values['high']
        native_wall = {arm:values[arm]['execution_wall'] for arm in ('low', 'high')}
        overlap = min(v['native_finished_wall'] for v in native_wall.values())-max(
            v['native_started_wall'] for v in native_wall.values())
        checks = dict(native_phase_overlap=overlap > 0,
            selected_exact=low['selected'] == high['selected'],
            prefix_exact=low['prefix'] == high['prefix'],
            prediction_exact=low['prediction'] == high['prediction'],
            sibling_actions_exact=low['native']['other_batch_actions'] == high['native']['other_batch_actions'],
            external_seed_present=low['prefix'].get('external_seed') is not None)
        receipt = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            target_id=args.target_id, launches=launches, native_wall=native_wall,
            overlap_seconds=max(0., overlap), suffixes=2, suffix_native_seconds=600,
            training_labels=0, integrity=dict(all_pass=all(checks.values()), checks=checks))
        write(args.out/'parallel_execution.json', receipt)
        if not receipt['integrity']['all_pass']:
            raise RuntimeError('Frozen native siblings lost pairing or did not overlap')
        return receipt
    except BaseException as error:
        write(args.out/'failure.json', dict(status='failed', target_id=args.target_id,
            type=type(error).__name__, message=str(error), launches=launches))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
