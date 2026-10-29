"""Verify frozen models, collect native chains, then score all available suffixes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def read_if_present(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, separators=(',', ':'), allow_nan=False)
        output.write('\n')


def phase(name, script, arguments, out):
    command = [sys.executable, '-u', str(HERE/script), *map(str, arguments)]
    started = time.time()
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=name, status='starting')), flush=True)
    with (out/(name+'.log')).open('x') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    receipt = dict(returncode=result.returncode, started=started, finished=time.time())
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=name, **receipt)), flush=True)
    return receipt


def pipeline(args, protocol):
    """A failed native gate retains partial scoring and still fails this job."""
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    phases, current = {}, 'model_verify'
    model_gate = native = scored = None
    started = time.time()
    source = HERE.parent/protocol['source_v37_directory']
    model_root = HERE.parent/protocol['source_v37r1_directory']/'runs'
    common = ['--scenario', args.scenario, '--model-root', model_root, '--source-root', source]

    def budget():
        failure = read_if_present(out/'native'/'failure.json')
        value = dict(native['budget'] if native is not None and 'budget' in native
                     else (failure or {}).get('budget', {}))
        if model_gate is not None:
            value['frozen_training_slots'] = model_gate.get('source_feedback_slots')
            value['frozen_updates_per_model'] = model_gate.get('source_updates')
        if failure is not None:
            value['interrupted_native_clock'] = failure.get('native_time')
        return value

    def failure_record(message, phase_name, returncode=None):
        return dict(status='failed', phase=phase_name, returncode=returncode,
            message=message, phases=phases, native_budget=budget(),
            native_failure=read_if_present(out/'native'/'failure.json'))

    try:
        phases[current] = phase(current, 'frozen.py', [*common, '--verify-only',
            '--out', out/'model_gate.json'], out)
        model_gate = read_if_present(out/'model_gate.json')
        if phases[current]['returncode'] or model_gate is None or not model_gate.get('all_pass'):
            raise RuntimeError('Frozen-model reproduction did not pass; native prefix was not started')

        current = 'native'
        phases[current] = phase(current, 'collect.py', ['--scenario', args.scenario,
            '--seed', args.seed, '--out', out/'native'], out)
        native = read_if_present(out/'native'/'summary.json')

        # A collector can have complete legal suffixes before a missing-state
        # gate or a later native failure. Keep and score these records once.
        available = (out/'native').exists() and any((out/'native').rglob('events.jsonl'))
        if available:
            current = 'scoring'
            phases[current] = phase(current, 'frozen.py', [*common,
                '--native-root', out/'native', '--out', out/'scored'], out)
            scored = read_if_present(out/'scored'/'summary.json')

        checks = dict(frozen_reproduction=model_gate['all_pass'],
            native_process=phases['native']['returncode'] == 0,
            native_integrity=native is not None and native.get('integrity', {}).get('all_pass') is True,
            suffix_scoring_started=available,
            scoring_process=available and phases['scoring']['returncode'] == 0,
            scoring_integrity=scored is not None and scored.get('integrity', {}).get('all_pass') is True,
            no_training=native is not None and native.get('budget', {}).get('new_optimizer_updates') == 0
                and native.get('budget', {}).get('new_training_labels') == 0
                and scored is not None and scored.get('new_updates') == 0
                and scored.get('new_feedback') == 0 and model_gate.get('new_updates') == 0)
        complete = all(checks.values())
        summary = dict(status='complete' if complete else 'integrity_failed',
            scenario=args.scenario, initialization_seed=args.seed,
            integrity=dict(all_pass=complete, checks=checks), phases=phases,
            model_gate=str(out/'model_gate.json'), native_summary=str(out/'native'/'summary.json'),
            scoring_summary=str(out/'scored'/'summary.json') if available else None,
            new_updates=0 if checks['no_training'] else None,
            new_feedback=0 if checks['no_training'] else None,
            budget=budget(), wall_seconds=time.time()-started)
        write(out/'summary.json', summary)
        if not complete:
            failed = next((name for name, value in phases.items() if value['returncode']),
                          'native' if not checks['native_integrity'] else 'scoring')
            write(out/'failure.json', failure_record('v38 pipeline did not pass every actual stage',
                failed, phases.get(failed, {}).get('returncode')))
            return 1
        return 0
    except Exception as error:
        write(out/'failure.json', failure_record(str(error), current,
            phases.get(current, {}).get('returncode')))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True, choices=('stable', 'local_speed_drop'))
    parser.add_argument('--seed', type=int, default=6202)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seed != 6202:
        parser.error('v38 retains initializer seed6202')
    return pipeline(args, json.loads((HERE/'protocol.json').read_text()))


if __name__ == '__main__':
    raise SystemExit(main())
