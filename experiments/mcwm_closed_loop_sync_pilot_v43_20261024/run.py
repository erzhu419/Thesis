"""One prospective native day; no trace loss or old diagnostic outcomes."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('stable', 'local_speed_drop'), required=True)
    parser.add_argument('--seed', type=int, default=6202)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE/'protocol.json').read_text())
    started = time.time()
    try:
        from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.collect import collect
        from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import PilotUpdates, write
        updates = PilotUpdates(args.out/'learning', protocol)
        native = collect(args, protocol, on_warmup_packet=updates.warmup, on_response=updates.response)
        learning = json.loads((args.out/'learning/update_summary.json').read_text())
        checks = dict(native=native['integrity']['all_pass'], updates=learning['integrity']['all_pass'])
        result = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), scenario=args.scenario,
            seed=args.seed, native_external_seed=protocol['native_external_seed'],
            native=native, learning=learning, wall=dict(started=started, finished=time.time()),
            trace_training_started=False)
        write(args.out/'pilot_summary.json', result)
        if not result['integrity']['all_pass']:
            raise RuntimeError('Native pilot did not pass its factual budget checks')
        print(json.dumps(dict(status=result['status'], integrity=result['integrity'])), flush=True)
    except BaseException as error:
        args.out.mkdir(parents=True, exist_ok=True)
        failure = args.out/'pilot_failure.json'
        if not failure.exists():
            resident = args.out/'failure.json'
            failure.write_text(json.dumps(dict(type=type(error).__name__, message=str(error),
                native_failure=json.loads(resident.read_text()) if resident.exists() else None,
                started=started, finished=time.time()), separators=(',', ':'))+'\n')
        raise


if __name__ == '__main__':
    main()
