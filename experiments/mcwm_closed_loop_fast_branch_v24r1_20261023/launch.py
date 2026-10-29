"""Long native route-stream regression, then unchanged whole-day equivalence."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent
NATIVE_RESULT = 'native_stream_probe.json'


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def native_job():
    cfg = Path(infra.environment()['THESIS_SUMO_ROOT'])/'online_control/control_sim_traci_period.sumocfg'
    row = infra.task(protocol(), 'native_stream_probe', ['--sumo-cfg', str(cfg),
        '--prefix-ticks', 1000, '--suffix-ticks', 3000, '--timeout-seconds', 300,
        '--out', infra.remote(HERE/NATIVE_RESULT)], runner='native_fork_probe.py', ram_mb=8000)
    row['cpu'] = 2
    row['allowed_nodes'] = [node for node in row['allowed_nodes'] if node != 'node004']
    return row


def submit_day():
    gate = infra.collect(NATIVE_RESULT, max_bytes=15000)
    if gate.get('all_pass') is not True:
        raise RuntimeError('Cross-buffer native state/route-stream regression must pass first')
    first = json.loads((HERE.parent/'mcwm_closed_loop_fast_branch_v24_20261023/equivalence_gate.json').read_text())
    if first.get('integrity', {}).get('all_pass') is not True:
        raise RuntimeError('The existing bus first-pair equivalence must already pass')
    row = infra.task(protocol(), 'whole_day_execution_benchmark', ['--mode', 'day',
        '--seed', 6201, '--scenario', 'stable', '--preflight',
        '--out', infra.remote(HERE/'benchmark_day')], runner='fast_day.py', ram_mb=24000)
    row['cpu'] = 2
    row['allowed_nodes'] = [node for node in row['allowed_nodes'] if node != 'node004']
    return infra.submit([row], 'day_benchmark')


def submit_day_compare():
    original = HERE.parent/'mcwm_closed_loop_stratified_pairs_v23_20261023/stage_a/runs/stable/seed6201'
    row = infra.task(protocol(), 'whole_day_execution_equivalence', ['--fast',
        infra.remote(HERE/'benchmark_day'), '--source', infra.remote(original),
        '--out', infra.remote(HERE/'day_equivalence_gate.json'), '--wait'],
        runner='day_compare.py', ram_mb=2048)
    return infra.submit([row], 'day_equivalence_finalize')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'native', 'day', 'day-finalize',
        'collect-native', 'collect-day'))
    args = parser.parse_args()
    if args.operation == 'stage':
        names = ('bootstrap.py', 'infra.py', 'history.py', 'candidate.py', 'shift_model.py',
            'sampling.py', 'fork_state.py', 'route_streams.py', 'native_fork_probe.py',
            'compare.py', 'fast_day.py', 'day_compare.py', 'launch.py', 'protocol.json')
        result = infra.stage([HERE/name for name in names])
    elif args.operation == 'native':
        result = infra.submit([native_job()], 'native_stream_probe')
    elif args.operation == 'day': result = submit_day()
    elif args.operation == 'day-finalize': result = submit_day_compare()
    elif args.operation == 'collect-native': result = infra.collect(NATIVE_RESULT, max_bytes=15000)
    else: result = infra.collect('day_equivalence_gate.json', max_bytes=20000)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
