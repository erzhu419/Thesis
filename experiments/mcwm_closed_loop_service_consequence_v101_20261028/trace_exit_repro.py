"""Reproduce the old logger's unflushed child exit using only temporary files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def child(path, close):
    from types import SimpleNamespace
    sys.path.insert(0, str(HERE.parents[1]))
    from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
    trace = NativePrefixTrace(path)
    event = dict(agent='a', line_id='7X', vehicle_id='v', trip_id='v', station_id='s',
        domain='target', sequence=0, episode_id=0, started_at=12000.,
        s=[0.], raw_s=[0.], event=[0.], a=[0.], proposed_a=[0.])
    point = SimpleNamespace(time_seconds=12000., waiting_passengers=2,
        in_vehicle_passengers=3, accrued_wait_seconds=4., accrued_in_vehicle_seconds=5.)
    trace.observe(event, point)
    if close:
        trace.close()
    os._exit(0)


def main():
    observed = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, close in [('old_exit', False), ('close_before_exit', True)]:
            path = Path(directory)/(name+'.jsonl')
            subprocess.run([sys.executable, '-B', str(HERE/'trace_exit_repro.py'),
                '--child', str(path), str(int(close))], check=True)
            observed[name] = len(path.read_text().splitlines())
    result = dict(expected_rows=1, persisted_rows=observed,
        reproduced=observed == {'old_exit': 0, 'close_before_exit': 1},
        new_native_seconds=0, new_fits=0, old_files_changed=False)
    with (HERE/'trace_exit_repro.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    if sys.argv[1:2] == ['--child']:
        child(sys.argv[2], bool(int(sys.argv[3])))
    else:
        main()
