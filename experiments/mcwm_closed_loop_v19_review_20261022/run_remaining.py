"""Run only the five unqueried v19 arms with the corrected clock observation."""
import argparse
import importlib.util
from pathlib import Path
import sys

from clock_policy import NativeClockPolicy

SOURCE = Path(__file__).resolve().parent.parent / 'mcwm_closed_loop_execution_phase_v19_20261022'


def run(args):
    if (args.target_id, args.branch) == ('train/7X/late', 'low'):
        raise ValueError('The completed 7X low suffix must be reused, not queried again')
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('v19_native_runner_clock_revision', SOURCE / 'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.PhasePolicy = NativeClockPolicy
    return runner.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('branch',), required=True)
    parser.add_argument('--seed', type=int, choices=(6201,), required=True)
    parser.add_argument('--scenario', choices=('stable',), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target-id', required=True)
    parser.add_argument('--branch', choices=('low', 'high'), required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
