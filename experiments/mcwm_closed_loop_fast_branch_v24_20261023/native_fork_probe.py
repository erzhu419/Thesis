"""Isolated Linux libsumo fork experiment; not a production bus runner.

The parent advances once, forks two resident native states, and stays paused.
Children take one/two suffix lengths; parent later repeats both checkpoints.
Full vehicle/RNG values are compared locally, while only a small report remains.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET


def snapshot(sumo, path):
    # SUMO applies output-prefix even to manually saved states. A basename is
    # intentional: the absolute scratch prefix supplies its full output path.
    sumo.simulation.saveState(path.name)
    path = path.with_name('native-' + path.name)
    root = ET.parse(path).getroot()
    rng = [ET.tostring(node, encoding='unicode') for node in root.iter()
           if 'rng' in node.tag.lower()]
    if not rng:
        raise RuntimeError('saveState did not contain any native RNG states')
    vehicle_ids = sorted(sumo.vehicle.getIDList())
    return dict(time=float(sumo.simulation.getTime()), rng=rng,
                vehicles=[[vehicle, sumo.vehicle.getRoadID(vehicle),
                           sumo.vehicle.getLanePosition(vehicle),
                           sumo.vehicle.getSpeed(vehicle)]
                          for vehicle in vehicle_ids],
                persons=sorted(sumo.person.getIDList()))


def isolated_config(source, destination):
    """Move config outputs into probe scratch and preserve original inputs."""
    tree = ET.parse(source)
    root = tree.getroot()
    for section in list(root):
        if section.tag in ('output', 'report'):
            root.remove(section)
    inputs = root.find('input')
    if inputs is not None:
        for item in inputs:
            if item.tag.endswith('file') or item.tag.endswith('files'):
                item.set('value', ','.join(str((source.parent / value.strip()).resolve())
                                         for value in item.get('value', '').split(',')
                                         if value.strip()))
    for section in root:
        for item in list(section):
            if item.tag.startswith('save-state.'):
                section.remove(item)
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def wait_children(children, timeout):
    statuses = {}
    deadline = time.monotonic() + timeout
    remaining = set(children)
    while remaining and time.monotonic() < deadline:
        for pid in list(remaining):
            found, status = os.waitpid(pid, os.WNOHANG)
            if found:
                statuses[pid] = os.waitstatus_to_exitcode(status)
                remaining.remove(pid)
        if remaining:
            time.sleep(0.05)
    for pid in remaining:
        os.kill(pid, signal.SIGKILL)
    for pid in remaining:
        os.waitpid(pid, 0)
        statuses[pid] = 'timeout'
    return statuses


def run(args, scratch):
    import torch
    import libsumo as sumo

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    actor = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.Tanh(),
                                torch.nn.Linear(32, 2)).eval()
    actor_input = torch.tensor([[1., 2., 3., 4.]])

    def forward():
        with torch.inference_mode():
            return actor(actor_input).tolist()

    actor_before = forward()  # Initialize CPU inference before fork as in the runner.
    cfg = scratch / 'isolated.sumocfg'
    isolated_config(args.sumo_cfg, cfg)
    command = ['sumo', '-c', str(cfg), '--seed', str(args.seed), '--threads', '1',
               '--device.rerouting.threads', '1', '--save-state.rng', 'true',
               '--no-step-log', 'true', '--duration-log.disable', 'true',
               '--output-prefix', str(scratch / 'native-')]
    prefix_start = time.monotonic()
    started = False
    try:
        sumo.start(command)
        started = True
        for _ in range(args.prefix_ticks):
            sumo.simulationStep()
        prefix_wall = time.monotonic() - prefix_start
        before = snapshot(sumo, scratch / 'parent-before.xml')
        children = {}
        release = []
        for index in range(2):
            reader, writer = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(writer)
                result = dict(index=index, pid=os.getpid())
                result_path = scratch / f'child-{index}.json'
                try:
                    os.read(reader, 1)
                    os.close(reader)
                    result['initial'] = snapshot(sumo, scratch / f'child-{index}-before.xml')
                    result['actor_output'] = forward()
                    result['native_started'] = time.monotonic()
                    for _ in range(args.suffix_ticks):
                        sumo.simulationStep()
                    result['checkpoint'] = snapshot(sumo, scratch / f'child-{index}-checkpoint.xml')
                    if index == 1:
                        for _ in range(args.suffix_ticks):
                            sumo.simulationStep()
                    result['native_finished'] = time.monotonic()
                    result['final'] = snapshot(sumo, scratch / f'child-{index}-final.xml')
                    result['status'] = 'complete'
                except BaseException:
                    result.update(status='error', error=traceback.format_exc())
                result_path.write_text(json.dumps(result))
                os._exit(0 if result['status'] == 'complete' else 1)
            os.close(reader)
            children[pid] = index
            release.append(writer)
        for writer in release:
            os.write(writer, b'1')
            os.close(writer)
        wait_start = time.monotonic()
        statuses = wait_children(children, args.timeout_seconds)
        wait_wall = time.monotonic() - wait_start
        child_results = []
        for pid, index in children.items():
            path = scratch / f'child-{index}.json'
            child_results.append(json.loads(path.read_text()) if path.exists()
                                 else dict(index=index, pid=pid, status='no_result'))
        if any(value != 0 for value in statuses.values()):
            return dict(status='error', scope='native_fork_only',
                        child_exitcodes=statuses,
                        errors=[{key: value for key, value in child.items()
                                 if key in ('pid', 'index', 'status', 'error')}
                                for child in child_results])
        low, high = child_results
        after = snapshot(sumo, scratch / 'parent-after-children.xml')
        for _ in range(args.suffix_ticks):
            sumo.simulationStep()
        parent_checkpoint = snapshot(sumo, scratch / 'parent-checkpoint.xml')
        for _ in range(args.suffix_ticks):
            sumo.simulationStep()
        parent_final = snapshot(sumo, scratch / 'parent-final.xml')
        overlap = min(low['native_finished'], high['native_finished']) - max(
            low['native_started'], high['native_started'])
        checks = dict(
            both_child_initial_states_equal_parent=all(child['initial'] == before for child in child_results),
            parent_native_and_rng_unchanged_while_children_run=after == before,
            both_children_same_first_suffix=low['checkpoint'] == high['checkpoint'],
            low_matches_parent_continuation=low['final'] == parent_checkpoint,
            high_matches_parent_continuation=high['final'] == parent_final,
            child_clocks_advance_independently=low['final']['time'] < high['final']['time'],
            native_rng_advanced=parent_final['rng'] != before['rng'],
            cpu_actor_after_fork_exact=all(child['actor_output'] == actor_before for child in child_results),
            parent_cpu_actor_unchanged=forward() == actor_before,
            native_execution_overlaps=overlap > 0,
        )
        return dict(status='complete', all_pass=all(checks.values()), checks=checks,
                    scope='native_fork_only_not_full_bus_replay',
                    native_version=list(sumo.getVersion()), torch_version=torch.__version__,
                    cpu_threads=torch.get_num_threads(), seed=args.seed,
                    prefix_ticks=args.prefix_ticks, suffix_ticks=args.suffix_ticks,
                    parent_before_seconds=before['time'],
                    parent_after_children_seconds=after['time'],
                    child_final_seconds=[child['final']['time'] for child in child_results],
                    vehicle_counts=[len(before['vehicles']), len(parent_final['vehicles'])],
                    prefix_wall_seconds=prefix_wall, child_wait_wall_seconds=wait_wall,
                    child_overlap_seconds=max(0.0, overlap), child_exitcodes=statuses,
                    rng_blocks_compared=len(before['rng']),
                    limitations=['Current platform/build feasibility only; full Python bus state and model replay remain untested.',
                                 'Official support is not asserted for forking a running libsumo instance.'])
    finally:
        if started:
            sumo.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sumo-cfg', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prefix-ticks', type=int, default=100)
    parser.add_argument('--suffix-ticks', type=int, default=20)
    parser.add_argument('--seed', type=int, default=96201)
    parser.add_argument('--timeout-seconds', type=float, default=120)
    args = parser.parse_args()
    args.sumo_cfg = args.sumo_cfg.resolve()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error(f'Refusing to replace an existing probe report: {args.out}')
    try:
        with tempfile.TemporaryDirectory(prefix='native-fork-', dir=args.out.parent) as directory:
            result = run(args, Path(directory))
    except BaseException:
        result = dict(status='error', scope='native_fork_only', error=traceback.format_exc())
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result.get('all_pass') else 1


if __name__ == '__main__':
    sys.exit(main())
