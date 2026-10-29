"""Service siblings must both launch, overlap natively, and expose pair failures."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def arguments(tmp_path):
    return SimpleNamespace(seed=6201, scenario='stable', source=tmp_path/'source',
        target_id='heldout/7X/heldout1', out=tmp_path/'pair', preflight=True)


def install_children(monkeypatch, runner, args, *, overlap=True, failed=False):
    launches, children = [], []
    class Child:
        def __init__(self, command, **kwargs):
            self.args, self.pid = command, 100+len(launches)
            self.arm = command[command.index('--branch')+1]
            self.running = True
            launches.append(self.arm)
            children.append(self)
            out = Path(command[command.index('--out')+1])
            out.mkdir(parents=True)
            start = 10. if self.arm == 'low' else (12. if overlap else 21.)
            value = dict(status='complete', integrity=dict(all_pass=True),
                selected=dict(sequence=1), prefix=dict(external_seed=96201),
                prediction=dict(shared=[1.]), native=dict(other_batch_actions=[]),
                execution_wall=dict(native_started_wall=start, native_finished_wall=start+10))
            (out/'summary.json').write_text(json.dumps(value))
        def wait(self, timeout=None):
            assert launches == ['low','high'], 'Both children must launch before first wait'
            if failed and self.arm == 'low':
                self.running = False
                return 1
            self.running = False
            return 0
        def poll(self):
            return None if self.running else 0
        def terminate(self):
            self.terminated = True
            self.running = False
        def kill(self):
            self.running = False
    monkeypatch.setattr(runner.subprocess, 'Popen', Child)
    return launches, children


def test_service_pair_launches_both_before_wait_and_measures_native_overlap(monkeypatch, tmp_path):
    runner = module('v23_service_parallel_ok', HERE/'run_pair.py')
    args = arguments(tmp_path)
    launches, _ = install_children(monkeypatch, runner, args)
    result = runner.run(args)
    assert launches == ['low','high'] and result['integrity']['all_pass'] is True
    assert result['overlap_seconds'] == 8.
    assert result['training_labels'] == 0 and result['suffix_native_seconds'] == 600
    assert '--preflight' in runner.branch_command(args, 'low')


def test_process_overlap_alone_cannot_pass_serial_native_phases(monkeypatch, tmp_path):
    runner = module('v23_service_parallel_bad_overlap', HERE/'run_pair.py')
    args = arguments(tmp_path)
    install_children(monkeypatch, runner, args, overlap=False)
    with pytest.raises(RuntimeError, match='did not overlap'):
        runner.run(args)
    receipt = json.loads((args.out/'parallel_execution.json').read_text())
    assert receipt['integrity']['checks']['native_phase_overlap'] is False
    assert (args.out/'failure.json').is_file()


def test_failed_child_reaps_sibling_and_keeps_pair_failure(monkeypatch, tmp_path):
    runner = module('v23_service_parallel_bad_child', HERE/'run_pair.py')
    args = arguments(tmp_path)
    _, children = install_children(monkeypatch, runner, args, failed=True)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run(args)
    assert children[1].terminated is True
    assert (args.out/'failure.json').is_file()
    assert (args.out/'low.log').is_file() and (args.out/'high.log').is_file()


def test_pair_root_failure_is_terminal_in_coordinator(monkeypatch, tmp_path):
    infra = module('v23_pair_failure_infra', HERE/'infra.py')
    monkeypatch.setitem(sys.modules, 'infra', infra)
    launch = module('v23_pair_failure_launch', HERE/'launch.py')
    monkeypatch.setitem(sys.modules, 'launch', launch)
    monkeypatch.setattr(launch, 'protocol', lambda:dict(preflight=dict(seed=6201,scenario='stable'),
        target_lines=['7X'],heldout_windows=[dict(name='heldout1')]))
    c = module('v23_pair_failure_continuation', HERE/'continue_batch.py')
    monkeypatch.setattr(c, 'HERE', tmp_path)
    monkeypatch.setattr(c.infra, 'remote', str)
    monkeypatch.setattr(c, 'readonly_remote', lambda code:subprocess.check_output(
        [sys.executable,'-c',code],text=True))
    pair = tmp_path/launch.evaluation_path(launch.evaluation_cells()[0], 'low').parent
    pair.mkdir(parents=True)
    (pair/'failure.json').write_text(json.dumps(dict(type='RuntimeError',message='native overlap failed')))
    value = c.metadata()
    assert value['service_pairs']['terminal'] == 0
    assert value['service_pairs']['bad'][0]['path'] == str(pair.relative_to(tmp_path))
    monkeypatch.setattr(c, 'metadata', lambda:value)
    with pytest.raises(RuntimeError, match='no automatic simulation retry'):
        c.wait_until('test', lambda _:True)
