"""Verify the v15 branch budget, native gates and small-result continuation."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    infra = load_module('v15_driver_test_infra', HERE/'infra.py')
    monkeypatch.setitem(sys.modules, 'infra', infra)
    launch = load_module('v15_driver_test_launch', HERE/'launch.py')
    monkeypatch.setitem(sys.modules, 'launch', launch)
    driver = load_module('v15_driver_test_continue', HERE/'continue_batch.py')
    return launch, driver, infra


@pytest.fixture
def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def test_twenty_four_high_branches_have_distinct_paths_and_matched_seed(modules, protocol):
    launch, _driver, _infra = modules
    with patch.object(launch, 'protocol', return_value=protocol):
        cells = launch.cells()
        pairs = [(cell, arm, launch.make_branch(cell, arm)) for cell in cells for arm in ('low', 'high')]
    assert len(cells) == 12 and len(pairs) == 24
    assert len({row['signature'] for _cell, _arm, row in pairs}) == 24
    paths = []
    for cell, arm, task in pairs:
        argv = shlex.split(task['cmd'])
        assert task['priority'] == 'high' and task['cpu'] == 1 and task['vram'] == 0
        assert task['allowed_nodes'] == [f'node{i:03d}' for i in range(1, 7)]
        assert argv[argv.index('--seed')+1] == '6201'
        assert argv[argv.index('--window')+1] == cell['window']
        assert argv[argv.index('--target-line')+1] == cell['target_line']
        assert argv[argv.index('--branch')+1] == arm
        assert argv[argv.index('--manifest')+1] == launch.infra.remote(HERE/'selection_manifest.json')
        out = argv[argv.index('--out')+1]
        assert out == launch.infra.remote(HERE/'pairs'/cell['scenario']/cell['target_line']/'seed6201'/cell['window']/arm)
        paths.append(out)
    assert len(set(paths)) == 24


def test_preflight_allocates_only_two_fixed_training_state_branches(modules, protocol):
    launch, _driver, infra = modules
    with patch.object(launch, 'protocol', return_value=protocol), \
            patch.object(launch, 'require_selection') as gate, \
            patch.object(infra, 'submit', return_value={'submitted': []}) as submit:
        launch.submit_preflight()
    gate.assert_called_once_with(preflight=True)
    rows, batch = submit.call_args.args
    assert batch == 'preflight_pairs' and len(rows) == 2
    for arm, task in zip(('low', 'high'), rows):
        argv = shlex.split(task['cmd'])
        assert '--preflight' in argv and task['priority'] == 'high'
        assert argv[argv.index('--window')+1] == 'preflight'
        assert argv[argv.index('--out')+1] == infra.remote(HERE/'preflight/pair'/arm)
        assert argv[argv.index('--manifest')+1] == infra.remote(HERE/'preflight/selection_manifest.json')


@pytest.mark.parametrize('gate', [dict(status='failed', integrity={'all_pass': True}),
                                  dict(status='complete', integrity={'all_pass': False})])
def test_failed_native_gate_cannot_submit_main(modules, tmp_path, monkeypatch, gate):
    launch, _driver, infra = modules
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    with patch.object(infra, 'collect_json', return_value=gate) as collect, \
            patch.object(infra, 'submit') as submit:
        with pytest.raises(RuntimeError, match='preflight'):
            launch.submit_pairs()
    collect.assert_called_once_with('preflight/gate.json', max_bytes=20000)
    submit.assert_not_called()


def test_missing_main_state_is_preserved_but_not_accepted_for_preflight(modules, protocol, tmp_path, monkeypatch):
    launch, _driver, infra = modules
    missing = dict(status='complete', states=[dict(status='selected'), dict(status='selection_missing')])
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    with patch.object(infra, 'collect_json', return_value=missing) as collect:
        assert launch.require_selection() == missing
        with pytest.raises(RuntimeError, match='executable'):
            launch.require_selection(preflight=True)
    assert [call.args[0] for call in collect.call_args_list] == [
        'selection_summary.json', 'preflight/selection_summary.json']
    assert all(call.kwargs['max_bytes'] == 20000 for call in collect.call_args_list)
    with patch.object(launch, 'protocol', return_value=protocol), \
            patch.object(launch, 'require_gate'), \
            patch.object(launch, 'require_selection', return_value=missing), \
            patch.object(infra, 'remote', side_effect=str), \
            patch.object(infra, 'submit', return_value={'submitted': []}) as submit:
        launch.submit_pairs()
    assert len(submit.call_args.args[0]) == 24


@pytest.mark.parametrize('preflight', [False, True])
def test_finalizer_paths_and_manifest_stay_resident(modules, protocol, tmp_path, monkeypatch, preflight):
    launch, _driver, infra = modules
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    batch = 'preflight_pairs' if preflight else 'paired'
    (tmp_path/(batch+'_submission.json')).write_text('{"submitted":[]}')
    with patch.object(launch, 'protocol', return_value=protocol), \
            patch.object(infra, 'remote', side_effect=str), \
            patch.object(infra, 'collect_json') as collect, \
            patch.object(infra, 'submit', return_value={'submitted': []}) as submit:
        launch.submit_finalizer(preflight=preflight)
    rows, submitted_batch = submit.call_args.args
    assert len(rows) == 1 and rows[0]['priority'] == 'high'
    argv = shlex.split(rows[0]['cmd'])
    root = tmp_path/'preflight' if preflight else tmp_path
    assert argv[argv.index('--root')+1] == str(root)
    assert argv[argv.index('--manifest')+1] == str(root/'selection_manifest.json')
    assert argv[argv.index('--out')+1] == str(root/('gate.json' if preflight else 'training_pair_summary.json'))
    assert '--wait' in argv and ('--preflight' in argv) == preflight
    assert submitted_batch == ('preflight_finalize' if preflight else 'diagnostic_finalize')
    collect.assert_not_called()
    assert launch.SUMMARY_MAX_BYTES == 100000 and launch.SMALL_MAX_BYTES == 20000


@pytest.mark.parametrize('relative', ['selection_failure.json', 'preflight/selection_failure.json',
                                    'summary_failure.json', 'preflight/summary_failure.json'])
def test_existing_resident_failure_stops_driver_before_submission(modules, protocol, tmp_path, monkeypatch, relative):
    launch, driver, infra = modules
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    monkeypatch.setattr(driver, 'HERE', tmp_path)
    (tmp_path/'selection_submission.json').write_text('{"submitted":[]}')
    fail = tmp_path/relative
    fail.parent.mkdir(parents=True, exist_ok=True)
    fail.write_text('{"error":"retained"}')

    def local_read(_node, command, **kwargs):
        # Execute exactly the resident metadata program against the temporary
        # fixture, without network access or source/result mutation.
        output = subprocess.check_output(shlex.split(command), text=True)
        return 0, output, ''

    with patch.object(launch, 'protocol', return_value=protocol), \
            patch.object(infra, 'remote', side_effect=str), \
            patch.object(infra, 'PYTHON', sys.executable), \
            patch.object(infra.scheduler, 'run_on', side_effect=local_read), \
            patch.object(launch, 'submit_preflight') as preflight, \
            patch.object(launch, 'submit_pairs') as pairs, \
            patch.object(driver.time, 'sleep') as sleep:
        metadata = driver.metadata()
        assert metadata['failures'] == [relative]
        with pytest.raises(RuntimeError, match='Preserved native'):
            driver.main()
    preflight.assert_not_called()
    pairs.assert_not_called()
    sleep.assert_not_called()


def test_protocol_diagnostic_and_future_pair_budget_match_source(protocol):
    source = json.loads((HERE/protocol['reference_v14_root']/'protocol.json').resolve().read_text())
    for key in ('initializer_updates', 'history_rows', 'source_rows', 'horizon_seconds',
                'target_record_budget', 'policy_update_budget', 'model_init_updates',
                'model_refit_updates', 'target_seed_offset', 'paired_suffix_seconds'):
        assert protocol[key] == source[key]
    budget = protocol['diagnostic_budget']
    assert budget['design_states'] == 12 and budget['branches'] == 24
    assert budget['native_suffix_seconds'] == 24*300
    assert budget['training_suffix'] == budget['diagnostic_labels_enter_fit'] == 0
    assert protocol['future_learning']['paired_action_outcomes_per_training_pair'] == 2
    assert protocol['future_learning']['total_slots_if_started'] == 285
    assert protocol['future_learning']['status'] == 'not_submitted'
