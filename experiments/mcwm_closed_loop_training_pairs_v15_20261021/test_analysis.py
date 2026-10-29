"""Native training-pair audit invariants and compact output on actual cohort shape."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v15_pair_analysis', HERE/'analyze.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def endpoint(cost, *, dt=300.):
    return dict(status='complete', dt=dt, wait=cost/3., ride=cost/3., cost=cost)


def fixture_state(scene='stable', line='7X', window='pre_shock', index=1, actual_gain=-5., unsupported=()):
    selected = dict(sequence=index, started_at=6000.+index, line_id=line, vehicle_id='bus'+str(index),
        trip_id='trip'+str(index), station_id='stop', s=[.1]*26, raw_s=[.2]*26, event=[.3]*20,
        issue_ledger=dict(time_seconds=6000.+index, waiting_passengers=100., in_vehicle_passengers=150.,
            accrued_wait_seconds=1e5, accrued_in_vehicle_seconds=2e5))
    prediction = {}
    for n, model in enumerate(a.MODELS):
        gains = dict(global_next=0., same_vehicle_next=-20., fixed300=float(n+1))
        arms = {}
        for action in ('low', 'high'):
            arms[action] = dict(endpoints={name: dict(dt=300. if name=='fixed300' else 0. if name=='global_next' else 30.,
                cost=(100000. if name=='fixed300' else 0. if name=='global_next' else 10000.)-
                    (gains[name] if action=='high' else 0.)) for name in a.ENDPOINTS})
        if model in unsupported:
            gains['fixed300'] = None
            for arm in arms.values():
                arm['endpoints']['fixed300'] = None
        prediction[model] = dict(gains=gains, raw_global_next_gain=10., **arms)
    prediction['innovation_diag'] = dict(state_values={name: float(index)*(j+1)/100. for j, name in enumerate(a.STATE_FEATURES)})
    prediction['global_comparator'] = dict(arms={model: dict(cost_gain_low_minus_high=0., raw_cost_gain_low_minus_high=8.,
        low=dict(dt=0., cost=0.), high=dict(dt=0., cost=0.)) for model in ('base_analytic', 'global58')})
    source = dict(external_seed=83201, initializer_updates=50000, horizon_seconds=24000,
        prefix_target_labels=index*5, prefix_policy_updates=index*8, prefix_receipts=[], prefix_packets=index,
        day_relative='runs/'+scene+'/seed6201', full_day_budget=dict(target_labels=285, native_seconds=24000))
    context = dict(assignment_position=index%2, assignment_success_index=index, assignment_block_index=index//2,
        innovation_background=dict(sequence=index, issued_at=selected['started_at'], endpoints={name: dict(
            status='fitted', mean=[0.1, .2] if name=='fixed300' else [0., .1, .2], fit_rows=10,
            max_available_at=selected['started_at']-30, max_sequence=index-1) for name in a.ENDPOINTS}))
    prefix = dict(initial_updates=50000, external_seed=83201, counters=dict(policy=50000+index*8, world=index*20, trust=index*8),
        receipts=[], freeze_clock=selected['started_at'], native_steps=int(selected['started_at']), issued_count=index+1)
    state = dict(state_id=scene+'/'+line+'/'+window, status='selected',
        cell=dict(seed=6201, scenario=scene, target_line=line, window_name=window, window_index=0 if window=='pre_shock' else 1),
        selected=selected, nominal_a=[-1.], nominal_hold_seconds=0., prediction=prediction, source=source,
        source_truth={name: endpoint(0. if name=='global_next' else 70000., dt=0. if name=='global_next' else 300.) for name in a.ENDPOINTS})
    branches = []
    for action, hold in (('low', 0.), ('high', 7.5)):
        truth = dict(global_next=endpoint(0., dt=0.), same_vehicle_next=endpoint(10000.+(20. if action=='high' else 0.), dt=30.),
            fixed300=endpoint(100000.+(0. if action=='low' else -actual_gain)))
        branch = dict(status='complete', state_id=state['state_id'], selected=copy.deepcopy(selected),
            prefix=copy.deepcopy(prefix), prediction=copy.deepcopy(prediction), context=copy.deepcopy(context),
            integrity=dict(all_pass=True, checks=dict(full_native_prefix_exact=True)), source=copy.deepcopy(source),
            source_truth=copy.deepcopy(state['source_truth']), source_original_hold_seconds=0.,
            suffix=dict(seconds=300., training_delta=dict(policy=0, world=0, trust=0)),
            execution=dict(requested_hold_seconds=hold, executed_hold_seconds=hold), truth=truth,
            coverage=dict(innovation=dict(endpoint_coverage={name: {line: dict(n=12, zero_holding=6, high_holding=6)} for name in a.ENDPOINTS},
                diagnostics={name: dict(status='fitted', background_rows=20, first_rows=12, second_rows=8,
                    first_coverage={line: dict(n=12, zero_holding=6, high_holding=6)}, cold_start_first_rows=1,
                    snapshot_scope_valid=True, action_design=dict(rows=12, columns=9, rank=9, active_columns=9, ridge=1., ridge_precision_condition=100.),
                    innovation_error_rmse=dict(dt=0, wait=5000., ride=7000., cost=15000.), score_error_rmse=dict(dt=0, wait=4000., ride=6000., cost=13000.)) for name in a.ENDPOINTS})),
            native=dict(prefix_matched_issues=index+1, other_batch_actions=[]),
            budget=dict(prefix_sim_seconds=selected['started_at'], suffix_sim_seconds=300., replay_target_labels=index*5,
                replay_policy_updates=index*8, truth_labels_enter_training=0))
        branches.append(branch)
    return state, branches


def write_branches(root, state, branches, *, preflight=False):
    for action, branch in zip(('low', 'high'), branches):
        path = a.branch_path(root, state, action, preflight=preflight)
        path.mkdir(parents=True)
        (path/'summary.json').write_text(json.dumps(branch))


def test_independent_cells_load_their_own_prefix_truth_and_budget(tmp_path):
    rows = [fixture_state(index=1, actual_gain=-7), fixture_state(line='102X', window='post_shock', index=9, actual_gain=123)]
    for state, branches in rows:
        write_branches(tmp_path, state, branches)
    result = a.analyze(tmp_path, dict(states=[state for state, _ in rows]))
    assert result['integrity']['all_pass']
    assert [row['truth']['fixed300']['gain'] for row in result['pairs']] == [-7, 123]
    assert [row['selected']['sequence'] for row in result['pairs']] == [1, 9]
    assert result['budget']['replay_target_labels'] == 100
    assert result['budget']['suffix_sim_seconds'] == 1200


def test_source_continued_learning_truth_is_descriptive_not_exact_gate():
    state, branches = fixture_state(actual_gain=5)
    result = a.analyze_pair(state, *branches)
    assert result['integrity']['all_pass']
    reference = result['source_vs_frozen_nominal']['endpoints']['fixed300']
    assert reference['frozen_minus_source'][-1] == 30000
    assert result['background']['fixed300']['physical_mean']['cost'] == 4000
    assert result['background_residuals']['fixed300'][0][3] == -4000
    assert result['truth']['fixed300']['gain'] == 5


@pytest.mark.parametrize('field', ['training', 'budget', 'snapshot', 'clock', 'command'])
def test_actual_suffix_and_timing_failures_are_detected(field):
    state, branches = fixture_state()
    if field == 'training': branches[1]['suffix']['training_delta']['policy'] = 1
    if field == 'budget': branches[1]['budget']['replay_target_labels'] += 5
    if field == 'snapshot': branches[0]['context']['innovation_background']['endpoints']['fixed300']['max_available_at'] = 99999
    if field == 'clock': branches[0]['truth']['fixed300']['dt'] = 299
    if field == 'command': branches[1]['execution']['executed_hold_seconds'] = 0
    result = a.analyze_pair(state, *branches)
    assert result['integrity']['all_pass'] is False


def test_unsupported_prediction_remains_missing_and_changes_common_cohort():
    first, first_branches = fixture_state(unsupported=('innovation', 'innovation_line'))
    second, second_branches = fixture_state(line='102X', index=2)
    rows = [a.analyze_pair(first, *first_branches), a.analyze_pair(second, *second_branches)]
    result = a.summarize(rows, 'fixed300')
    assert result['model_metrics']['baseline'][0] == 2
    assert result['model_metrics']['innovation'][0] == 1
    assert result['common_n'] == 1
    assert result['common_state_ids'] == [second['state_id']]
    assert rows[0]['effects']['fixed300']['innovation'] is None
    assert rows[0]['truth']['fixed300']['status'] == 'complete'


def test_zero_clock_raw_unsupported_is_not_fabricated_zero():
    state, branches = fixture_state()
    for branch in branches:
        branch['prediction']['innovation']['raw_global_next_gain'] = None
    state['prediction'] = copy.deepcopy(branches[0]['prediction'])
    result = a.analyze_pair(state, *branches)
    assert result['duration'] == 'zero_dt'
    assert result['effects']['global_next']['innovation'][0] == 0
    assert result['raw_global_effects']['innovation'] is None
    assert result['global_to_fixed300_proxy']['global58'] is not None
    assert 'global58' not in result['effects']['fixed300']


def test_missing_selection_is_complete_audit_without_fake_truth_but_blocks_preflight(tmp_path):
    state, _ = fixture_state()
    state.update(status='selection_missing', selected=None, prediction=None)
    missing = dict(status='selection_missing', budget=dict(prefix_sim_seconds=0, suffix_sim_seconds=0,
        replay_target_labels=0, replay_policy_updates=0, truth_labels_enter_training=0))
    write_branches(tmp_path, state, [missing, missing])
    result = a.analyze(tmp_path, dict(states=[state]))
    assert result['status'] == 'complete' and result['integrity']['all_pass']
    assert result['missing']['selection'] == 1 and result['missing']['truth']['fixed300'] == 1
    assert result['overall']['fixed300']['model_metrics']['baseline'][0] == 0
    assert result['overall']['fixed300']['model_metrics']['baseline'][-1] is None
    assert result['decision']['review_ready'] is False
    pref = tmp_path/'preflight'
    write_branches(pref, state, [missing, missing], preflight=True)
    assert a.analyze(pref, dict(states=[state]), preflight=True)['integrity']['all_pass'] is False


def twelve_states():
    rows = []
    for scene in ('stable', 'local_speed_drop'):
        for i, line in enumerate(('7X', '102X', '705X')):
            for window in ('pre_shock', 'post_shock'):
                # The three pre-shock states physically repeat across scene labels.
                index = 10+i if window == 'pre_shock' else 20+i+(3 if scene=='local_speed_drop' else 0)
                state, branches = fixture_state(scene, line, window, index=index, actual_gain=index-15)
                for branch in branches:
                    branch['coverage']['innovation_line'] = copy.deepcopy(branch['coverage']['innovation'])
                    for diag in branch['coverage']['innovation_line']['diagnostics'].values():
                        diag['action_design']['columns'] = 3
                        diag['action_design']['rank'] = 3
                rows.append((state, branches))
    return rows


def test_twelve_design_cells_are_nine_physical_prefixes_and_fit_small_summary(tmp_path):
    rows = twelve_states()
    for state, branches in rows:
        write_branches(tmp_path, state, branches)
    result = a.analyze(tmp_path, dict(states=[state for state, _ in rows]))
    assert result['integrity']['all_pass']
    assert result['state_counts'] == dict(design_cells=12, selected_cells=12, unique_physical_prefixes=9)
    assert len(result['duplicate_prefixes']) == 3
    assert result['overall']['fixed300']['design_n'] == 12
    assert result['unique_prefix_overall']['fixed300']['design_n'] == 9
    assert result['action_design']['unique_physical_prefixes']['n'] == 9
    assert result['action_design']['unique_physical_prefixes']['rank'] <= 9
    assert sum(row['selected_cells'] for row in result['action_design']['unique_prefix_source_coverage'].values()) == 9
    assert result['budget']['suffix_sim_seconds'] == 7200
    assert result['budget']['paired_diagnostic_labels_enter_fit'] == 0
    payload = json.dumps(result, separators=(',', ':'), allow_nan=False)
    assert len(payload.encode()) < a.MAX_BYTES


def test_expected_pre_shock_scene_mismatch_cannot_escape_as_unique_group(tmp_path):
    rows = twelve_states()
    state, branches = rows[-2]
    assert state['cell']['scenario'] == 'local_speed_drop' and state['cell']['window_name'] == 'pre_shock'
    for branch in branches:
        branch['prefix']['counters']['world'] += 1
    for state, branches in rows:
        write_branches(tmp_path, state, branches)
    result = a.analyze(tmp_path, dict(states=[state for state, _ in rows]))
    assert result['integrity']['checks']['duplicate_pre_shock_truth_exact'] is False


def test_analyzer_failure_writes_small_terminal_artifact(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{bad json')
    out = tmp_path/'result.json'
    run = subprocess.run([sys.executable, str(HERE/'analyze.py'), '--root', str(tmp_path),
        '--manifest', str(manifest), '--out', str(out)], capture_output=True, text=True)
    assert run.returncode != 0
    failure = json.loads((tmp_path/'summary_failure.json').read_text())
    assert failure['type'] == 'JSONDecodeError'
    assert not out.exists()


def test_real_v14_runtime_coverage_and_long_numeric_fields_fit_small_budget(tmp_path):
    reference = HERE.parent/'mcwm_closed_loop_assignment_innovation_v14_20261021'/'innovation_summary.json'
    old = json.loads(reference.read_text())
    count_order = old['schema']['innovation']['first_coverage_order']
    metric_order = old['schema']['innovation']['physical_metric_order']
    counter_names = list(old['pairs'][0]['suffix_training_delta']['low'])
    rows = twelve_states()
    for index, (state, branches) in enumerate(rows):
        cell = old['cells'][index % len(old['cells'])]
        coverage = {}
        for arm in ('innovation', 'innovation_line'):
            model = cell[arm+'_model']
            local = {endpoint: {line: dict(zip(count_order, values)) for line, values in by_line.items()}
                     for endpoint, by_line in model['endpoint_coverage'].items()}
            diagnostics = {}
            for endpoint, full in model['innovation_diagnostics'].items():
                row = {key: copy.deepcopy(full[key]) for key in ('status', 'background_rows', 'first_rows',
                    'second_rows', 'cold_start_first_rows', 'snapshot_scope_valid', 'action_design')}
                row['first_coverage'] = copy.deepcopy(local[endpoint])
                for key in ('innovation_error_rmse', 'score_error_rmse'):
                    row[key] = dict(zip(metric_order, full[key]))
                diagnostics[endpoint] = row
            coverage[arm] = dict(endpoint_coverage=local, diagnostics=diagnostics)
        # Actual 13 training counters, full per-line support and eight-digit
        # passenger predictions stress the field-name compression, not physics.
        for branch in branches:
            branch['coverage'] = copy.deepcopy(coverage)
            branch['prefix']['counters'] = {name: 123456.78+state['selected']['sequence'] for name in counter_names}
            branch['suffix']['training_delta'] = {name: 0 for name in counter_names}
            for model in a.MODELS:
                for action in ('low', 'high'):
                    for ep in a.ENDPOINTS:
                        branch['prediction'][model][action]['endpoints'][ep]['cost'] += 12345.678
        state['prediction'] = copy.deepcopy(branches[0]['prediction'])
        # Keep physically shared pre states' predictions identical across scenes.
        write_branches(tmp_path, state, branches)
    result = a.analyze(tmp_path, dict(states=[state for state, _ in rows]))
    assert result['integrity']['all_pass']
    assert result['schema']['coverage']['line_order'] == ['7X', '102X', '705X']
    assert len(json.dumps(result, separators=(',', ':'), allow_nan=False).encode()) < a.MAX_BYTES
