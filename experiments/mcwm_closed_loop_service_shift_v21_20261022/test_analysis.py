import copy
import json
from pathlib import Path

from experiments.mcwm_closed_loop_service_shift_v21_20261022 import analyze as a


def history(at=100):
    return dict(status='supported', mean_residual_scaled=.2,
        segments=[dict(latest=dict(completed_at=at-10-i, observed_at=at-9-i, age=10+i)) for i in range(3)])


def acquisition(line='7X', at=100, *, response=False):
    prediction = {name: dict(status='supported', gain=dict(dt=0., wait=0., ride=value, cost=value),
        fixed300_clock=dict(low=300., high=300.)) for name, value in
        [('shared', -3.), ('route', -4.)]+([('candidate', 4.)] if response else [])}
    return dict(id='heldout/'+line+'/heldout1', line_id=line, selected=dict(started_at=at, sequence=1),
        prefix=dict(external_seed=85201), prediction=prediction, service_history=history(at),
        service_prediction={name: dict(status='supported', arrival=[value]*3, departure=[value]*3)
            for name, value in (('shared', 7.5), ('history', 6.))})


def branches(source):
    result = []
    for hold, shift in ((0., 0.), (7.5, 6.)):
        value = copy.deepcopy(source)
        value.update(status='complete', integrity=dict(all_pass=True, checks={k: True for k in a.PREFIX_CHECKS}),
            execution=dict(requested_hold_seconds=hold, executed_hold_seconds=hold),
            native=dict(other_batch_actions=[]), suffix=dict(seconds=300, training_delta=dict(policy=0, service_refits=0)),
            budget=dict(truth_labels_enter_training=0, prefix_sim_seconds=source['selected']['started_at'],
                suffix_sim_seconds=300, replay_target_labels=200, replay_policy_updates=320),
            service_shift=dict(stations=['next1', 'next2', 'next3'],
                arrival=[70+shift, 110+shift, 190+shift], departure=[80+shift, 120+shift, 200+shift],
                status=dict(arrival=['complete']*3, departure=['complete']*3)))
        wait, ride = (20., 30.) if hold == 0 else (21., 32.)
        value['truth'] = {endpoint: dict(status='complete', dt=dt, wait=wait, ride=ride, cost=2*wait+ride)
            for endpoint, dt in (('global_next', 0. if source['line_id'] == '705X' else 5.),
                                 ('same_vehicle_next', 75. if hold == 0 else 81.), ('fixed300', 300.))}
        result.append(value)
    return result


def test_service_score_uses_absolute_high_minus_low_not_training_residual():
    source = acquisition()
    low, high = branches(source)
    value = a.service_score(source, low, high)
    assert value['mae'] == dict(shared=1.5, history=0.)
    assert value['rows'][0][2:] == [6., 7.5, 6.]


def test_service_gate_cannot_pass_on_first_stop_only_and_retains_censoring():
    source = acquisition()
    low, high = branches(source)
    for value in (low, high):
        for kind in a.KINDS:
            value['service_shift'][kind][1:] = [None, None]
            value['service_shift']['status'][kind][1:] = ['censored', 'censored']
    scored = a.service_score(source, low, high)
    assert scored['status'] == 'unscored'
    assert not scored['downstream_observed']
    assert scored['missing']['high/arrival/censored'] == 2
    assert scored['rows'][1][2] is None


def test_supported_history_at_issue_is_not_causal():
    value = history()
    assert a.history_coverage(value, 100)['causal']
    value['segments'][0]['latest']['observed_at'] = 100
    assert not a.history_coverage(value, 100)['causal']


def test_service_gate_is_macro_over_states_and_rejects_one_line_regression():
    cases = [dict(line_id=line, history=dict(supported=True, causal=True),
        service=dict(status='scored', mae=dict(shared=2., history=1.))) for line in a.LINES for _ in range(2)]
    gate = a.scientific_gate(cases, dict(all_pass=True))
    assert gate['passed'] and gate['independent_states'] == 6
    cases[-1]['service']['mae']['history'] = 4.
    gate = a.scientific_gate(cases, dict(all_pass=True))
    assert gate['checks']['macro_mae_improved']
    assert not gate['checks']['no_line_regression']
    assert not gate['passed']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, stage):
    write(root/'protocol.json', dict(target_lines=list(a.LINES), heldout_windows=[dict(name=x) for x in ('heldout1', 'heldout2')]))
    base = root/('stage_a' if stage == 'service' else 'stage_b')
    for seed, scene in a.CELLS[stage]:
        acquisitions = []
        for line in a.LINES:
            for window in ('heldout1', 'heldout2'):
                source = acquisition(line, response=stage == 'response')
                source['prefix']['external_seed'] = seed+(79000 if stage == 'service' else 80000)
                source['id'] = f'heldout/{line}/{window}'
                acquisitions.append(source)
                for arm, value in zip(('low', 'high'), branches(source)):
                    write(base/'pairs'/scene/f'seed{seed}'/line/window/arm/'summary.json', value)
        day = dict(status='complete', integrity=dict(all_pass=True),
            initializer=dict(initial_updates=50000, history_rows=32768, source_rows=4096),
            budget=dict(target_labels=285, ordinary_labels=267, paired_action_labels=18,
                training_pairs=9, paired_outcomes_per_pair=2, native_seconds=24000,
                policy_updates=456, trust_updates=456),
            design_coverage=dict(all_pass=True), acquisitions=acquisitions,
            readouts={name: {} for name in (('shared', 'route') if stage == 'service' else ('shared', 'route', 'candidate'))},
            training_branch_budget=dict(prefix_sim_seconds=1000, suffix_sim_seconds=5400,
                replay_target_labels=100, replay_policy_updates=160), training_branches=[{}]*18)
        write(base/'runs'/scene/f'seed{seed}'/'summary.json', day)


def test_full_service_stage_gates_without_a_dummy_cost_candidate(tmp_path):
    fixture(tmp_path, 'service')
    value = a.analyze(tmp_path, 'service')
    assert value['status'] == 'complete' and value['scientific_gate']['passed']
    assert value['budget']['training_labels_observed'] == 285
    assert value['budget']['heldout_branch_completed'] == 12
    assert 'response' not in value
    assert len(json.dumps(value, separators=(',', ':')).encode()) < 20000
    path = tmp_path/'stage_a/runs/stable/seed6201/summary.json'
    day = json.loads(path.read_text()); day['readouts']['candidate'] = {}
    write(path, day)
    value = a.analyze(tmp_path, 'service')
    assert value['status'] == 'integrity_failed'
    assert not value['scientific_gate']['passed']


def test_response_uses_only_three_new_days_and_exposes_stable_degradation(tmp_path):
    fixture(tmp_path, 'response')
    value = a.analyze(tmp_path, 'response')
    assert value['status'] == 'complete'
    assert value['budget']['training_labels_observed'] == 855
    assert value['budget']['training_branch_count'] == 54
    assert value['budget']['heldout_branch_completed'] == 36
    assert value['response']['stable_comparison']['shared'] == dict(states=6,
        candidate_minus_baseline_regret=24., harmful_states=6, improved_states=0)
    assert value['response']['duration'] == dict(positive_dt=12, zero_dt=6)
    assert len(json.dumps(value, separators=(',', ':')).encode()) < 50000


def test_candidate_unsupported_is_never_reported_as_fixed_zero(tmp_path):
    source = acquisition(response=True)
    source['prediction']['candidate'] = dict(status='unsupported', gain=None)
    low, high = branches(source)
    row = a.response_row(source, low, high)
    cases = [dict(scenario='stable', line_id='7X', window='heldout1', prediction=source['prediction'], response=row)]
    value = a.response_summary(cases)
    assert value['unsupported']['candidate'] == 1
    assert value['overall']['candidate'][0] == 0
    assert value['stable_comparison']['zero']['states'] == 0
    assert value['stable_comparison']['zero']['candidate_minus_baseline_regret'] is None


def test_terminal_integrity_failure_does_not_wait_for_other_branches(tmp_path):
    fixture(tmp_path, 'service')
    bad = tmp_path/'stage_a/pairs/stable/seed6201/7X/heldout1/low/summary.json'
    value = json.loads(bad.read_text()); value['integrity']['all_pass'] = False
    write(bad, value)
    pending = tmp_path/'stage_a/pairs/stable/seed6201/705X/heldout2/low/summary.json'
    pending.unlink()
    assert a.analyze(tmp_path, 'service')['status'] == 'integrity_failed'
