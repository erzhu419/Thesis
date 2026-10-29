"""Meaningful gates: causal replay, scoreable reversals and native concurrency."""
from copy import deepcopy
import json

import pytest

from experiments.mcwm_closed_loop_stratified_pairs_v23_20261023 import analyze as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def history(at, value=1.):
    return dict(status='supported', mean_residual_scaled=value/60,
        segments=[dict(available=True, latest=dict(completed_at=at-10, observed_at=at-9,
            residual=value, age=10.)) for _ in range(3)])


def source(identifier, line, at, count=0, window='bootstrap'):
    return dict(id=identifier, role=identifier.split('/')[0], line_id=line, window_name=window,
        selected=dict(started_at=at, sequence=int(at)), features=[float(line == r) for r in a.LINES]+[.2],
        prefix=dict(external_seed=96201, counters=dict(service_shared_pairs=count, service_history_pairs=count)),
        prediction={}, service_history=history(at), service_prediction={})


def branches(issue, truth=None):
    truth = truth or [6.]*6
    result = []
    for arm, hold in (('low', 0.), ('high', 7.5)):
        value = deepcopy(issue)
        value.update(status='complete', integrity=dict(all_pass=True, checks={k: True for k in a.old.PREFIX_CHECKS}),
            native=dict(other_batch_actions=[]), execution=dict(requested_hold_seconds=hold, executed_hold_seconds=hold),
            suffix=dict(seconds=300, training_delta=dict(policy=0, world=0, service_refits=0)),
            budget=dict(truth_labels_enter_training=0, prefix_sim_seconds=issue['selected']['started_at'],
                suffix_sim_seconds=300, replay_target_labels=20, replay_policy_updates=32),
            execution_wall=dict(pid=101 if arm == 'low' else 102, native_started_wall=10. if arm == 'low' else 11., native_finished_wall=20.))
        value['service_shift'] = dict(stations=['a', 'b', 'c'],
            arrival=[100.+(truth[i] if arm == 'high' else 0.) for i in range(3)],
            departure=[110.+(truth[i+3] if arm == 'high' else 0.) for i in range(3)],
            status=dict(arrival=['complete']*3, departure=['complete']*3))
        result.append(value)
    return result


def fixture(root, *, scored_reversal=True):
    p = dict(service_seed=6201, target_seed_offset=90000, target_lines=list(a.LINES),
        heldout_windows=[dict(name=w) for w in ('heldout1', 'heldout2')])
    write(root/'protocol.json', p)
    path = root/'stage_a/runs/stable/seed6201'
    issues, training, metadata, receipt, native = [], [], [], [], []
    for window_index, (window, start) in enumerate((('bootstrap', 1200), ('response', 3000), ('late', 15000))):
        for line_index, line in enumerate(a.LINES):
            index = window_index*3+line_index
            identifier, at = 'train/'+line+'/'+window, start+line_index*10
            issue = source(identifier, line, at, window_index*3, window)
            issue['service_history'] = history(at, (-8., 1., 9.)[window_index])
            targets = [-1.5]*6
            if window_index == 0 or (window_index == 2 and scored_reversal):
                targets[2] = targets[5] = -10.
            record = dict(id=identifier, index=index, role='train', line_id=line, features=issue['features'],
                issued_at=at, available_at=at+330, service_history=issue['service_history'],
                service_targets=dict(arrival=targets[:3], departure=targets[3:]), selected_for_training=True, admitted_at=at+400)
            training.append(record)
            issues.append(issue)
            metadata.append(dict(id=identifier, admitted=True, admitted_at=at+400))
            receipt.append(dict(pair_id=identifier, native_wall={arm:dict(pid=pid, native_started_wall=start,
                native_finished_wall=start+8.) for arm, pid, start in (('low', 101, 1.), ('high', 102, 2.))},
                parent_native_before=at, parent_native_after=at, all_pass=True, overlap_seconds=7.))
    for issue, record in zip(issues, training):
        issue['service_prediction'] = {m:a.replay(training, issue, m)[0] for m in a.MODELS}
        write(path/'pair_cache'/(str(record['index'])+'.json'), record)
        for arm, value in zip(('low', 'high'), branches(issue, [t+7.5 for t in a.flatten(record['service_targets'])])):
            write(path/'training_branches'/str(record['index'])/arm/'summary.json', value)
            native.append(dict(status='complete', budget=value['budget'], integrity=value['integrity']))
    for line in a.LINES:
        for w in ('heldout1', 'heldout2'):
            issue = source('heldout/'+line+'/'+w, line, 18500., 9, w)
            issue['service_prediction'] = {m:dict(status='supported', arrival=[6., 6., -2.] if m == 'history' else [7.5]*3,
                departure=[6., 6., -2.] if m == 'history' else [7.5]*3) for m in a.MODELS}
            issues.append(issue)
            for arm, value in zip(('low', 'high'), branches(issue, [6., 6., -2.]*2)):
                write(root/'stage_a/pairs/stable/seed6201'/line/w/arm/'summary.json', value)
            write(root/'stage_a/pairs/stable/seed6201'/line/w/'parallel_execution.json',
                dict(status='complete', integrity=dict(all_pass=True)))
    day = dict(status='complete', integrity=dict(all_pass=True),
        initializer=dict(initial_updates=50000, history_rows=32768, source_rows=4096),
        budget=dict(target_labels=285, ordinary_labels=267, paired_action_labels=18, training_pairs=9,
            paired_outcomes_per_pair=2, native_seconds=24000, policy_updates=456, trust_updates=456,
            admitted_pair_ids=[r['id'] for r in training]), design_coverage=dict(all_pass=True),
        readouts=dict(shared={}, route={}), acquisitions=issues, training_pairs=metadata, training_branches=native,
        training_branch_budget=dict(prefix_sim_seconds=1000., suffix_sim_seconds=5400., replay_target_labels=100),
        sampling_summary=dict(all_pass=True, checks=dict(exactly_nine_cells=True, selected_once=True, selected_only_planned=True, all_planned_cells_selected=True, three_pairs_per_line=True), by_line={line:dict(unique_joint_strata_count=2) for line in a.LINES}),
        parallel_pairs=receipt)
    write(path/'summary.json', day)
    write(path/'acquisitions.json', dict(records=issues))
    return path


def test_training_gate_has_chronological_support_and_does_not_select_on_mae(tmp_path):
    fixture(tmp_path)
    result = a.analyze(tmp_path, 'train')
    assert result['status'] == 'complete'
    assert result['training_gate']['passed'] and result['service_validation_eligible']
    assert result['prediction_replay']['max_absolute_error'] == 0.
    assert result['coverage']['scored_states'] == 6
    assert len(result['coverage']['scored_reversal_states']) == 3
    assert result['budget']['training_labels'] == 285
    assert result['decision']['stage_b'] == 'closed'
    assert len(json.dumps(result)) < 42000


def test_cold_start_negative_truth_does_not_open_service_validation(tmp_path):
    fixture(tmp_path, scored_reversal=False)
    result = a.analyze(tmp_path, 'train')
    assert result['integrity']['all_pass']
    assert result['coverage']['unscored_reversal_endpoints'] == 6
    assert result['coverage']['scored_reversal_endpoints'] == 0
    assert not result['training_gate']['passed']
    assert a.analyze(tmp_path, 'service')['status'] == 'blocked_by_train_gate'


def test_issue_state_diversity_is_separate_from_structural_integrity(tmp_path):
    source_path = fixture(tmp_path)
    day = a.read(source_path/'summary.json')
    day['sampling_summary']['by_line']['7X']['unique_joint_strata_count'] = 1
    write(source_path/'summary.json', day)
    result = a.analyze(tmp_path, 'train')
    assert result['integrity']['all_pass']
    assert not result['training_gate']['checks']['issue_state_diversity']
    assert not result['service_validation_eligible']


def test_history_equal_to_issue_is_not_allowed_and_future_truth_cannot_fit(tmp_path):
    source_path = fixture(tmp_path)
    journal = a.read(source_path/'acquisitions.json')
    journal['records'][0]['service_history']['segments'][0]['latest']['observed_at'] = 1200
    write(source_path/'acquisitions.json', journal)
    result = a.analyze(tmp_path, 'train')
    assert not result['integrity']['all_pass']
    record = dict(id='later', role='train', selected_for_training=True, issued_at=20., available_at=200.,
        admitted_at=210., line_id='7X', features=[1.,0.,0.,.2], service_history=history(20.),
        service_targets=dict(arrival=[0.]*3, departure=[0.]*3))
    with pytest.raises(ValueError, match='chronological release'):
        a.replay([record], source('train/7X/next', '7X', 100., 1), 'history')


def test_parallel_audit_requires_native_phase_overlap_and_paused_parent():
    row = dict(pair_id='p', native_wall=dict(low=dict(pid=101,native_started_wall=1.,native_finished_wall=3.),
        high=dict(pid=102,native_started_wall=2.,native_finished_wall=4.)),
        launches=dict(low=dict(started_wall=0.,ended_wall=8.), high=dict(started_wall=0.,ended_wall=8.)),
        parent_native_before=1200.,parent_native_after=1200.,all_pass=True)
    assert a.parallel_audit([row], ['p'])['all_pass']
    row['native_wall']['high']['native_started_wall'] = 3.5
    assert not a.parallel_audit([row], ['p'])['all_pass']
    row['native_wall']['high']['native_started_wall'] = 2.
    row['parent_native_after'] = 1201.
    assert not a.parallel_audit([row], ['p'])['all_pass']


def test_service_gate_requires_fresh_reversal_hit_and_never_opens_b(tmp_path):
    fixture(tmp_path)
    value = a.analyze(tmp_path, 'service')
    assert value['status'] == 'complete' and value['scientific_gate']['passed']
    assert value['scientific_gate']['reversal']['history_hits'] == 12
    assert value['budget']['heldout_branches_completed'] == 12
    assert value['decision']['stage_b'] == 'closed_pending_manual_review'
    journal_path = tmp_path/'stage_a/runs/stable/seed6201/acquisitions.json'
    journal = a.read(journal_path)
    for issue in journal['records']:
        if issue['role'] != 'heldout':
            continue
        issue['service_prediction']['history'] = dict(status='supported', arrival=[6.]*3, departure=[6.]*3)
        for arm, branch in zip(('low', 'high'), branches(issue, [6.]*6)):
            write(tmp_path/'stage_a/pairs/stable/seed6201'/issue['line_id']/issue['window_name']/arm/'summary.json', branch)
    write(journal_path, journal)
    value = a.analyze(tmp_path, 'service')
    assert value['scientific_gate']['checks']['macro_mae_improved']
    assert not value['scientific_gate']['checks']['fresh_reversal_state']
    assert not value['scientific_gate']['passed']


def test_missing_or_unadmitted_pairs_are_scientific_coverage_not_integrity_failure(tmp_path):
    source_path = fixture(tmp_path)
    day = a.read(source_path/'summary.json')
    removed = day['budget']['admitted_pair_ids'].pop()
    day['budget'].update(training_pairs=8, paired_action_labels=16, ordinary_labels=269)
    day['training_pairs'][-1]['admitted'] = False
    write(source_path/'summary.json', day)
    unadmitted = a.analyze(tmp_path, 'train')
    assert unadmitted['status'] == 'complete' and unadmitted['integrity']['all_pass']
    assert unadmitted['coverage']['unadmitted_ids'] == [removed]
    assert unadmitted['coverage']['scored_states'] == 5
    assert not unadmitted['training_gate']['passed']
    journal = a.read(source_path/'acquisitions.json')
    journal['records'] = [r for r in journal['records'] if r['id'] != removed]
    write(source_path/'acquisitions.json', journal)
    day['sampling_summary']['checks'].update(all_planned_cells_selected=False, three_pairs_per_line=False)
    day['sampling_summary'].update(all_pass=False, missing_ids=[removed])
    day['parallel_pairs'].pop()
    day['training_pairs'].pop()
    day['training_branches'] = day['training_branches'][:-2]
    write(source_path/'summary.json', day)
    (source_path/'pair_cache/8.json').unlink()
    missing = a.analyze(tmp_path, 'train')
    assert missing['status'] == 'complete' and missing['integrity']['all_pass']
    assert missing['coverage']['missing_selected_ids'] == [removed]
    assert not missing['training_gate']['passed']


def test_service_summary_waits_for_pair_launcher_acceptance(tmp_path):
    fixture(tmp_path)
    pair = tmp_path/'stage_a/pairs/stable/seed6201/7X/heldout1'
    (pair/'parallel_execution.json').unlink()
    assert a.analyze(tmp_path, 'service')['status'] == 'incomplete'
    write(pair/'failure.json', dict(status='failed', message='native phases did not overlap'))
    assert a.analyze(tmp_path, 'service')['status'] == 'failed'
