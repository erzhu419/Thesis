"""V70 admits only the declared target change and preserves actual-cost gates."""
import json

import pytest

from experiments.mcwm_closed_loop_recent30_cql_v69_20261026 import test_analyze as previous
from experiments.mcwm_closed_loop_executed_cql_v68_20261026 import test_analyze as prior
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import analyze as module


def populate(tmp_path, *, fresh=(98.,)*4):
    root, paths, cached = previous.populate(tmp_path, fresh=fresh)
    target = root.with_name('v70')
    root.rename(target)
    paths = [target/path.relative_to(root) for path in paths]
    protocol = json.loads((module.HERE/'protocol.json').read_text())
    prior.write(target/'protocol.json', protocol)
    for path in paths:
        value = module.old.read(path/'summary.json')
        value['cell']['recipe'] = protocol['recipe']
        value['integrity']['checks'].pop('fresh_value_target_scope')
        value['integrity']['checks']['fresh_mixed_target_scope'] = True
        if path.name == 'fresh':
            value['target_audit'].update(fresh_value_replacement_calls=0,
                fresh_value_replacement_rows=0, fresh_mixed_target_calls=456,
                fresh_mixed_target_rows=456*64, mixed_backup_lambda=.1,
                mixed_target_generated_before_replacement=False)
        prior.write(path/'summary.json', value)
    return target, paths, cached


def test_current_mixed_and_historical_value_targets_use_separate_audits(tmp_path):
    root, _, _ = populate(tmp_path)
    main, recency = module.analyze(root)
    assert main['status'] == 'complete' and main['integrity']['all_pass']
    assert main['decision']['scientific_gate_pass']
    assert main['recipe'] == 'recent30_mixed_cql'
    assert main['aggregates']['matched']['full_day']['overall']['mean_cost_difference'] == -1.
    assert main['aggregates']['original_no_fresh']['full_day']['overall']['mean_cost_difference'] == -2.
    assert recency['aggregates']['v68_no_fresh']['full_day']['overall']['mean_cost_difference'] == -1.
    assert recency['aggregates']['v68_fresh']['full_day']['overall']['mean_cost_difference'] == -3.
    assert all(all(row['no_fresh_exact'].values()) for row in recency['native_evidence'])
    assert all(row['targets']['fresh']['fresh_mixed_target_calls'] == 456
        for row in recency['native_evidence'])
    assert main['acceptance_protocol']['early_window'] == 'descriptive_only'
    assert not main['decision']['ordinary_dripping_goal_achieved']


@pytest.mark.parametrize('field,value', (
    ('fresh_value_replacement_calls',456), ('fresh_mixed_target_calls',455),
    ('mixed_backup_lambda',.2), ('fresh_mixed_target_rows',456*63),
    ('mixed_target_generated_before_replacement',True)))
def test_value_only_or_wrong_mixed_target_blocks_admission(tmp_path, field, value):
    root, paths, _ = populate(tmp_path)
    prior.modify(paths[0]/'summary.json', ('target_audit',field), value)
    main, _ = module.analyze(root)
    assert main['status'] == 'integrity_failed'
    assert not main['decision']['scientific_gate_pass']


def test_historical_v68_fresh_value_target_cannot_be_reclassified(tmp_path):
    root, _, cached = populate(tmp_path)
    path = cached/'runs/executed_cql/development/seed6201/stable/fresh/summary.json'
    prior.modify(path, ('target_audit','fresh_value_replacement_calls'), 0)
    main, _ = module.analyze(root)
    assert main['status'] == 'integrity_failed'
    assert any('v68_fresh/only_declared_fresh_rows_replaced' in name
        for name in main['integrity']['failed_checks'])


@pytest.mark.parametrize('failure', ('late_trace','ledger','target_path','metrics'))
def test_no_fresh_must_reproduce_complete_v68_path(tmp_path, failure):
    root, paths, _ = populate(tmp_path)
    path = paths[1]
    if failure == 'late_trace':
        prior.jsonl(path/'prefix_trace.jsonl', [dict(started_at=1000,a=0),
            dict(started_at=1300,a=1.875)])
    elif failure == 'ledger':
        prior.modify(path/'summary.json', ('cost','accrued_wait_seconds'),2001.)
    elif failure == 'target_path':
        prior.modify(path/'summary.json', ('target_audit','rule'), 'changed')
    else:
        rows = module.old.read_metrics(path,module.old.read(path/'summary.json'))
        rows[0]['accrued_wait_seconds'] += 1
        prior.jsonl(path/'metrics.jsonl', rows)
    main, recency = module.analyze(root)
    assert main['status'] == 'integrity_failed'
    assert any('/matched_v68_no_fresh/' in name for name in recency['integrity']['failed_checks'])


@pytest.mark.parametrize('fresh', ((99.,)*4, (94.,102.,94.,102.)))
def test_fresh_zeroed_or_shock_harm_cannot_pass(tmp_path, fresh):
    root, _, _ = populate(tmp_path,fresh=fresh)
    main, _ = module.analyze(root)
    assert main['integrity']['all_pass']
    assert not main['decision']['matched_cost_gate_pass']
    assert not main['decision']['scientific_gate_pass']


def test_worst_cell_can_close_an_average_winner(tmp_path):
    root, _, _ = populate(tmp_path,fresh=(102.,95.,95.,95.))
    main, _ = module.analyze(root)
    assert main['decision']['matched_cost_gate_pass']
    assert main['decision']['original_reference_gate_pass']
    assert not main['decision']['worst_cell_gate_pass']
    assert not main['decision']['scientific_gate_pass']


def test_early_harm_descriptive_but_tail_harm_closes_candidate(tmp_path):
    root, paths, _ = populate(tmp_path)
    for tail_fraction in (.8,.5):
        for path in paths[::2]:
            value = module.old.read(path/'summary.json')['cost']
            prior.jsonl(path/'metrics.jsonl',[dict(time_seconds=t,
                accrued_wait_seconds=value['accrued_wait_seconds']*f,
                accrued_in_vehicle_seconds=value['accrued_in_vehicle_seconds']*f)
                for t,f in ((9000,.1),(12000,.45),(18300,tail_fraction),(24000,1.))])
        main, _ = module.analyze(root)
        assert main['aggregates']['matched']['early']['overall']['mean_cost_difference'] > 0
        assert main['decision']['matched_cost_gate_pass']
        assert main['decision']['scientific_gate_pass'] is (tail_fraction == .8)


def test_native_size_export_preserves_costs_heads_lines_and_three_pull_caps(tmp_path):
    root, paths, _ = populate(tmp_path)
    lines=('102S','102X','122S','122X','311S','311X','406S','406X','705S','705X','7S','7X')
    for path in paths:
        value=module.old.read(path/'summary.json')
        group=value['policy_drift']['groups'][0]
        value['policy_drift']['groups']=[dict(group,line=line,window=window,
            count=5 if i==35 else 8) for i,(line,window) in enumerate(
                (line,window) for line in lines for window in ('early','late_training','pre_early'))]
        prior.write(path/'summary.json',value)
    for file in tmp_path.rglob('summary.json'):
        value=module.old.read(file)
        scalar=value['cost']['generalized_seconds_per_departed_passenger']
        total=15079*(2209.123456789+scalar-100.)
        wait=8438598.256789+23.5136789123*scalar
        ride=total-2*wait
        value['cost']=dict(departed_passengers=15079,accrued_wait_seconds=wait,
            accrued_in_vehicle_seconds=ride,generalized_passenger_seconds=2*wait+ride,
            generalized_seconds_per_departed_passenger=(2*wait+ride)/15079)
        if 'cql_audit' in value: value['cql_audit']['regularizer_wall_seconds']=36.123456789
        prior.write(file,value)
        prior.jsonl(file.parent/'metrics.jsonl',[dict(time_seconds=t,
            accrued_wait_seconds=wait*f,accrued_in_vehicle_seconds=ride*f)
            for t,f in ((9000,.300123456),(12000,.450234567),(18300,.800345678),(24000,1.))])
    main,extra=module.analyze(root)
    assert main['integrity']['all_pass']
    receipt=module.export(root,root/'results/development.json')
    assert set(receipt['files']) == {'development.json','policy_drift.json','recency.json'}
    assert all(size<=48000 for size in receipt['files'].values())
    assert module.old.read(root/'results/development.json')['aggregates'] == main['aggregates']
    assert module.old.read(root/'results/recency.json') == extra
    policy=module.old.read(root/'results/policy_drift.json')
    assert all(len(arm['rows'])==36 and len({row[0] for row in arm['rows']})==12
        for cell in policy['native_evidence'] for arm in cell['policy_drift'].values())
    cost=module.old.read(root/'results/development.json')
    assert all(len(arm['head_rows'])==5 for cell in cost['native_evidence']
        for arm in cell['executed_cql'].values())
