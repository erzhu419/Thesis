"""Both fixed cases are allocated together and no result expands this diagnostic."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mcwm_closed_loop_mid_policy_exchange_v76_20261027 import launch as mod


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    helper = mod.module('v76_test_readonly', mod.V60/'supervisor.py')
    protocol = dict(seeds=[6208, 6207], native_preflight=dict(seed=6208, scenario='stable',
        reuse_prefix=True, feedback_slots=5, policy_updates=8, critic_updates=8,
        value_updates=8, actor_updates=8, alpha_updates=8))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'execution_ready.json').write_text('{"ready":true}')
    outputs, submissions, states, collections = {}, [], {}, []

    def task(label, runner, argv):
        return dict(signature='v76/'+label, runner=str(runner), argv=argv, priority='high')

    def submit(rows, batch):
        path = tmp_path/(batch+'_submission.json')
        assert not path.exists(), 'Existing tasks must be reused'
        value = dict(submitted=[dict(id=f't{len(submissions)}_{i}', signature=row['signature']) for i, row in enumerate(rows)])
        path.write_text(json.dumps(value))
        submissions.append(dict(batch=batch, rows=rows))
        return value

    def collect(path):
        collections.append(path)
        return outputs[path]

    infra = SimpleNamespace(task=task, remote=str, submit=submit,
        presence=lambda paths: {path: path in outputs for path in paths}, collect=collect,
        status=lambda ids: [dict(id=i, status=states.get(i, 'running')) for i in ids],
        common_prefix=lambda rows: dict(all_pass=True, rows_compared=45))
    monkeypatch.setattr(mod, 'HERE', tmp_path)
    monkeypatch.setattr(mod, 'module', lambda name, path: helper if path.name == 'supervisor.py' else infra)

    def preflight():
        for arm in ('fresh', 'no_fresh'):
            outputs[f'runs/seed6208/{arm}/preflight.json'] = dict(technical_gate_pass=True,
                seed=6208, scenario='stable', arm=arm, feedback_slots=5, policy_updates=8,
                critic_updates=8, value_updates=8, actor_updates=8, alpha_updates=8,
                packet_schedule=[8]*4, initializer=dict(updates=50000),
                first_feedback=dict(time=1205), execution_contract=dict(F='unchanged'),
                prefix_before_first_update=dict(trace_path='resident', rows=45, clock=1205., initial_updates=50000))

    def native_done():
        preflight()
        for seed, identity in ((6208, 't0_0'), (6207, 't0_1')):
            outputs[f'runs/seed{seed}/seed_summary.json'] = dict(status='complete',
                integrity=dict(all_pass=True, checks=dict(native=True)))
            states[identity] = 'done'

    return SimpleNamespace(root=tmp_path, infra=infra, outputs=outputs, submissions=submissions,
        states=states, collections=collections, preflight=preflight, native_done=native_done)


def test_both_seed_jobs_start_together_and_existing_receipt_is_reused(campaign):
    current = mod.advance()
    mod.advance()
    assert current['stage'] == 'mid_policy_exchange_native_preflight'
    assert len(campaign.submissions) == 1
    jobs = campaign.submissions[0]['rows']
    assert [job['argv'][1] for job in jobs] == [6208, 6207]
    assert all(Path(job['runner']).name == 'pair.py' and job['priority'] == 'high' for job in jobs)
    assert all('--stop' not in job['argv'] for job in jobs)


def test_preflight_is_reused_without_another_native_allocation(campaign):
    campaign.preflight()
    assert mod.advance()['stage'] == 'mid_policy_exchange_native'
    mod.advance()
    assert len(campaign.submissions) == 1 and len(campaign.submissions[0]['rows']) == 2


def test_bad_preflight_closes_without_replacement_or_case_expansion(campaign):
    campaign.preflight()
    campaign.outputs['runs/seed6208/no_fresh/preflight.json']['initializer'] = dict(updates=49999)
    result = mod.advance()
    assert result['done'] and result['requires_agent_review']
    assert result['native_tasks_may_still_run']
    assert len(campaign.submissions) == 1 and len(campaign.submissions[0]['rows']) == 2


def test_missing_terminal_seed_summary_does_not_become_zero_cost(campaign):
    campaign.preflight()
    campaign.states['t0_0'] = 'done'
    current = mod.advance()
    assert current['stage'] == 'mid_policy_exchange_native_attention'
    assert current['completed'] == 0
    assert current['failures'][0]['reason'] == 'terminal_seed_summary_missing'


def test_summary_waits_for_both_fixed_seed_tasks(campaign):
    campaign.preflight()
    campaign.outputs['runs/seed6208/seed_summary.json'] = dict(status='complete', integrity=dict(all_pass=True, checks=dict(native=True)))
    campaign.states['t0_0'] = 'done'
    assert mod.advance()['stage'] == 'mid_policy_exchange_native'
    assert len(campaign.submissions) == 1
    campaign.native_done()
    assert mod.advance()['stage'] == 'mid_policy_exchange_summary'
    mod.advance()
    assert len(campaign.submissions) == 2
    assert len(campaign.submissions[-1]['rows']) == 1
    assert Path(campaign.submissions[-1]['rows'][0]['runner']).name == 'analyze.py'


@pytest.mark.parametrize('supported', [True, False])
def test_either_finite_diagnostic_outcome_stops_without_a_new_algorithm(campaign, supported):
    campaign.native_done()
    campaign.outputs['results/diagnostic.json'] = dict(status='complete', integrity=dict(all_pass=True),
        decision=dict(baseline_protection_may_be_considered=supported))
    campaign.outputs['results/evidence.json'] = dict(status='complete')
    campaign.states['t1_0'] = 'done'
    current = mod.advance()
    mod.advance()
    assert current['done'] and current['stage'] == 'mid_policy_exchange_complete_requires_root_review'
    assert current['decision']['baseline_protection_may_be_considered'] is supported
    assert current['new_recipe_started'] is False
    assert len(campaign.submissions) == 2


def test_terminal_partial_export_is_not_a_completed_analysis(campaign):
    campaign.native_done()
    campaign.outputs['results/diagnostic.json'] = dict(status='complete', integrity=dict(all_pass=True))
    campaign.states['t1_0'] = 'done'
    result = mod.advance()
    assert result['stage'] == 'mid_policy_exchange_summary_attention'
    assert result['missing_summary'] == ['results/evidence.json']


def test_transient_read_reuses_known_task_ids(campaign):
    original = campaign.infra.status
    campaign.infra.status = lambda ids: (_ for _ in ()).throw(RuntimeError('waiting for scheduler state lock'))
    assert mod.advance()['stage'] == 'waiting_transient_mid_exchange_read'
    campaign.infra.status = original
    assert mod.advance()['stage'] == 'mid_policy_exchange_native_preflight'
    assert len(campaign.submissions) == 1


def test_seed_cost_sign_never_changes_the_fixed_cases(campaign):
    campaign.preflight()
    campaign.outputs['runs/seed6208/fresh/summary.json'] = dict(cost='bad')
    campaign.outputs['runs/seed6207/fresh/summary.json'] = dict(cost='good')
    mod.advance()
    assert len(campaign.submissions[0]['rows']) == 2
    assert not any(path.endswith('/fresh/summary.json') for path in campaign.collections)
