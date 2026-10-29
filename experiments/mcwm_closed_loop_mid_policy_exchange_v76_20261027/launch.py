"""Submit both fixed seed tasks together, collect two bounded readouts, then stop."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.launch import preflight

V60 = HERE.parent/'mcwm_closed_loop_control_campaign_v60_20261025'
TERMINAL = {'done', 'failed', 'cancelled'}
OUTPUTS = ('results/diagnostic.json', 'results/evidence.json')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def receipt(infra, batch, *, native=False, protocol=None):
    path = HERE/(batch+'_submission.json')
    if path.exists(): return json.loads(path.read_text())
    if native:
        jobs = [infra.task(f'seed{seed}', HERE/'pair.py', ['--seed', seed,
            '--out', infra.remote(HERE/'runs'/f'seed{seed}')]) for seed in protocol['seeds']]
    else:
        jobs = [infra.task('mid_policy_exchange_analysis', HERE/'analyze.py',
            ['--out', infra.remote(HERE/OUTPUTS[0])])]
    return infra.submit(jobs, batch)


def statuses(infra, value, read_only):
    ids = [row['id'] for row in value['submitted']]
    rows = read_only(infra.status, ids)
    if len(rows) != len(ids) or {row['id'] for row in rows} != set(ids):
        raise ValueError('Scheduler omitted an existing task ID; no replacement is permitted')
    return rows


def watch(infra, value, protocol, read_only):
    jobs = statuses(infra, value, read_only)
    index = {row['signature'].rsplit('/', 1)[-1]: row['id'] for row in value['submitted']}
    if set(index) != {f'seed{seed}' for seed in protocol['seeds']}:
        raise ValueError('Native receipt differs from the two fixed seed tasks')
    by_id = {row['id']: row for row in jobs}
    paths = [f'runs/seed{seed}/seed_summary.json' for seed in protocol['seeds']]
    found = read_only(infra.presence, paths)
    completed, failures = [], []
    for seed, path in zip(protocol['seeds'], paths):
        job = by_id[index[f'seed{seed}']]
        if job['status'] in ('failed', 'cancelled'):
            failures.append(dict(seed=seed, task=job, reason='native_seed_task_failed'))
        elif found.get(path):
            summary = read_only(infra.collect, path)
            audit = summary.get('integrity', {})
            valid = summary.get('status') == 'complete' and audit.get('all_pass') is True \
                and bool(audit.get('checks')) and all(value is True for value in audit['checks'].values())
            if not valid:
                failures.append(dict(seed=seed, reason='native_seed_integrity_failed', path=path))
            elif job['status'] == 'done':
                completed.append(seed)
        elif job['status'] in TERMINAL:
            failures.append(dict(seed=seed, task=job, reason='terminal_seed_summary_missing', path=path))
    return dict(completed=len(completed), completed_seeds=completed, failures=failures,
        pending=[row['id'] for row in jobs if row['status'] not in TERMINAL],
        all_terminal=all(row['status'] in TERMINAL for row in jobs), native_jobs=jobs)


def attention(current, stage, **fields):
    return dict(current, stage=stage, done=True, goal_achieved=False,
        requires_agent_review=True, native_tasks_may_still_run=bool(current.get('pending')), **fields)


def _advance(read_only):
    ready = HERE/'execution_ready.json'
    if not ready.exists() or json.loads(ready.read_text()).get('ready') is not True:
        return dict(stage='preparing_mid_policy_exchange', done=False, goal_achieved=False)
    infra = module('v76_launch_infra', HERE/'infra.py')
    protocol = json.loads((HERE/'protocol.json').read_text())
    if protocol['seeds'] != [6208, 6207] or protocol['native_preflight']['seed'] != 6208 \
            or protocol['native_preflight'].get('reuse_prefix') is not True:
        raise ValueError('v76 fixes6208 bad case and6207 positive case; no adaptive case selection')
    native = receipt(infra, 'mid_policy_exchange_native', native=True, protocol=protocol)
    current = watch(infra, native, protocol, read_only)
    if current['failures']:
        return attention(current, 'mid_policy_exchange_native_attention')
    # Both seed jobs are already submitted. This technical preflight never
    # chooses whether the positive case or any extra seed should be run.
    paths = [f'runs/seed6208/{arm}/preflight.json' for arm in ('fresh', 'no_fresh')]
    found = read_only(infra.presence, paths)
    missing = [path for path in paths if not found.get(path)]
    if missing:
        if current['all_terminal']:
            return attention(current, 'mid_policy_exchange_preflight_attention', missing_preflight=missing)
        return dict(current, stage='mid_policy_exchange_native_preflight', done=False,
            goal_achieved=False, missing_preflight=missing)
    gate = preflight([read_only(infra.collect, path) for path in paths], protocol)
    if gate['all_pass']:
        gate['common_prefix'] = read_only(infra.common_prefix, [read_only(infra.collect, path) for path in paths])
        if not gate['common_prefix']['all_pass']:
            gate['all_pass'] = False
            gate['failed_checks'].append('first_packet_common_prefix')
    if not gate['all_pass']:
        return attention(current, 'mid_policy_exchange_preflight_attention', preflight=gate)
    if not current['all_terminal']:
        return dict(current, stage='mid_policy_exchange_native', done=False,
            goal_achieved=False, preflight=gate)
    summary = receipt(infra, 'mid_policy_exchange_analysis')
    state = statuses(infra, summary, read_only)[0]
    found = read_only(infra.presence, OUTPUTS)
    missing = [path for path in OUTPUTS if not found.get(path)]
    if state['status'] in ('failed', 'cancelled') or missing and state['status'] in TERMINAL:
        return attention(current, 'mid_policy_exchange_summary_attention', preflight=gate,
            attention=state, missing_summary=missing)
    if missing or state['status'] != 'done':
        return dict(current, stage='mid_policy_exchange_summary', done=False,
            goal_achieved=False, preflight=gate, missing_summary=missing)
    result, detail = [read_only(infra.collect, path) for path in OUTPUTS]
    if result.get('status') != 'complete' or result.get('integrity', {}).get('all_pass') is not True:
        return attention(current, 'mid_policy_exchange_technical_attention', preflight=gate,
            failed_checks=result.get('integrity', {}).get('failed_checks', []),
            missing=result.get('missing', []), result_failures=result.get('failures', []))
    choice = result.get('decision', {})
    if not isinstance(choice.get('baseline_protection_may_be_considered'), bool):
        raise ValueError('Completed diagnosis has no boolean bounded follow-up decision')
    return attention(current, 'mid_policy_exchange_complete_requires_root_review',
        preflight=gate, result=OUTPUTS[0], evidence=OUTPUTS[1], decision=choice,
        evidence_status=detail.get('status'), new_recipe_started=False)


def advance():
    helper = module('v76_read_only', V60/'supervisor.py')
    try:
        return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage='waiting_transient_mid_exchange_read', done=False,
            goal_achieved=False, message=str(error))
