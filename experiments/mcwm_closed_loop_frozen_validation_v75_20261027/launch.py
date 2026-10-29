"""Three paid initializers, one reused native pair, then ten fixed full days and stop."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.launch import preflight, watch

V60 = HERE.parent/'mcwm_closed_loop_control_campaign_v60_20261025'
TERMINAL = {'done', 'failed', 'cancelled'}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cells(protocol):
    rows = [dict(seed=seed, scenario=scene, arm=arm,
        relative=f'runs/executed_cql/validation/seed{seed}/{scene}/{arm}')
        for seed in protocol['development_seeds'] for scene in protocol['scenarios']
        for arm in ('fresh', 'no_fresh')]
    if len(rows) != 12 or len({row['relative'] for row in rows}) != 12:
        raise ValueError('The fixed new study is three seeds by two scenarios by two arms')
    return rows


def native_jobs(expected, infra):
    return [infra.task(cell['relative'], HERE/'control.py',
        ['--seed', cell['seed'], '--scenario', cell['scenario'], '--arm', cell['arm'],
         '--out', infra.remote(HERE/cell['relative'])]) for cell in expected]


def receipt(infra, batch, expected):
    path = HERE/(batch+'_submission.json')
    return json.loads(path.read_text()) if path.exists() else infra.submit(native_jobs(expected, infra), batch)


def initializer_gate(value, seed):
    return bool(value.get('status') == 'complete' and value.get('all_pass') is True
        and value.get('seed') == seed and value.get('offline_updates') == 50000
        and value.get('online_updates') == 0 and value.get('recipe_exact') is True
        and value.get('initializer_path'))


def initialization(infra, protocol, read_only):
    # GPU workers do not share the HPC filesystem. Publish their immutable
    # checkpoint first and its completed gate last, before any native load.
    read_only(infra.promote_initializers, protocol['development_seeds'])
    paths = [f'inputs/seed{seed}/'+name for seed in protocol['development_seeds']
             for name in ('initializer_gate.json', 'failure.json')]
    found = read_only(infra.presence, paths)
    gates, failures = {}, []
    for seed in protocol['development_seeds']:
        base = f'inputs/seed{seed}/'
        if found.get(base+'failure.json'):
            failures.append(dict(seed=seed, failure=read_only(infra.collect, base+'failure.json')))
        elif found.get(base+'initializer_gate.json'):
            gate = read_only(infra.collect, base+'initializer_gate.json')
            gates[str(seed)] = gate
            if not initializer_gate(gate, seed):
                failures.append(dict(seed=seed, gate=gate))
    ready = len(gates) == 3 and not failures
    pending, dispatch = [], None
    submission = HERE/'initializers_submission.json'
    if not ready and not failures and submission.exists():
        receipt = json.loads(submission.read_text())
        ids = [row['id'] for row in receipt['submitted']]
        statuses = read_only(infra.status, ids)
        if {row['id'] for row in statuses} != set(ids):
            raise ValueError('Initializer receipt task was omitted; no duplicate allowed')
        if all(row['status'] in TERMINAL for row in statuses):
            # A successful process must have its complete gate. Keep actual failed
            # work visible rather than silently allocating a replacement initializer.
            failures.append(dict(reason='Terminal initializer batch without three complete gates', tasks=statuses))
        else:
            failures.extend(row for row in statuses if row['status'] in ('failed', 'cancelled'))
            pending = [row for row in statuses if row['status'] not in TERMINAL]
            queued = [row for row in pending if row['status'] == 'queued']
            if queued and not failures:
                # A third full-cache initializer waits for one of the two GPUs.
                # This mutates only dispatch state for existing receipt IDs and
                # is deliberately outside the read-only retry helper.
                dispatch = infra.dispatch_pending_initializers(queued)
    return dict(ready=ready, gates=gates, failures=failures, pending=pending,
        pending_initializer_dispatch=dispatch)


def summary_task(infra, batch, outputs, read_only):
    path = HERE/(batch+'_submission.json')
    if not path.exists():
        infra.submit([infra.task('analysis/'+batch, HERE/'analyze.py',
            ['--out', infra.remote(HERE/outputs[0])])], batch)
    identifier = json.loads(path.read_text())['submitted'][0]['id']
    statuses = read_only(infra.status, [identifier])
    if len(statuses) != 1 or statuses[0]['id'] != identifier:
        raise ValueError('Summary receipt omitted; no duplicate allowed')
    status = statuses[0]
    found = read_only(infra.presence, outputs)
    missing = [name for name in outputs if not found.get(name)]
    return dict(id=identifier, status=status, missing=missing,
        failed=status['status'] in ('failed', 'cancelled') or bool(missing and status['status'] in TERMINAL),
        ready=not missing and status['status'] not in ('failed', 'cancelled'))


def _advance(read_only):
    ready = HERE/'execution_ready.json'
    if not ready.exists() or json.loads(ready.read_text()).get('ready') is not True:
        return dict(stage='preparing_frozen_v73_validation', done=False, goal_achieved=False,
            completed=0, pending=[], failures=[], all_terminal=False)
    infra = module('v75_launch_infra', HERE/'infra.py')
    protocol = json.loads((HERE/'protocol.json').read_text())
    initializers = initialization(infra, protocol, read_only)
    if initializers['failures']:
        return dict(stage='initializer_attention', done=True, goal_achieved=False,
            initialization=initializers, requires_agent_review=True, performance_verdict='not_evaluated_incomplete')
    if not initializers['ready']:
        return dict(stage='waiting_original_recipe_initializers', done=False,
            goal_achieved=False, initialization=initializers, native_days_submitted=0)
    expected = cells(protocol)
    declared = protocol['native_preflight']
    first = [cell for cell in expected if cell['seed'] == declared['seed'] and cell['scenario'] == declared['scenario']]
    if len(first) != 2 or declared.get('reuse_full_day') is not True or first[0]['arm'] != 'fresh':
        raise ValueError('Preflight must reuse both first-condition full days')
    batch = 'frozen_v73_validation'
    first_receipt = receipt(infra, batch+'_preflight', first)
    paths = [cell['relative']+'/preflight.json' for cell in first]
    found = read_only(infra.presence, paths)
    if not all(found.get(path) for path in paths):
        current = watch(first_receipt, first, infra, read_only)
        stopped = current['all_terminal'] or bool(current['failures'])
        return dict(current, stage='validation_preflight_attention' if stopped else 'validation_native_preflight',
            done=stopped, goal_achieved=False, requires_agent_review=stopped,
            native_tasks_may_still_run=bool(current['pending']), missing_preflight=[p for p in paths if not found.get(p)])
    rows = [read_only(infra.collect, path) for path in paths]
    gate = preflight(rows, protocol)
    if gate['all_pass']:
        prefix = read_only(infra.common_prefix, rows)
        gate['common_prefix'] = prefix
        if prefix['all_pass'] is not True:
            gate['all_pass'] = False
            gate['failed_checks'].append('full_preupdate_prefix_exact')
    first_state = watch(first_receipt, first, infra, read_only)
    if not gate['all_pass'] or first_state['failures']:
        return dict(first_state, stage='validation_preflight_attention', done=True, goal_achieved=False,
            preflight=gate, requires_agent_review=True, native_tasks_may_still_run=bool(first_state['pending']),
            performance_verdict='not_evaluated_incomplete')
    remaining = [cell for cell in expected if cell not in first]
    if len(remaining) != 10:
        raise ValueError('Only ten remaining full days are authorized')
    later = receipt(infra, batch+'_remaining', remaining)
    combined = dict(submitted=first_receipt['submitted']+later['submitted'])
    current = watch(combined, expected, infra, read_only)
    if current['failures']:
        return dict(current, stage='validation_native_attention', done=True, goal_achieved=False,
            preflight=gate, requires_agent_review=True, native_tasks_may_still_run=bool(current['pending']),
            performance_verdict='not_evaluated_incomplete')
    if not current['all_terminal']:
        return dict(current, stage='frozen_v73_validation_native', done=False, goal_achieved=False,
            preflight=gate, initialization_completed=3)
    outputs = protocol['summary_outputs']
    summary = summary_task(infra, batch+'_analysis', outputs, read_only)
    if summary['failed']:
        return dict(current, stage='validation_summary_attention', done=True, goal_achieved=False,
            summary=summary, requires_agent_review=True, performance_verdict='not_evaluated_incomplete')
    if not summary['ready']:
        return dict(current, stage='frozen_v73_validation_summary', done=False, goal_achieved=False,
            preflight=gate, missing_summary=summary['missing'])
    result, *_ = [read_only(infra.collect, path) for path in outputs]
    if result.get('status') != 'complete' or result.get('integrity', {}).get('all_pass') is not True:
        return dict(current, stage='validation_technical_attention', done=True, goal_achieved=False,
            failed_checks=result.get('integrity', {}).get('failed_checks', []), summary_outputs=outputs,
            requires_agent_review=True, performance_verdict='not_evaluated_incomplete')
    decision = result.get('decision', {})
    passed = decision.get('scientific_gate_pass')
    if not isinstance(passed, bool):
        raise ValueError('Validation result has no boolean primary cost verdict')
    return dict(current, stage='fixed_validation_complete', done=True, goal_achieved=False,
        candidate_passed=passed, decision=decision, preflight=gate, summary_outputs=outputs,
        requires_agent_review=True, old_v73_gate_reopened=False)


def advance():
    helper = module('v75_v60_readonly', V60/'supervisor.py')
    try:
        return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage='waiting_transient_validation_read', done=False, goal_achieved=False, message=str(error))
