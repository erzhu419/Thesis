"""One reused native pair, six subsequent fixed days, one resident readout."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.launch import (
    module, preflight, watch)


def cells(protocol):
    rows = [dict(seed=seed, scenario=scene, arm=arm,
        relative=f'runs/executed_cql/development/seed{seed}/{scene}/{arm}')
        for seed in protocol['development_seeds'] for scene in protocol['scenarios']
        for arm in protocol['arms']]
    if len(rows) != 8 or len({r['relative'] for r in rows}) != 8:
        raise ValueError('Only the fixed two-seed, two-scenario, two-arm cohort is authorized')
    return rows


def receipt(infra, batch, expected):
    path = HERE/(batch+'_submission.json')
    if path.exists():
        return json.loads(path.read_text())
    jobs = [infra.task(cell['relative'], HERE/'control.py',
        ['--seed', cell['seed'], '--scenario', cell['scenario'], '--arm', cell['arm'],
         '--out', infra.remote(HERE/cell['relative'])]) for cell in expected]
    return infra.submit(jobs, batch)


def _advance(read_only):
    ready = HERE/'execution_ready.json'
    if not ready.exists() or json.loads(ready.read_text()).get('ready') is not True:
        return dict(stage='preparing_v83', done=False)
    infra = module('v83_local_infra', HERE/'infra.py')
    p = json.loads((HERE/'protocol.json').read_text())
    expected = cells(p)
    first = [cell for cell in expected if cell['seed'] == p['native_preflight']['seed']
             and cell['scenario'] == p['native_preflight']['scenario']]
    first_receipt = receipt(infra, 'context_preflight', first)
    paths = [cell['relative']+'/preflight.json' for cell in first]
    found = read_only(infra.presence, paths)
    current = watch(first_receipt, first, infra, read_only)
    if current['failures']:
        return dict(current, stage='preflight_attention', done=True, requires_agent_review=True)
    if not all(found.values()):
        stopped = current['all_terminal']
        return dict(current, stage='preflight_attention' if stopped else 'first_pair_native_preflight',
            done=stopped, missing_preflight=[path for path in paths if not found[path]])
    rows = [read_only(infra.collect, path) for path in paths]
    gate = preflight(rows, p)
    gate['context_first_packet'] = all(row.get('context_first_packet', {}).get('all_pass') is True
                                       for row in rows)
    if not gate['context_first_packet']:
        gate['all_pass'] = False
        gate['failed_checks'].append('context_first_packet')
    gate['singleton_td_first_packet'] = all(row.get('td_gradient_gate', {}).get('all_pass') is True
                                           for row in rows)
    if not gate['singleton_td_first_packet']:
        gate['all_pass'] = False
        gate['failed_checks'].append('singleton_td_first_packet')
    if gate['all_pass']:
        gate['common_prefix'] = read_only(infra.common_prefix, rows)
        if not gate['common_prefix']['all_pass']:
            gate['all_pass'] = False
            gate['failed_checks'].append('full_core_and_context_prefix_exact')
    if not gate['all_pass']:
        return dict(current, stage='preflight_attention', done=True, preflight=gate,
            requires_agent_review=True)
    remaining = [cell for cell in expected if cell not in first]
    later = receipt(infra, 'context_remaining', remaining)
    combined = dict(submitted=first_receipt['submitted']+later['submitted'])
    current = watch(combined, expected, infra, read_only)
    if current['failures']:
        return dict(current, stage='native_attention', done=True, preflight=gate,
            requires_agent_review=True)
    if not current['all_terminal']:
        return dict(current, stage='eight_fixed_native_days', done=False, preflight=gate)
    output = p['summary_outputs']
    path = HERE/'context_analysis_submission.json'
    if not path.exists():
        infra.submit([infra.task('analysis/context', HERE/'analyze.py',
            ['--out', infra.remote(HERE/output[0])])], 'context_analysis')
    identifier = json.loads(path.read_text())['submitted'][0]['id']
    status = read_only(infra.status, [identifier])
    found = read_only(infra.presence, output)
    if not all(found.values()):
        stopped = status[0]['status'] in ('failed','cancelled','done')
        return dict(current, stage='summary_attention' if stopped else 'resident_summary',
            done=stopped, preflight=gate, summary_task=status,
            missing=[path for path in output if not found[path]])
    results = [read_only(infra.collect, path) for path in output]
    main = results[0]
    return dict(current, stage='v83_complete', done=True, preflight=gate,
        integrity=main.get('integrity'), decision=main.get('decision'),
        summary_outputs=output, old_gate_changed=False, requires_agent_review=True)


def advance():
    helper = module('v83_resident_read', HERE.parent/'mcwm_closed_loop_control_campaign_v60_20261025/supervisor.py')
    try:
        return _advance(helper.read_only)
    except helper.ReadUnavailable as error:
        return dict(stage='waiting_transient_v83_read', done=False, message=str(error))
