"""Small output: design certificates and old-log stress are separate gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_trace_design_v40_20261024.budget_plan import build_plan

COLUMNS = ('slot', 'arm', 'events', 'first_reward_index0', 'reward_within_fixed45',
    'reward_mathematically_supported', 'reward_weight', 'reward_log10_weight',
    'ideal_same_input_same_snapshot_reward_weight', 'old_dirac_reward_weight')


def analyze(runs, out):
    cases = {}; plan = build_plan()
    for scenario in ('stable', 'local_speed_drop'):
        root = runs/scenario
        if not (root/'summary.json').exists():
            failure = json.loads((root/'failure.json').read_text()) if (root/'failure.json').exists() else None
            cases[scenario] = dict(status='missing_or_failed', failure=failure)
            continue
        source = json.loads((root/'summary.json').read_text())
        models = {}
        for model, rows in source['models'].items():
            numeric = [r['reward_weight'] for r in rows if r['reward_weight'] is not None]
            logs = [r['reward_log10_weight'] for r in rows if r['reward_log10_weight'] is not None]
            aggregate = dict(chains=len(rows), fixed45_physical_pass=sum(r['physical']['all_pass'] for r in rows),
                reward_in45=sum(r['reward_within_fixed45'] for r in rows),
                mathematical_reward_reach=sum(r['reward_mathematically_supported'] for r in rows),
                numerical_reward_reach=sum(w > 0. for w in numeric),
                minimum_weight=min(numeric, default=None), maximum_weight=max(numeric, default=None),
                minimum_log10_weight=min(logs, default=None), maximum_log10_weight=max(logs, default=None),
                global_map_certificates=sum(r['global_map_support_certificates'] for r in rows),
                different_execution_maps=sum(r['different_execution_maps'] for r in rows),
                successor_links=sum(r['events']-1 for r in rows),
                support_failures=sum(r['point_support_failures'] for r in rows),
                float32_link_zeros=sum(r['float32_link_zeros'] for r in rows),
                atom_mass_underflow=sum(r['atom_mass_underflow'] for r in rows),
                zero_dt=sum(r['physical']['zero_duration_events'] for r in rows))
            route = {}
            for line in ('7X', '102X', '705X'):
                group = [r for r in rows if r['slot'].split('/')[0] == line]
                route[line] = dict(chains=len(group),
                    math_reach=sum(r['reward_mathematically_supported'] for r in group),
                    numeric_reach=sum(r['reward_weight'] is not None and r['reward_weight'] > 0 for r in group),
                    min_weight=min((r['reward_weight'] for r in group if r['reward_weight'] is not None), default=None))
            models[model] = dict(aggregate=aggregate, routes=route,
                rows=[[r[k] for k in COLUMNS] for r in rows],
                physical_failures=[dict(slot=r['slot'], arm=r['arm'], reasons=r['physical']['reasons'])
                    for r in rows if not r['physical']['all_pass']])
        cases[scenario] = dict(status=source['status'], integrity=source['integrity'],
            identities=source['identities'], models=models, elapsed_seconds=source['elapsed_seconds'],
            final_model_checkpoints_loaded=source['final_model_checkpoints_loaded'])
    complete = all(c.get('integrity', {}).get('all_pass') for c in cases.values())
    models = [m['aggregate'] for c in cases.values() for m in c.get('models', {}).values()]
    supported = complete and all(m['mathematical_reward_reach'] == 12 for m in models)
    numerical = complete and all(m['numerical_reward_reach'] == 12 for m in models)
    result = dict(status='complete' if complete else 'partial', priority='high',
        integrity=dict(all_pass=complete), schema=dict(columns=COLUMNS,
            reward_weight='OLD trajectory coefficient under a proposed new measure; no new policy rollout or training'),
        prospective_budget=dict(all_pass=plan['all_pass'], financing=plan['financing'],
            pairs=plan['pairs'], feedback_packets=57, planned_updates=456,
            ideal_same_input_same_snapshot_c=1,
            ideal_same_input_same_snapshot_last45_reward_weight=.9**44,
            physical_availability_is_not_admission=True),
        cases=cases,
        gates=dict(design_and_old_evidence_integrity=complete,
            exact285_slot_calendar=plan['all_pass'],
            proposed_measure_reaches_old_reward_witnesses=supported,
            numeric_old_reward_reach=numerical,
            actual_new_stochastic_prefix_validated=False,
            actual_new_budgeted_response_chains_validated=False),
        decision=dict(design_precheck_pass=complete and supported and numerical,
            collection_launched=False, training_ready=False, training_launched=False,
            next_gate='Review fixed-budget age/line-window compromise and old-log RN attenuation before authorizing any new common-prefix collection; require actual causal behavior logging, physical actions, chain admission and zero suffix training before any learning.'),
        compute_budget=dict(new_native_prefixes=0,new_native_branches=0,new_native_seconds=0,
            new_training_labels=0,new_optimizer_updates=0,heldout_feedback_admitted=0,
            checkpoints_downloaded=0),
        limitations=['old deterministic logs retain their original generating behavior',
            'new stochastic-policy reward timing and reach remain unobserved',
            'one staged issue window per line confounds line and window; no stable-control benefit measured',
            'queue age up to6000s; only already admitted contiguous prefixes may enter any future trace',
            'same-snapshot unit ratios do not certify ratios after future updates',
            'mixed V/SAC candidate is not a proved pure-policy Retrace operator'])
    encoded = json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > 24000: raise RuntimeError('Summary exceeds small-result limit')
    out.mkdir(parents=True, exist_ok=False)
    (out/'summary.json').write_text(encoded)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    try:
        value=analyze(args.runs,args.out)
        print(json.dumps(dict(status=value['status'],decision=value['decision'])))
    except BaseException as error:
        if not isinstance(error,SystemExit):
            args.out.mkdir(parents=True,exist_ok=True)
            with (args.out/'failure.json').open('x') as handle:
                json.dump(dict(type=type(error).__name__,message=str(error)),handle)
        raise
