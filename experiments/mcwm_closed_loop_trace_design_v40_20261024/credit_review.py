"""Fixed-weight reward visibility and the final, non-stale release proposal."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_trace_support_v39_20261024.trace import construct_trace
from experiments.mcwm_closed_loop_trace_design_v40_20261024.release_plan import (
    build_release_plan, compact_release_plan,
)


def read(path): return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as handle: return [json.loads(row) for row in handle if row.strip()]


def reward_visibility(native, baseline, c, witness, own_reward, other_reward, bootstrap):
    """Change one observed reward component, retaining every Q/ratio/endpoint.

    This algebra diagnostic creates no transition or training label.  The
    opposite reward is the already observed semantically matched native event,
    not a fixed300 aggregate or a newly simulated counterfactual.
    """
    if np.float32(np.float32(own_reward)+np.float32(bootstrap)) != np.float32(native[witness]):
        raise RuntimeError('Saved reward+bootstrap does not reproduce its native target')
    base = construct_trace(native, baseline, c, [True]*len(native), 1.)
    changed = list(native)
    changed[witness] = float(np.float32(np.float32(other_reward)+np.float32(bootstrap)))
    alt = construct_trace(changed, baseline, c, [True]*len(native), 1.)
    root = np.float32(base['targets'][0]); changed_root = np.float32(alt['targets'][0])
    coefficient = base['reach_weights'][witness]
    return dict(root=float(root), changed_root=float(changed_root),
        fixed_weight_root_changed=bool(root.view(np.uint32) != changed_root.view(np.uint32)),
        root_delta=float(changed_root)-float(root),
        own_reward_scaled=own_reward, opposite_reward_scaled=other_reward,
        native_witness_target_changed=changed[witness] != native[witness],
        coefficient=coefficient,
        ideal_linear_reward_delta=coefficient*(other_reward-own_reward),
        root_float32_spacing=float(abs(np.spacing(root))),
        new_training_labels=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    p = read(HERE/'protocol.json'); old = HERE.parent/p['source_v39_directory']
    recovered = HERE.parent/p['source_v39r1_directory']
    source = read(HERE/'analysis/summary.json')
    plan = build_release_plan(); cases = {}; all_checks = []
    for scenario, case in source['cases'].items():
        models = {}
        for model, result in case['models'].items():
            indexed = {(r[0], r[1]): r for r in result['rows']}
            cache = {}
            for slot, arm in indexed:
                name = slot.replace('/', '_')+'_'+arm+'.jsonl'
                path = recovered/'runs'/scenario/model/name
                if not path.exists(): path = old/'runs'/scenario/model/name
                cache[slot, arm] = lines(path)[:45]
            rows = []
            for (slot, arm), original in indexed.items():
                current = cache[slot, arm]; k = original[3]
                opposite = 'high' if arm == 'low' else 'low'
                opposite_k = indexed[slot, opposite][3]
                native = [r['probe']['native_target'] for r in current]
                baseline = [r['probe']['recorded_next_target_q'] for r in current[:-1]]
                ratio_path = HERE/'runs'/scenario/model/(slot.replace('/', '_')+'_'+arm+'.jsonl')
                c = [r['ratio']['c'] for r in lines(ratio_path)]
                parts = current[k]['probe']['parts']
                contrast = cache[slot, opposite][opposite_k]['probe']['parts']['reward_scaled']
                value = reward_visibility(native, baseline, c, k, parts['reward_scaled'],
                    contrast, parts['bootstrap_contribution'])
                all_checks.append(value['coefficient'] == original[6])
                rows.append([slot, arm, k, value['coefficient'], value['ideal_linear_reward_delta'],
                    value['root_float32_spacing'], value['native_witness_target_changed'],
                    value['fixed_weight_root_changed'], value['root_delta']])
            models[model] = dict(chains=len(rows),
                native_witness_changes=sum(r[6] for r in rows),
                float32_root_changes=sum(r[7] for r in rows),
                minimum_reward_weight=result['aggregate']['minimum_weight'],
                maximum_reward_weight=result['aggregate']['maximum_weight'], rows=rows)
        cases[scenario] = models
    output = dict(status='complete', priority='high',
        integrity=dict(all_pass=all(all_checks) and source['integrity']['all_pass'] and plan['all_pass'],
            prior_trace_weights_exact=all(all_checks), final_budget_arithmetic=plan['all_pass']),
        columns=['slot','arm','first_reward_index0','coefficient','linear_scaled_reward_delta',
            'root_float32_spacing','native_witness_target_changed','fixed_weight_root_changed','root_delta'],
        cases=cases, final_budget=dict(
            slots=285, ordinary_slots=15, response_slots=270, pairs=3, events_per_arm=45,
            common_issue_window=[9000,9300], response_native_release=9630,
            maximum_label_age_seconds=630, warmup_updates=24, response_updates=432,
            response_feedback_all_admitted_before_response_updates=True,
            target_horizon=24000, initializer=50000, F_unchanged=True, trust_unchanged=True,
            calendar_changed=True, future_one_step_control_must_share_new_calendar=True,
            financing=plan['financing']),
        decision=dict(structural_support_and_capacity_pass=source['decision']['design_precheck_pass'] and plan['all_pass'],
            positive_weight_is_not_a_usable_signal_certificate=True,
            collection_launched=False, training_ready=False, training_launched=False,
            next_gate='Resolve target/behavior drift and fixed-weight float32 credit attenuation before collecting; same-snapshot unit ratios are insufficient for future updated targets.'),
        compute_budget=dict(new_native_prefixes=0,new_native_branches=0,new_native_seconds=0,
            new_training_labels=0,new_optimizer_updates=0,heldout_feedback_admitted=0),
        limitations=['old deterministic trajectories are counterfactual support stress only',
            'single semantically matched reward perturbation holds Q and ratios fixed; it is not total300s credit or a policy effect',
            'final same-window release calendar is prospective and has not been natively collected',
            'old targets trained on the former dataset need not match targets under future new data'])
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in [('summary.json',output),('release_plan.json',compact_release_plan(plan))]:
        text=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
        if len(text.encode()) > 24000: raise RuntimeError('Small-result budget exceeded')
        (args.out/name).write_text(text)
    print(json.dumps(dict(status=output['status'],integrity=output['integrity'],decision=output['decision'])))


if __name__ == '__main__': main()
