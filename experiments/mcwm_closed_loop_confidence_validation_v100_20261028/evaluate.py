"""Evaluate six frozen first-accept policies on the single reserved V seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_confidence_validation_v100_20261028.analysis import (
    COSTS, LINES, MODELS, SCENES, SEEDS, finite, policy_scores, scores_equal, tail_readout)

METHODS = ('reference','raw_history6','line3','service5','state_independent','calibrated_candidate')
RAW_MODELS = dict(raw_history6='service_history6',line3='line3',service5='service5')
PASSENGERS = 15079


def reference_tail(pair):
    endpoint, boundary = pair['reference_full_endpoint'], pair['tail_boundary']['reference']
    wait = endpoint['wait']-boundary['accrued_wait_seconds']
    ride = endpoint['ride']-boundary['accrued_in_vehicle_seconds']
    return dict(wait=wait,ride=ride,cost=2*wait+ride)


def validate(summary, protocol, fit_model, calibration):
    pairs = summary['pairs']
    expected = {(SEEDS[0], scene, line) for scene in SCENES for line in LINES}
    keys = [(row['external_seed'],row['scenario'],row['line_id']) for row in pairs]
    bias_binding = dict(source_fit=protocol['source_fit'],primary_model=fit_model['primary_model'],
        bias_model=fit_model['bias_model'],source_summary=fit_model['source_summary'],source_protocol=fit_model['source_protocol'])
    calibration_binding = dict(source_calibration=protocol['source_calibration'],source_fit=protocol['source_fit'],
        alpha=calibration['alpha'],C=calibration['C'],state_independent_C=calibration['state_independent']['C'],
        ready_for_validation=calibration['ready_for_validation'])
    checks = dict(
        collection_complete=summary['status'] == 'complete' and summary['collection_gate']['passed'],
        collection_integrity=summary['integrity']['all_pass'] and bool(summary['integrity']['checks'])
            and all(summary['integrity']['checks'].values()),
        six_declared_pairs=len(pairs) == 6 and len(set(keys)) == 6 and set(keys) == expected,
        six_distinct_roots=len({tuple(row['physical_root_id']) for row in pairs}) == 6,
        twelve_labels=summary['budget']['labels_observed'] == 12,
        all_pairs_valid=bool(pairs) and all(row['valid'] and bool(row['checks'])
            and all(value for key,value in row['checks'].items() if key != 'physical_separation') for row in pairs),
        finite_truth=bool(pairs) and all(row['label_delta_per_passenger'] is not None
            and all(finite(row['label_delta_per_passenger'].get(key)) for key in COSTS) for row in pairs),
        finite_reference_endpoints=bool(pairs) and all(row.get('reference_full_endpoint') is not None
            and all(finite(row['reference_full_endpoint'].get(key)) for key in COSTS) for row in pairs),
        frozen_fit_binding=summary.get('frozen_bias') == bias_binding,
        frozen_calibration_binding=summary.get('frozen_calibration') == calibration_binding,
        calibration_ready=calibration['status'] == 'complete' and calibration['ready_for_validation'] is True
            and calibration['integrity']['all_pass'] and bool(calibration['integrity']['checks'])
            and all(calibration['integrity']['checks'].values()),
        calibration_model_unchanged=calibration['frozen_model'] == fit_model
            and calibration['source_identity']['source_fit'] == protocol['source_fit'],
        finite_frozen_margins=all(finite(value) and value >= 0.
            for value in (calibration['C'],calibration['state_independent']['C'])),
        fixed_protocol=protocol['external_seeds'] == list(SEEDS) and protocol['scenarios'] == list(SCENES)
            and protocol['validation']['comparators'] == list(METHODS)
            and calibration['alpha'] == protocol['calibration']['alpha'] == .2,
        no_V_fitting=summary['budget']['new_fits'] == 0 and summary['interpretation']['model_fits'] == 0)
    issue_scores_valid = bool(pairs) and checks['finite_frozen_margins']
    tail_complete, reference_checks = bool(pairs), {}
    for row in pairs:
        values_valid = finite(row['condition'].get('value')) and all(finite(row['predictions'].get(name)) for name in MODELS)
        issue_scores_valid = issue_scores_valid and values_valid and scores_equal(row.get('policy_scores'),
            policy_scores(row['predictions']['service_history6'],row['condition']['value'],bias_binding,calibration_binding))
        tail = tail_readout(row, row['observed_delta'])
        tail_complete = tail_complete and tail['observed'] and row.get('tail') == tail
        endpoint = row.get('reference_full_endpoint')
        if tail['observed'] and endpoint is not None and all(finite(endpoint.get(key)) for key in COSTS):
            for role in ('reference','candidate'):
                bound = row['tail_boundary'][role]
                for metric,field in (('wait','accrued_wait_seconds'),('ride','accrued_in_vehicle_seconds')):
                    end = endpoint[metric]+(row['observed_delta'][metric] if role == 'candidate' else 0.)
                    tail_complete = tail_complete and finite(end) and end >= bound[field]
        elif endpoint is None:
            tail_complete = False
    for scene in SCENES:
        group = [row for row in pairs if row['scenario'] == scene]
        full = len(group) == 3 and all(row['reference_full_endpoint'] == group[0]['reference_full_endpoint'] for row in group)
        tail = len(group) == 3 and all(row.get('tail_boundary') is not None
            and row['tail_boundary'].get('reference') is not None for row in group)
        tail = tail and all(row['tail_boundary']['reference'] == group[0]['tail_boundary']['reference'] for row in group)
        reference_checks[scene] = dict(same_reference_full_day=full,same_reference_tail_boundary=tail)
    checks.update(issue_only_policy_scores=issue_scores_valid,all_exact_tail_boundaries=tail_complete,
        same_reference_ledger_across_roots=all(all(value.values()) for value in reference_checks.values()))
    return dict(all_pass=all(checks.values()),checks=checks,reference_checks=reference_checks)


def choose(rows, method):
    if method == 'reference':
        return None
    for row in sorted(rows,key=lambda value:(value['root']['started_at'],value['root']['sequence'])):
        accepted = (row['predictions'][RAW_MODELS[method]] < 0. if method in RAW_MODELS
                    else row['policy_scores'][method]['accept'])
        if accepted:
            return row
    return None


def costs(values):
    return dict(wait=values['wait'],weighted_wait=2*values['wait'],ride=values['ride'],cost=values['cost'])


def mean_costs(values):
    values = list(values)
    return {key:sum(row[key] for row in values)/len(values) for key in ('wait','weighted_wait','ride','cost')}


def evaluate(summary, protocol, fit_model, calibration):
    integrity = validate(summary,protocol,fit_model,calibration)
    result = dict(status='hold',integrity=integrity,passed=None,gates=None,
        by_scene={scene:dict(status='incomplete',methods=None) for scene in SCENES},aggregate=None,
        source_identity=dict(source_fit=protocol['source_fit'],source_calibration=protocol['source_calibration'],
            source_model=protocol['source_model']),
        frozen_policy=dict(bias_model=fit_model['bias_model'],C=calibration['C'],
            state_independent_C=calibration['state_independent']['C'],primary_model=fit_model['primary_model']),
        independent_unit=dict(external_seeds=1,seed=SEEDS[0],related_scenarios=2,paired_queries=6,physical_arms=12),
        decision='HOLD_REQUIRED_V_INPUTS',
        budget=dict(native_seconds=0,new_labels=0,new_fits=0,new_calibrations=0,optimizer_updates=0),
        interpretation=dict(cost_unit='seconds per passenger; J=2W+R',tail_window=[18300,24000],
            selection_uses_future_truth=False,one_selected_suffix_per_scene=True,
            policy_selection_after_V=False,further_collection=False,
            claim_scope='One reserved external seed with two related scenarios; not six independent validation units'))
    if not integrity['all_pass']:
        return result
    for scene in SCENES:
        rows = [row for row in summary['pairs'] if row['scenario'] == scene]
        reference = {key:value/PASSENGERS for key,value in rows[0]['reference_full_endpoint'].items()}
        reference_tail_cost = {key:value/PASSENGERS for key,value in reference_tail(rows[0]).items()}
        methods = {}
        for method in METHODS:
            selected = choose(rows,method)
            delta = selected['label_delta_per_passenger'] if selected else dict.fromkeys(COSTS,0.)
            tail_delta = selected['tail']['delta_per_passenger'] if selected else dict.fromkeys(COSTS,0.)
            methods[method] = dict(selected_root=selected['physical_root_id'] if selected else None,
                selected_line=selected['line_id'] if selected else None,
                interventions=int(selected is not None),
                physical_interventions=int(selected is not None and not selected['physical_noop']),
                physical_noop=selected['physical_noop'] if selected else None,
                full_day=costs({key:reference[key]+delta[key] for key in COSTS}),
                tail=costs({key:reference_tail_cost[key]+tail_delta[key] for key in COSTS}),
                delta_vs_reference=dict(full_day=costs(delta),tail=costs(tail_delta)))
        result['by_scene'][scene] = dict(status='complete',methods=methods)
    aggregate = {}
    for method in METHODS:
        records = [result['by_scene'][scene]['methods'][method] for scene in SCENES]
        worst_scene = max(SCENES,key=lambda scene:result['by_scene'][scene]['methods'][method]['delta_vs_reference']['full_day']['cost'])
        aggregate[method] = dict(mean_full_day=mean_costs(row['full_day'] for row in records),
            mean_tail=mean_costs(row['tail'] for row in records),
            mean_delta_vs_reference=mean_costs(row['delta_vs_reference']['full_day'] for row in records),
            mean_delta_vs_raw_history6=mean_costs({key:result['by_scene'][scene]['methods'][method]['delta_vs_reference']['full_day'][key]
                -result['by_scene'][scene]['methods']['raw_history6']['delta_vs_reference']['full_day'][key]
                for key in ('wait','weighted_wait','ride','cost')} for scene in SCENES),
            worst_scene=worst_scene,worst_scene_delta_vs_reference=result['by_scene'][worst_scene]['methods'][method]['delta_vs_reference']['full_day']['cost'],
            interventions=sum(row['interventions'] for row in records),
            physical_interventions=sum(row['physical_interventions'] for row in records))
    candidate = aggregate['calibrated_candidate']
    gates = dict(real_nonzero_intervention=candidate['physical_interventions'] > 0,
        mean_cost_better_than_reference=candidate['mean_delta_vs_reference']['cost'] < 0.,
        mean_cost_better_than_raw_history6=candidate['mean_delta_vs_raw_history6']['cost'] < 0.,
        no_scene_worse_than_reference=all(result['by_scene'][scene]['methods']['calibrated_candidate']['delta_vs_reference']['full_day']['cost'] <= 0. for scene in SCENES),
        stable_tail_nonpositive=result['by_scene']['stable']['methods']['calibrated_candidate']['delta_vs_reference']['tail']['cost'] <= 0.)
    result.update(status='complete',aggregate=aggregate,gates=gates,passed=all(gates.values()),
        decision='PASS_RESERVED_V_GATES' if all(gates.values()) else 'FAIL_RESERVED_V_GATES')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name,filename in (('summary','summary.json'),('protocol','protocol.json'),
                          ('fit-model','source_fit.json'),('calibration','source_calibration.json'),
                          ('out','evaluation.json')):
        parser.add_argument('--'+name,type=Path,default=HERE/filename)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('The reserved V evaluation already exists; do not retry or overwrite it')
    load = lambda path:json.loads(path.read_text())
    protocol = load(args.protocol)
    result = evaluate(load(args.summary),protocol,load(args.fit_model),load(args.calibration))
    raw = json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode()) > protocol['retrieval']['final_summary_bytes']:
        raise ValueError('Evaluation exceeds its declared small-summary byte limit')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=result['status'],passed=result['passed'],decision=result['decision'],bytes=len(raw.encode()))))
    return result


if __name__ == '__main__':
    main()
