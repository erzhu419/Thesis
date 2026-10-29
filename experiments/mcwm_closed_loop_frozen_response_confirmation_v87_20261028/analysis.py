"""Independent three-seed confirmation of the frozen v86 one-intervention rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_fresh_response_v86_20261027.analysis import (
    BUDGET_FIELDS, COST_KEYS, LINES, MODELS, SCENARIOS,
    _group, _reference_valid, _single_intervention, _training_checks, pair_readout,
)

SEEDS = (416207, 516207, 616207)
POLICY = 'first_supported_negative_once_then_reference'


def _freeze_checks(audit, model):
    expected = {key: model[key] for key in ('action_support', 'legacy_action_support', 'ridge',
        'denominator', 'hold_scale', 'feature_names')}
    expected.update(decision_threshold=0, policy=POLICY,
        coefficients={arm: {name: model['arms'][arm]['models'][name]['coef']
            for name in ('line', 'state')} for arm in ('stale', 'fresh')})
    return dict(resident_freeze_audit=audit['all_pass'] and all(audit['checks'].values()),
        original_model_contract=all(_training_checks(model).values()),
        exact_frozen_model_and_rule=audit['frozen'] == expected,
        fixed_confirmation_seeds=audit['confirmation_seeds'] == list(SEEDS),
        same_source_initializer=audit['source_initializer'] == 6207,
        same_reference_paths=audit['source_reference_paths'] == model['source_reference_paths'],
        four_original_fits_only=audit['original_training_fits'] == model['fits'] == 4,
        zero_new_fits=audit['new_fits'] == 0,
        zero_new_neural_updates=audit['new_neural_updates'] == 0,
        zero_new_truth_in_model=audit['heldout_truth_used'] == 0)


def _compact_policy(value):
    return dict(status=value['status'], opportunity_order=value['opportunity_order'],
        oracle_best_single_root_delta_cost=value['oracle_best_single_root_delta_cost'],
        models={name: {key: row[key] for key in ('selected_root', 'interventions',
            'selected_minus_reference', 'selected_minus_reference_per_passenger',
            'full_selected_endpoint', 'full_selected_cost_per_passenger',
            'single_opportunity_oracle_regret_per_passenger')}
            for name, row in value['models'].items()},
        comparisons_per_passenger=value['comparisons_per_passenger'])


def _seed_readout(scenes, expected_seed):
    """Audit the actual seed without changing it for the reused v86 readout functions."""
    indexed = {row['scenario']: row for row in scenes}
    rows, missing, selections, endpoints, cells = [], [], [], {}, {}
    for scenario in SCENARIOS:
        if scenario not in indexed:
            missing.extend(dict(external_seed=expected_seed, scenario=scenario, line_id=line,
                reason='missing_scenario') for line in LINES)
            value = _single_intervention([], 15079)
            cells[scenario] = _compact_policy(value)
            endpoints[scenario] = dict(complete=False, all_equal=False)
            continue
        scene = indexed[scenario]
        by_line = {row['line_id']: row for row in scene['pairs']}
        if len(by_line) != len(scene['pairs']) or set(by_line)-set(LINES):
            raise ValueError('Duplicate or undeclared heldout line')
        scene_rows = []
        for line in LINES:
            selection = scene['selection'][line]
            selections.append(dict(external_seed=expected_seed, scenario=scenario,
                line_id=line, selection=selection))
            if line not in by_line:
                missing.append(dict(external_seed=expected_seed, scenario=scenario,
                    line_id=line, reason='missing_pair', selection=selection))
                continue
            row = pair_readout(scene, by_line[line])
            row['external_seed'] = expected_seed
            rows.append(row)
            scene_rows.append(row)
        complete = len(scene_rows) == 3
        spread = {key: max(row['reference_full_endpoint'][key] for row in scene_rows)
            -min(row['reference_full_endpoint'][key] for row in scene_rows) if complete else None
            for key in COST_KEYS}
        endpoints[scenario] = dict(complete=complete, max_minus_min=spread,
            all_equal=complete and all(value == 0 for value in spread.values()))
        cells[scenario] = _compact_policy(_single_intervention(scene_rows, scene['departed_passengers']))
    budget = {key: sum(row['budget'][key] for row in scenes) for key in BUDGET_FIELDS}
    budget['new_fits'] = sum(row['budget']['new_fits'] for row in scenes)
    checks = dict(two_complete_scenarios=len(scenes) == 2 and all(row['status'] == 'complete' for row in scenes),
        runtime_scene_integrity=all(row['integrity']['all_pass'] for row in scenes),
        actual_declared_seed=all(row['external_seed'] == expected_seed for row in scenes),
        source_initializer=all(row['seed'] == 6207 for row in scenes),
        denominator=all(row['departed_passengers'] == 15079 for row in scenes),
        distinct_prefixes=len(scenes) == 2 and all(bool(row['prefix_id']) for row in scenes)
            and len({row['prefix_id'] for row in scenes}) == 2,
        six_declared_pairs=len(rows) == 6 and not missing,
        six_valid_pairs=len(rows) == 6 and all(row['valid_for_evaluation'] for row in rows),
        fixed_prior_reference=all(_reference_valid(row) for row in scenes),
        fixed_issue_window=all(12000 <= row['root']['started_at'] < 12600 for row in rows),
        reference_endpoints_equal=all(row['all_equal'] for row in endpoints.values()),
        actual_paired_parallel=len(rows) == 6 and all(row['overlap_seconds'] > 0 for row in rows),
        observed_labels=budget['labels_observed'] == 12,
        native_units=budget['prefixes'] == 2 and budget['suffixes'] == 12,
        native_budget=budget['total_native_seconds'] <= 169200,
        native_budget_identity=budget['total_native_seconds']
            == budget['native_prefix_seconds']+budget['suffix_native_seconds'],
        no_initializer_updates=budget['new_initialization_updates'] == 0,
        no_prefix_updates=budget['prefix_training_updates'] == 0,
        no_suffix_updates=budget['suffix_updates'] == 0,
        no_truth_feedback=budget['labels_fed_back'] == 0,
        no_new_fits=budget['new_fits'] == 0,
        frozen_source=all(row['frozen_source']['new_fits'] == 0
            and row['frozen_source']['source_model_exact'] is True for row in scenes),
        reference_metadata_actual_seed=all(row['reference_snapshot']['heldout_external_seed'] == expected_seed
            for row in scenes))
    return dict(external_seed=expected_seed, integrity=dict(all_pass=all(checks.values()), checks=checks),
        cells=cells, rows=rows, missing=missing, support_selection=selections, budget=budget,
        reference_endpoint_audit=endpoints)


def _cost_summary(cells):
    """Equal-weight completed full-day cells, never a sum of separate root queries."""
    complete = bool(cells) and all(row['status'] == 'complete' for row in cells)
    result = dict(cells=len(cells), complete_cells=sum(row['status'] == 'complete' for row in cells), models={})
    for name in ('reference', *MODELS):
        values = [cell['models'][name] for cell in cells]
        result['models'][name] = dict(
            mean_full_day_cost_per_passenger=sum(value['full_selected_cost_per_passenger']
                for value in values)/len(cells) if complete else None,
            mean_selected_minus_reference_per_passenger={key: sum(
                value['selected_minus_reference_per_passenger'][key] for value in values)/len(cells)
                for key in COST_KEYS} if complete else None,
            interventions=sum(value['interventions'] for value in values) if complete else None)
    result['comparisons'] = {}
    for label in ('fresh_minus_stale', 'stale_minus_legacy_narrow', 'fresh_minus_reference'):
        values = [cell['comparisons_per_passenger'][label] for cell in cells]
        costs = [value['cost'] for value in values] if complete else []
        result['comparisons'][label] = dict(
            mean={key: sum(value[key] for value in values)/len(cells) for key in COST_KEYS} if complete else None,
            wins=sum(value < 0 for value in costs) if complete else None,
            ties=sum(value == 0 for value in costs) if complete else None,
            losses=sum(value > 0 for value in costs) if complete else None,
            win_rate=sum(value < 0 for value in costs)/len(cells) if complete else None,
            tie_rate=sum(value == 0 for value in costs)/len(cells) if complete else None,
            loss_rate=sum(value > 0 for value in costs)/len(cells) if complete else None,
            worst_cell=max(costs) if complete else None)
    return result


def _compact_pair(row):
    return dict(external_seed=row['external_seed'], scenario=row['scenario'], line_id=row['line_id'],
        root=row['root'], valid_for_evaluation=row['valid_for_evaluation'],
        reference_hold_seconds=row['reference_hold_seconds'], candidate_hold_seconds=row['candidate_hold_seconds'],
        overlap_seconds=row['overlap_seconds'], physical=row['physical'],
        z=row['z'], actual_delta=row['actual_delta'], actual_delta_per_passenger=row['actual_delta_per_passenger'],
        truth_direction=row['truth_direction'], models={name: {key: value[key]
            for key in ('supported', 'predicted_delta', 'choice', 'fallback_reason', 'direction_correct',
                'two_action_regret_per_passenger', 'selected_minus_reference_per_passenger')}
            for name, value in row['models'].items()})


def analyze(scenes, audit, model):
    indexed = {}
    for scene in scenes:
        key = (scene['external_seed'], scene['scenario'])
        if key[0] not in SEEDS or key[1] not in SCENARIOS:
            raise ValueError('Only the three new declared external seeds and two scenarios may be scored')
        if key in indexed:
            raise ValueError('Duplicate seed/scenario collector')
        indexed[key] = scene
    seeds = {seed: _seed_readout([indexed[(seed, scenario)] for scenario in SCENARIOS
        if (seed, scenario) in indexed], seed) for seed in SEEDS}
    rows = [row for seed in seeds.values() for row in seed['rows']]
    cells = {str(seed)+'/'+scenario: seeds[seed]['cells'][scenario]
        for seed in SEEDS for scenario in SCENARIOS}
    overall = _cost_summary(list(cells.values()))
    by_seed = {str(seed): _cost_summary(list(value['cells'].values())) for seed, value in seeds.items()}
    by_scenario = {scenario: _cost_summary([seeds[seed]['cells'][scenario] for seed in SEEDS])
        for scenario in SCENARIOS}
    budget = {key: sum(seed['budget'][key] for seed in seeds.values()) for key in BUDGET_FIELDS}
    budget['new_fits'] = sum(seed['budget']['new_fits'] for seed in seeds.values())
    freeze_checks = _freeze_checks(audit, model)
    checks = dict(frozen_source_and_rule=all(freeze_checks.values()), all_six_declared_collectors=len(indexed) == 6,
        all_seed_integrity=all(seed['integrity']['all_pass'] for seed in seeds.values()),
        eighteen_valid_pairs=len(rows) == 18 and all(row['valid_for_evaluation'] for row in rows),
        six_distinct_new_prefixes=len(scenes) == 6 and len({row['prefix_id'] for row in scenes}) == 6,
        native_budget=budget['total_native_seconds'] <= 507600,
        native_units=budget['prefixes'] == 6 and budget['suffixes'] == 36,
        thirty_six_new_labels=budget['labels_observed'] == 36)
    complete = overall['complete_cells'] == 6
    promotion = dict(technical_integrity=all(checks.values()), complete_full_day_policies=complete,
        fresh_actually_intervenes=complete and overall['models']['fresh']['interventions'] > 0,
        fresh_mean_beats_stale=complete and overall['comparisons']['fresh_minus_stale']['mean']['cost'] < 0,
        fresh_mean_beats_reference=complete and overall['comparisons']['fresh_minus_reference']['mean']['cost'] < 0,
        every_cell_not_worse_than_stale=complete and overall['comparisons']['fresh_minus_stale']['worst_cell'] <= 0,
        every_cell_not_worse_than_reference=complete and overall['comparisons']['fresh_minus_reference']['worst_cell'] <= 0)
    for scenario, value in by_scenario.items():
        valid = value['complete_cells'] == 3
        for baseline in ('stale', 'reference'):
            promotion[scenario+'_mean_not_worse_than_'+baseline] = valid and value['comparisons'][
                'fresh_minus_'+baseline]['mean']['cost'] <= 0
    passed = all(promotion.values())
    return dict(status='complete' if all(checks.values()) else 'hold',
        integrity=dict(all_pass=all(checks.values()), checks=checks), freeze_checks=freeze_checks,
        primary=dict(policy=POLICY, overall=overall, by_seed=by_seed, by_scenario=by_scenario,
            by_cell=cells, reconstructed_from_existing_native_suffixes=True, new_native_simulation_seconds=0),
        query_metrics=dict(overall=_group(rows),
            by_line={line: _group([row for row in rows if row['line_id'] == line]) for line in LINES}),
        pairs=[_compact_pair(row) for row in rows], budget=dict(budget, new_neural_updates=0),
        seed_integrity={str(seed): dict(integrity=value['integrity'],
            reference_endpoint_audit=value['reference_endpoint_audit']) for seed, value in seeds.items()},
        exceptions=dict(missing=[row for seed in seeds.values() for row in seed['missing']],
            invalid_pairs=[dict(external_seed=row['external_seed'], scenario=row['scenario'],
                line_id=row['line_id'], checks=row['checks']) for row in rows if not row['valid_for_evaluation']],
            outside_action_support=[dict(external_seed=row['external_seed'], scenario=row['scenario'],
                line_id=row['line_id'], model=name, reason=row['models'][name]['fallback_reason'])
                for row in rows for name in MODELS if not row['models'][name]['supported']],
            zero_duration=[dict(external_seed=row['external_seed'], scenario=row['scenario'], line_id=row['line_id'])
                for row in rows if not row['checks']['nonzero_duration']],
            support_selection=[row for seed in seeds.values() for row in seed['support_selection']]),
        promotion=dict(passed=passed, checks=promotion, decision='Frozen one-intervention rule passed this three-seed confirmation'
            if passed else 'HOLD this frozen confirmation; no retuning on its truth or automatic replacement'),
        interpretation=dict(confirmation_seeds=list(SEEDS), source_initializer=6207,
            primary='Six actual full-day cells under one intervention at most; independent query sums are auxiliary',
            source_model='Frozen v86 coefficients, support domains and zero threshold; zero new fits',
            old_v86_not_pooled=True, old_v85_failure_unchanged=True,
            no_automatic_control_takeover=True,
            scope='Three new exogenous seeds with one source initialization; not repeated intervention or a comparison with historical final MC-WM'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'audit', 'model', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())['confirmation']
    expected = dict(seeds=list(SEEDS), scenarios=list(SCENARIOS), lines=list(LINES),
        issue_window=[12000, 12600], output_root='runs')
    if any(protocol[key] != value for key, value in expected.items()):
        raise ValueError('Confirmation protocol differs from the frozen three-seed declaration')
    scenes = []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            path = args.protocol.parent/protocol['output_root']/('external'+str(seed))/scenario/'collector_summary.json'
            if path.exists():
                scenes.append(json.loads(path.read_text()))
    value = analyze(scenes, json.loads(args.audit.read_text()), json.loads(args.model.read_text()))
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 128000:
        raise ValueError('Confirmation summary exceeds128KB; raw native records must stay resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=value['status'], bytes=len(raw.encode()), promotion=value['promotion']['passed'])))


if __name__ == '__main__':
    main()
