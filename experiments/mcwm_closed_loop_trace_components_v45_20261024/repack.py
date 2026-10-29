"""Recover the bounded summary from completed resident component arrays only.

No torch/model/environment module is imported.  Original failed runs and their
analysis stay intact; the only write is analysis_repacked/summary.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_trace_components_v45_20261024.components import (
    COMPONENTS, compare_pair, decompose_components,
)
from experiments.mcwm_closed_loop_trace_components_v45_20261024.summary import (
    CACHE_COLUMNS, EVENT_COLUMNS, checkpoint_small,
)


PART_FIELDS = ('native_target', 'reward_scaled', 'bootstrap_contribution',
    'value_contribution', 'sac_contribution', 'next_value', 'sac_q_without_entropy',
    'sac_entropy_bonus', 'target_subset', 'recorded_next_target_q', 'baseline_missing_reason')
FAILURE_MESSAGE = 'Reduce the resident summary before a bounded local pull'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def compact(value):
    if isinstance(value, float):
        return float(f'{value:.7g}')
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return value


def validate_checkpoint(detail, source, seeds):
    keys = {f'{actor}/{head}' for actor in seeds for head in seeds}
    require(set(detail['factor_rows']) == keys and set(detail['factor_pairs']) == keys,
            'Resident crossed factor grid is not the prescribed complete9 cells')
    source_bank = {seed: {(chain['line_id'], chain['arm']): chain
        for chain in source['seeds'][str(seed)]['chains']} for seed in seeds}
    residuals, diagonal_rows = [], 0
    for key in sorted(keys):
        actor, head = map(int, key.split('/'))
        for arm in ('low', 'high'):
            rows = detail['factor_rows'][key][arm]
            require(len(rows) == 45, 'A saved factor cell is missing real45-event rows')
            require(finite(rows), 'A saved factor row contains nonfinite data')
            for event, row in enumerate(rows):
                require(len(row['target_subset']) == 2 and sorted(row['target_subset']) == [0, 1],
                        'Saved native target membership is not two-of-two')
                factors = row['factor_detail']
                require(factors['actor_seed'] == actor and factors['head_seed'] == head
                        and factors['target_subset'] == row['target_subset'],
                        'Saved actor/head factor identity differs')
                if actor == head:
                    original = source_bank[actor][('705X', arm)]['rows'][event]
                    require(all(row[field] == original[field] for field in PART_FIELDS),
                            f'Old native diagonal differs at{key}/{arm}/{event}')
                    diagonal_rows += 1
            c = source_bank[seeds[0]][('705X', arm)]['c']
            require(all(source_bank[seed][('705X', arm)]['c'] == c for seed in seeds),
                    'Original executed ratios unexpectedly depend on scoring seed')
            calculated = decompose_components(rows, c)
            saved = detail['factor_pairs'][key][arm]
            require(calculated['trace_target'] == saved['trace_target'] and
                    calculated['native_target'] == saved['native_target'] and
                    calculated['links'] == saved['links'],
                    'Saved recurrence or original executed ratios differ')
            require(all(calculated['component_totals'][name] == saved['component_totals'][name]
                        for name in COMPONENTS), 'Saved component totals differ from resident rows')
            require(finite(saved), 'A saved decomposition contains nonfinite data')
            residuals.append(abs(saved['reconstruction_residual']))
        saved_pair = detail['factor_pairs'][key]['pair']
        calculated_pair = compare_pair(detail['factor_pairs'][key]['low'],
                                      detail['factor_pairs'][key]['high'])
        require(saved_pair['trace_target_difference'] == calculated_pair['trace_target_difference']
                and saved_pair['component_totals'] == calculated_pair['component_totals'],
                'Saved paired component accounting differs')
        residuals.append(abs(saved_pair['reconstruction_residual']))
    require(max(residuals) < 1e-8, 'Saved component closure exceeds its original tolerance')
    small = checkpoint_small(detail['cached_pairs'], detail['factor_pairs'], seeds)
    require(small['native_diagonal_pair_deltas_exact'], 'Saved old paired diagonals differ')
    require(set(detail['cached_pairs']) == {'7X', '102X', '705X'},
            'Saved three-line cached context is incomplete')
    for line, bank in detail['cached_pairs'].items():
        require(set(bank) == set(map(str, seeds)), 'Cached context is missing a declared scoring seed')
        for seed in seeds:
            for arm in ('low', 'high'):
                original = source_bank[seed][(line, arm)]
                calculated = decompose_components(original['rows'], original['c'])
                saved = bank[str(seed)][arm]
                require(calculated['trace_target'] == saved['trace_target'] and
                        calculated['component_totals'] == saved['component_totals'],
                        'Cached context differs from original resident v44 rows')
    noise = {}
    for metric, item in small['crossed_noise'].items():
        noise[metric] = {key: item[key] for key in ('grand_mean', 'sums_of_squares',
            'actor_draw_fixed_subset_ranges', 'subset_fixed_actor_ranges')}
        if metric in ('trace_delta', 'sac_q_without_entropy', 'entropy'):
            noise[metric]['matrix'] = [[
                detail['factor_pairs'][f'{actor}/{head}']['pair']['trace_target_difference']
                if metric == 'trace_delta' else
                detail['factor_pairs'][f'{actor}/{head}']['pair']['component_totals'][metric]
                for head in seeds] for actor in seeds]
    return dict(cached_context=small['cached_context'], focus705X=small['focus705X'],
        crossed_noise=noise, max_component_residual=max(residuals),
        proof=dict(saved_grid_cells=9, saved_rows_per_arm_per_cell=45,
            native_diagonal_rows_exact=diagonal_rows,
            v44_native_targets_parts_matching_executed_Q_exact=True,
            original_executed_ratios_exact=True, all_component_closures_pass=True,
            finite_arrays=True)), max(residuals)


def main():
    started = time.time()
    protocol = read(HERE/'protocol.json')
    require(protocol['scenarios'] == ['stable', 'local_speed_drop'] and
            protocol['checkpoints'] == ['actual_post24', 'post456'],
            'The original two cases and two checkpoints must remain fixed')
    # These checks document which already-completed control-flow guards made the
    # resident arrays writable.  They are not a new model/RNG/Adam inspection.
    source_code = (HERE/'run.py').read_text()
    guard = ('learner.config.ensemble_size == 2 and learner.config.target_subset_size == 2',
             'and not learner.config.target_subset_replacement',
             "proof = unchanged(learner, before, rng)",
             "if not _equal(adam, optimizer_versions(learner)):",
             "write(out/'component_detail.json', details)")
    require(all(text in source_code for text in guard),
            'Original completed-run configuration or immutable-state guards are absent')
    cases, failures = {}, {}
    source_budget = None
    for scenario in protocol['scenarios']:
        out = HERE/'runs'/scenario
        failure = read(out/'failure.json')
        require(failure['status'] == 'failed' and failure['type'] == 'RuntimeError'
                and failure['message'] == FAILURE_MESSAGE,
                'Recovery is restricted to the completed scoring/summary-size failure')
        detail = read(out/'component_detail.json')
        arrays = HERE.parent/protocol['array_source_version']/'runs'/scenario
        source, source_summary = read(arrays/'probe_detail.json'), read(arrays/'summary.json')
        require(source['integrity']['all_pass'] and source_summary['integrity']['all_pass'],
                'Original v44 native/source proof failed')
        budget = source['source_budget']
        require(budget['feedback_slots'] == 285 and budget['updates'] == 456
                and budget['initializer_updates'] == 50000 and budget['horizon_seconds'] == 24000,
                'Source285-slot/456-update/horizon contract differs')
        require(source['source_contract']['initialization_seed'] == 6202 and
                source['source_contract']['native_external_seed'] == 106243,
                'Source initialization or native external seed differs')
        if source_budget is None:
            source_budget = budget
        require(budget == source_budget, 'Scenario source budgets differ')
        require(set(detail) == set(protocol['checkpoints']), 'Saved checkpoints are incomplete')
        checkpoints = {}
        for checkpoint in protocol['checkpoints']:
            checkpoints[checkpoint], _ = validate_checkpoint(detail[checkpoint],
                source['checkpoints'][checkpoint], protocol['diagnostic_seeds'])
        cases[scenario] = dict(status='complete', checkpoints=checkpoints,
            original_scoring_wall_seconds=failure['elapsed_seconds'])
        failures[scenario] = dict(type=failure['type'], message=failure['message'],
            phase='summary_size_check_after_all_component_arrays_and_immutable_state_guards',
            original_elapsed_seconds=failure['elapsed_seconds'], original_failed_files_preserved=True)
    result = dict(status='complete', priority='high', cases=cases,
        integrity=dict(all_pass=True, recovery_scope='read_existing_JSON_and_recalculate_accounting_only',
            original_models_Adam_RNG_guards_completed_before_array_write=True,
            original_model_guard_provenance='original_run_control_flow_and_packaging_only_failure',
            fresh_model_Adam_RNG_reaudit=False, model_modules_imported=False),
        source_budget=source_budget,
        source_contract=dict(initialization_seed=6202, native_external_seed=106243,
            F_unchanged=True, trust_unchanged=True, source_version=protocol['model_source_version'],
            target_config=dict(ensemble_size=2, subset_size=2, replacement=False,
                membership_varies=False, source='original_run_guard_and_saved_full_membership_rows')),
        diagnostic_budget=dict(new_training_updates=0, new_feedback_slots=0,
            new_native_branches=0, new_native_seconds=0, checkpoints_downloaded=0,
            recovery_checkpoint_loads=0, recovery_scoring_calls=0),
        original_failures=failures,
        schema=dict(components=list(COMPONENTS), cached_context_columns=CACHE_COLUMNS,
            event_columns=EVENT_COLUMNS, bin_columns=['name', 'events', 'total', *COMPONENTS],
            seeds=protocol['diagnostic_seeds'], head_factor='full2of2 membership, order-only numerical control',
            event_scope='fixed45 real event ordinals, not common-clock or300s truth'),
        recovery_wall_seconds=time.time()-started,
        decision=dict(training_started=False, target_change_installed=False,
            next_gate='Review component and actor/head-order attribution, then name one target candidate.'))
    encoded = json.dumps(compact(result), separators=(',', ':'), allow_nan=False)+'\n'
    size = len(encoded.encode())
    require(size <= 48000, f'Repacked summary still exceeds48KB: {size}')
    destination = HERE/'analysis_repacked'
    destination.mkdir(exist_ok=False)
    with (destination/'summary.json').open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status='complete', integrity=dict(all_pass=True), bytes=size,
        original_failure='summary_size_only', new_training_updates=0, model_loads=0,
        recovery_wall_seconds=time.time()-started)), flush=True)


if __name__ == '__main__':
    main()
