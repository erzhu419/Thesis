"""Matched285/456 recovery with four real heldouts; missing states remain absent."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.analyze import (
    ARMS, compact_aggregate, compact_fit, decision, jsonlines, quality, read,
    require, rounded, truth,
)
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.difference import DirectDifferenceReadout

MODELS = [*ARMS, 'finite300_line', 'finite300_lowdim', 'fixed0', 'fixedhigh']
STAGES = ('packet_before', 'packet_after', 'frozen_final')
TARGETS = ('reward_scaled', 'value_contribution', 'sac_contribution', 'target')
MISSING_SLOTS = {'7X/tail', '102X/tail'}
SOURCE_FAILURES = {'all_declared_slots_selected', 'heldout6'}


def target_delta(score):
    crn = score['bellman_high_minus_low']
    native = score['native_mc_bellman_high_minus_low']
    noise = score['crn_sampling_noise']
    require(score['probe_contract']['primary'] == 'common_random_numbers_single_row_same_seed',
            'Paired Bellman targets do not use common random numbers')
    return [*[crn[key] for key in TARGETS], native['target'], noise['target']]


def state_compact(row):
    def costs(value):
        return None if value is None else [value[key] for key in ('wait', 'ride', 'cost')]
    models = []
    for name in MODELS:
        value = row['models'].get(name)
        models.append(None if value is None else [value.get('delta'), value['choice'],
            value['direction_correct'], value['regret'], value.get('min_heads')])
    return [row['scenario'], row['slot'], row['line'], row['window'], row['issue'],
        costs(row['benefit']), models,
        [costs(row['finite_predictions'][name]) for name in ('finite300_line', 'finite300_lowdim')],
        row['finite_support'], row['quality']]


def fit_readouts(train, dataset):
    readers = dict(finite300_line=DirectDifferenceReadout(line_only=True),
                   finite300_lowdim=DirectDifferenceReadout(line_only=False))
    admitted, fit_log = set(), []
    for packet in dataset['packets']:
        receipt = packet['receipt']
        new = [record for record in train if record['id'] in receipt['pair_ids']]
        require(not admitted.intersection(record['id'] for record in new), 'Repeated fit pair')
        if new:
            for reader in readers.values():
                reader.accept(new, now=receipt['processed_at'])
                reader.refit()
            admitted.update(record['id'] for record in new)
            fit_log.append([packet['index'], receipt['processed_at'], len(admitted)])
    require(len(admitted) == 9, 'Readout labels differ from the nine admitted pairs')
    return readers, fit_log


def grouped(rows):
    value = compact_aggregate(rows, MODELS)
    return [value['n'], [value['models'][name] for name in MODELS], value['shared_vs_controls']]


def analyze(root, source=None):
    root = Path(root)
    source = Path(source) if source is not None else HERE.parent/'mcwm_closed_loop_paired_coverage_v37_20261024'
    protocol = read(source/'protocol.json')
    cases, states, missing = {}, [], []
    for scenario in protocol['scenarios']:
        native_root, learned_root = source/'runs/collection'/scenario, root/scenario
        native, parent = read(native_root/'summary.json'), read(learned_root/'summary.json')
        failed = {key for key, value in native['integrity']['checks'].items() if not value}
        require(failed == SOURCE_FAILURES and not native['integrity']['all_pass'],
                'Original failed collection scope differs from the declared missing heldouts')
        require(parent['integrity']['all_pass'], 'Recovery parent training integrity failed')
        require(parent['optimizer_overlap_wall_seconds'] > 0., 'Three optimizers did not overlap')
        workers = {arm: read(learned_root/arm/'summary.json') for arm in ARMS}
        require(all(value['integrity']['all_pass'] and value['budget']['feedback_slots'] == 285
            and value['budget']['shadow_updates'] == 456 for value in workers.values()),
            'A worker did not complete the matched285/456 recovery')
        records = read(native_root/'pair_truth.json')
        train = [record for record in records if record['role'] == 'train' and record.get('selected_for_training')]
        heldout = [record for record in records if record['role'] == 'heldout']
        require(len(train) == 9 and {record['index'] for record in train} == set(range(9)),
                'The nine fixed training pairs are absent or repeated')
        slots = {record['id'].removeprefix('heldout/') for record in heldout}
        expected = {line+'/'+window for line in protocol['focus_lines'] for window in ('early', 'tail')}
        require(len(heldout) == 4 and expected-slots == MISSING_SLOTS,
                'Existing heldout coverage differs from the four available issues')
        require(all(truth(record) is not None for record in train+heldout), 'Incomplete300 response')
        for slot in sorted(MISSING_SLOTS):
            missing.append([scenario, slot, 'no_native_issue_in_predeclared_window'])
        readers, fit_log = fit_readouts(train, read(native_root/'dataset.json'))
        scores = {arm: {value['slot']: value for value in jsonlines(learned_root/arm/'heldout.jsonl')}
                  for arm in ARMS}
        require(all(set(values) == slots for values in scores.values()), 'A scored heldout is missing or invented')
        case_states = []
        for record in sorted(heldout, key=lambda value: value['id']):
            slot, benefit = record['id'].removeprefix('heldout/'), truth(record)
            q = quality(record)
            row = dict(scenario=scenario, slot=slot, line=record['line_id'],
                window=record['window_name'], issue=record['issued_at'], benefit=benefit, models={},
                quality=[q['zero_next_event'], q['physical'].get('status'),
                    q['physical'].get('departure_separable'), q['physical'].get('next_arrival_separable'),
                    q['overlap_seconds'], q['actual_holds'], q['next_event_delta']],
                finite_predictions={})
            for arm in ARMS:
                saved = scores[arm][slot]
                require(saved['sequence'] == record['source_sequence'] and saved['issue'] == record['issued_at'],
                        'Frozen Q prediction belongs to a different physical issue')
                row['models'][arm] = dict(decision(saved['prediction']['delta_min'], benefit['cost']),
                    min_heads=saved['prediction']['min_heads'])
            for name, reader in readers.items():
                predicted = reader.predict(record['features'])
                require(predicted['status'] == 'supported', 'A declared line lacks training pairs')
                row['finite_predictions'][name] = predicted['gain']
                row['models'][name] = decision(predicted['gain']['cost'], benefit['cost'])
            observed = [value['features'][3] for value in readers['finite300_lowdim'].records
                        if value['line_id'] == record['line_id']]
            row['finite_support'] = [len(observed), [min(observed), max(observed)],
                not min(observed) <= record['features'][3] <= max(observed)]
            row['models']['fixed0'] = decision(0., benefit['cost'])
            row['models']['fixedhigh'] = dict(decision(1., benefit['cost']), delta=None)
            case_states.append(row)
        pair_targets = {}
        for arm in ARMS:
            values = jsonlines(learned_root/arm/'pair_targets.jsonl')
            require(len(values) == 27, 'Nine pairs require three target stages each')
            pair_targets[arm] = []
            for index in range(9):
                group = [value for value in values if value['pair_index'] == index]
                phases = {value['phase']: value for value in group}
                require(len(group) == 3 and set(phases) == set(STAGES), 'A target stage is missing or duplicated')
                pair_targets[arm].append([[float(format(value,'.5g')) for value in
                    [phases[phase]['prediction']['delta_min'],*target_delta(phases[phase])]]
                    for phase in STAGES])
        mae = {name: {key: sum(abs(row['finite_predictions'][name][key]-row['benefit'][key])
                    for row in case_states)/len(case_states) for key in ('wait', 'ride', 'cost')}
               for name in readers}
        cases[scenario] = dict(training_integrity=dict(all_pass=True,
            source_full_integrity=False, source_failed_checks=sorted(failed),
            parent=True, workers={arm: value['integrity']['all_pass'] for arm,value in workers.items()}),
            native_budget={key:native['budget'][key] for key in ('target_labels','packets',
                'ordinary_labels','paired_action_labels','training_pairs','prefix_native_seconds',
                'suffixes','suffix_native_seconds','native_model_optimizer_updates')},
            worker_budgets={arm:[value['budget'][key] for key in
                ('feedback_slots','shadow_updates','ordinary_slots','paired_slots','paired_zero_duration_rows')]
                for arm,value in workers.items()},
            optimizer_overlap_seconds=parent['optimizer_overlap_wall_seconds'],
            target_contract=workers[ARMS[0]]['target_contract'],
            heldout_coverage=dict(expected=6, actual=4, missing=sorted(MISSING_SLOTS)),
            finite_fit={name:compact_fit(reader) for name,reader in readers.items()},
            finite_fit_packets=fit_log, finite300_heldout_mae=mae,
            overall=compact_aggregate(case_states, MODELS),
            by_line={line:grouped([row for row in case_states if row['line']==line])
                     for line in protocol['focus_lines']},
            by_window={window:grouped([row for row in case_states if row['window']==window])
                       for window in ('early','tail')},
            training_pairs=[[record['index'],record['line_id'],record['window_name'],record['issued_at'],
                record['admitted_at'],[truth(record)[key] for key in ('wait','ride','cost')],
                quality(record)['zero_next_event']] for record in sorted(train,key=lambda value:value['index'])],
            pair_target_evolution=pair_targets)
        states.extend(case_states)
    return rounded(dict(status='partial_complete', scientific_complete=False,
        training_integrity=dict(all_pass=True, scope='training_data_only'),
        formal_independent_gate=False, models=MODELS, cases=cases,
        states=[state_compact(row) for row in states], missing=missing,
        compute=dict(new_native_seconds=0,new_labels=0,shadow_optimizer_updates=2736,
            reused_feedback_slots_per_scenario=285,updates_per_worker=456,readout_SGD_updates=0),
        columns=dict(states=['scenario','slot','line','window','issue','true_wait_ride_cost_gain','models',
            'finite_line_lowdim_wait_ride_cost','finite_support','quality'],
            state_model=['delta','high_choice','direction_correct','regret','min_heads'],
            pair_stages=list(STAGES),pair_stage=['Qmin_high_minus_low',*TARGETS,
                'nativeMC_target_delta','nativeMC_minus_CRN_target'],
            pair_targets='Each arm has nine entries indexed by declared training-pair index0..8.',
            training_pairs=['index','line','window','issue','admitted','wait_ride_cost_gain','zero_next_event'],
            aggregate_model=['n','nonzero_truth_n','correct','high_choices','regret_sum'],
            group=['n','model_rows_in_models_order','shared_vs_controls'],
            worker_budget=['feedback_slots','updates','ordinary_slots','paired_slots','zero_duration_pair_rows'],
            shared_vs_controls=['zero_memory','per_head_level'],
            shared_regret=['n','net','improved','worsened','gain','loss'],
            finite_support=['line_pairs','headway_range','headway_extrapolation'],
            quality=['zero_next_event','physical','departure_separable','next_arrival_separable',
                'parallel_overlap_seconds','actual_holds','next_event_gain']),
        limitations=['Four predeclared heldouts remain missing; all comparisons use the eight actual issues only.',
            'Finite300 passive readouts and mixed long-term Bellman Q estimate different outcomes; Q has no300-cost MAE.',
            'Fixedhigh is an action selector, not a benefit predictor; its delta is null.',
            'NativeMC paired bootstrap differences include independent action draws; common-random-number deltas are primary.',
            'No new native control policies were executed; stable regret changes are paired diagnostic evidence.'],
        display_precision='Pair target stages5 significant digits; other floats6; direction/regret use full precision.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.root, args.source)
        payload = json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
        require(len(payload.encode()) <= 24000, 'Recovery summary exceeds24000-byte transfer budget')
        with args.out.open('x') as output:
            output.write(payload)
        print(json.dumps(dict(status=result['status'],summary_bytes=len(payload.encode()))),flush=True)
    except Exception as error:
        with args.out.with_name('recovery_summary_failure.json').open('x') as output:
            json.dump(dict(type=type(error).__name__,message=str(error)),output)
        raise


if __name__ == '__main__':
    main()
