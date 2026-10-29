"""Resident analysis of new paired coverage, finite300 predictions and long Q."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
ARMS = ('zero_memory', 'per_head_level', 'shared_level')


def read(path):
    return json.loads(Path(path).read_text())


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sign(value):
    return (value > 0) - (value < 0)


def decision(delta, benefit):
    choice = int(delta > 0)
    return dict(delta=delta, choice=choice,
        direction_correct=sign(delta) == sign(benefit) if benefit != 0 else None,
        regret=max(-benefit, 0.) if choice else max(benefit, 0.))


def truth(record):
    outcomes = record['outcomes']
    valid = all(outcomes[a]['truth']['status'] == 'complete' and
        outcomes[a]['truth']['dt'] == 300. for a in ('low', 'high'))
    if not valid:
        return None
    return {k:outcomes['low']['truth'][k] - outcomes['high']['truth'][k]
        for k in ('wait', 'ride', 'cost')}


def aggregate(rows, models):
    result = dict(n=len(rows), models={})
    for model in models:
        valid = [r['models'][model] for r in rows if r.get('benefit') is not None and model in r['models']]
        result['models'][model] = dict(n=len(valid), direction_n=sum(v['direction_correct'] is not None for v in valid),
            correct=sum(v['direction_correct'] is True for v in valid),
            high_choices=sum(v['choice'] for v in valid), regret=sum(v['regret'] for v in valid))
    for comparator in ('zero_memory', 'per_head_level'):
        values = [r['models']['shared_level']['regret'] - r['models'][comparator]['regret']
            for r in rows if r.get('benefit') is not None and all(m in r['models'] for m in ('shared_level', comparator))]
        result['shared_minus_' + comparator] = dict(n=len(values), net=sum(values),
            improved=sum(v < 0 for v in values), worsened=sum(v > 0 for v in values),
            gain=sum(max(-v, 0.) for v in values), loss=sum(max(v, 0.) for v in values))
    return result


def quality(record):
    return dict(zero_next_event=[a for a in ('low','high') if
        record['outcomes'][a]['next_event'].get('zero_duration') is True],
        physical=record['comparison']['physical_separation'],
        overlap_seconds=record['pair_receipt']['overlap_seconds'],
        actual_holds=record['holds'],
        next_event_delta={k:record['outcomes']['low']['next_event'].get(k, 0.) -
            record['outcomes']['high']['next_event'].get(k, 0.) for k in ('wait','ride','cost')}
        if all(record['outcomes'][a]['next_event'].get('status') == 'complete' for a in ('low','high')) else None)


def analyze(root):
    return analyze_cases(Path(root))


def compact_aggregate(rows, models):
    result = aggregate(rows, models)
    return dict(n=result['n'], models={k:[v['n'],v['direction_n'],v['correct'],v['high_choices'],v['regret']]
        for k,v in result['models'].items()}, shared_vs_controls=[
            list(result['shared_minus_' + arm].values()) for arm in ('zero_memory','per_head_level')])


def rounded(value):
    if isinstance(value, float):
        return float(format(value, '.6g'))
    if isinstance(value, dict):
        return {k:rounded(v) for k,v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def compact_fit(reader):
    value = reader.summary()
    return dict(pairs=value['pairs'], outcome_slots=value['outcome_slots'],
        by_line=value['by_line'], design=value['design'], ridge=value['ridge'],
        coefficients=[[v[k] for k in ('wait','ride','cost')] for v in value['coefficients'].values()],
        training_mae=[value['training_error'][k] for k in ('wait_mae','ride_mae','cost_mae')],
        refits=value['refits'])


def compact_state(row, models):
    def costs(value):
        return None if value is None else [value[k] for k in ('wait','ride','cost')]
    predictions = []
    for name in models:
        value = row['models'].get(name)
        predictions.append(None if value is None else [value['delta'],value['choice'],value['direction_correct'],
            value['regret'],value.get('min_heads')])
    q = row['quality']
    support = row['finite_support']['finite300_lowdim']
    return [row['scenario'],row['slot'],row['line'],row['window'],row['issue'],costs(row['benefit']),
        predictions,[costs(row['finite_predictions'][name]) for name in ('finite300_line','finite300_lowdim')],
        [support['pairs'],support['headway_range'],support['headway_extrapolation']],
        [q[k] for k in ('zero_next_event','physical','departure_separable','next_arrival_separable',
            'overlap_seconds','actual_holds')] + [costs(q['next_event_delta'])]]


def analyze_cases(root):
    from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.difference import DirectDifferenceReadout
    p = read(HERE/'protocol.json')
    models = [*ARMS, 'finite300_line', 'finite300_lowdim', 'fixed0']
    cases, states, missing = {}, [], []
    for scenario in p['scenarios']:
        collection, shadow = root/'collection'/scenario, root/'shadow'/scenario
        native, parent = read(collection/'summary.json'), read(shadow/'summary.json')
        require(native['integrity']['all_pass'] and parent['integrity']['all_pass'], 'Native or matched-learning integrity failed')
        workers = {arm:read(shadow/arm/'summary.json') for arm in ARMS}
        require(all(s['integrity']['all_pass'] and s['budget']['feedback_slots'] == 285 and
            s['budget']['shadow_updates'] == 456 for s in workers.values()), 'Worker285/456 or integrity failed')
        require(parent['optimizer_overlap_wall_seconds'] > 0., 'Three training workers never overlapped')
        records = read(collection/'pair_truth.json')
        train = [r for r in records if r['role'] == 'train' and r.get('selected_for_training')]
        heldout = [r for r in records if r['role'] == 'heldout']
        require(len(train) == 9 and len(heldout) == 6, 'New pair or heldout coverage is incomplete')
        require(all(truth(r) is not None for r in train), 'Admitted training pair has incomplete300 truth')
        for record in records:
            for arm in ('low','high'):
                absent = {k:v for k,v in record['outcomes'][arm].get('physical',{}).get('missing',{}).items() if v}
                if absent:
                    missing.append([scenario,record['id'],arm,'physical_observation_missing',absent])
        readers = dict(finite300_line=DirectDifferenceReadout(line_only=True),
            finite300_lowdim=DirectDifferenceReadout(line_only=False))
        dataset = read(collection/'dataset.json')
        admitted, fit_log = set(), []
        for packet in dataset['packets']:
            receipt = packet['receipt']
            new = [r for r in train if r['id'] in receipt['pair_ids']]
            require(not admitted.intersection(r['id'] for r in new), 'Finite300 readout received a repeated pair')
            if new:
                for reader in readers.values():
                    reader.accept(new, now=receipt['processed_at'])
                    reader.refit()
                admitted.update(r['id'] for r in new)
                fit_log.append([packet['index'],receipt['processed_at'],len(admitted)])
        require(len(admitted) == 9, 'Finite300 supervision differs from budget-admitted pairs')
        scores = {arm:{r['slot']:r for r in jsonlines(shadow/arm/'heldout.jsonl')} for arm in ARMS}
        require(all(len(rows) == 6 for rows in scores.values()), 'Heldout model scores missing or repeated')
        case_states = []
        for record in sorted(heldout, key=lambda r:r['id']):
            slot = record['id'].removeprefix('heldout/')
            require(all(slot in scores[a] for a in ARMS), 'Scored heldout issue does not match native truth')
            benefit = truth(record)
            row = dict(scenario=scenario, slot=slot, line=record['line_id'],
                window=record['window_name'], issue=record['issued_at'], benefit=benefit, models={})
            q = quality(record)
            row['quality'] = dict(zero_next_event=q['zero_next_event'],
                physical=q['physical'].get('status'), departure_separable=q['physical'].get('departure_separable'),
                next_arrival_separable=q['physical'].get('next_arrival_separable'),
                overlap_seconds=q['overlap_seconds'], actual_holds=q['actual_holds'],
                next_event_delta=q['next_event_delta'])
            if benefit is None:
                missing.append([scenario,slot,'incomplete300'])
            for arm in ARMS:
                saved = scores[arm][slot]
                require(saved['sequence'] == record['source_sequence'] and saved['issue'] == record['issued_at'],
                    'Frozen prediction joined to a different native issue')
                delta = saved['prediction']['delta_min']
                if benefit is not None:
                    row['models'][arm] = decision(delta, benefit['cost'])
                    row['models'][arm].update(min_heads=saved['prediction']['min_heads'],
                        bellman_delta=[saved['bellman_high_minus_low'][k] for k in
                            ('reward_scaled','value_contribution','sac_contribution','target')])
            row['finite_predictions'] = {}
            row['finite_support'] = {}
            for name, reader in readers.items():
                predicted = reader.predict(record['features'])
                row['finite_predictions'][name] = predicted['gain']
                observed = [r['features'][3] for r in reader.records if r['line_id'] == record['line_id']]
                row['finite_support'][name] = dict(pairs=len(observed),
                    headway_range=[min(observed),max(observed)] if observed else None,
                    headway_extrapolation=bool(observed and not min(observed) <= record['features'][3] <= max(observed)))
                if predicted['status'] == 'supported' and benefit is not None:
                    row['models'][name] = decision(predicted['gain']['cost'], benefit['cost'])
                elif predicted['status'] != 'supported':
                    missing.append([scenario,slot,name,'unsupported'])
            if benefit is not None:
                row['models']['fixed0'] = decision(0., benefit['cost'])
            case_states.append(row)
        pair_targets = []
        target_columns = ('reward_scaled','value_contribution','sac_contribution','target')
        for arm in ARMS:
            values = jsonlines(shadow/arm/'pair_targets.jsonl')
            require(len(values) == 27, 'Training pair target traces require before/after/frozen per pair')
            for index in range(9):
                stages = {v['phase']:v for v in values if v['pair_index'] == index}
                require(set(stages) == {'packet_before','packet_after','frozen_final'}, 'A paired target stage is absent')
                pair_targets.append([arm,index,[[stages[phase]['prediction']['delta_min'],
                    *[stages[phase]['bellman_high_minus_low'][k] for k in target_columns]]
                    for phase in ('packet_before','packet_after','frozen_final')]])
        line_result = {line:compact_aggregate([r for r in case_states if r['line'] == line], models)
            for line in p['focus_lines']}
        window_result = {window:compact_aggregate([r for r in case_states if r['window'] == window], models)
            for window in ('early','tail')}
        finite_error = {name:{k:sum(abs(r['finite_predictions'][name][k]-r['benefit'][k]) for r in case_states
            if r['benefit'] is not None and r['finite_predictions'][name] is not None)/sum(
                r['benefit'] is not None and r['finite_predictions'][name] is not None for r in case_states)
            for k in ('wait','ride','cost')} for name in readers}
        native_budget = {k:v for k,v in native['budget'].items() if not k.endswith('_ids') and
            'sequence' not in k and isinstance(v,(int,float))}
        cases[scenario] = dict(native_budget=native_budget,
            integrity=dict(collection=native['integrity']['all_pass'],shadow=parent['integrity']['all_pass'],
                worker_all_pass={a:s['integrity']['all_pass'] for a,s in workers.items()}),
            overlap_seconds=parent['optimizer_overlap_wall_seconds'], target_contract=workers[ARMS[0]]['target_contract'],
            finite_fit={name:compact_fit(r) for name,r in readers.items()}, finite_fit_packets=fit_log,
            finite300_heldout_mae=finite_error, overall=compact_aggregate(case_states,models),
            by_line=line_result, by_window=window_result,
            training_pairs=[[r['index'],r['line_id'],r['window_name'],r['issued_at'],r['admitted_at'],
                r['admitted_at']-r['issued_at'],[truth(r)[k] for k in ('wait','ride','cost')],quality(r)['zero_next_event'],
                r['comparison']['physical_separation']['status']] for r in train],
            pair_target_evolution={arm:[r[2] for r in pair_targets if r[0] == arm] for arm in ARMS})
        states.extend(case_states)
    return rounded(dict(status='complete',integrity=dict(all_pass=True),models=models,
        columns=dict(states=['scenario','slot','line','window','issue','true_wait_ride_cost_gain','models',
                'finite300_line_lowdim_gain','finite300_support','quality'],
            state_model=['delta','choice','direction_correct','regret','min_heads'],
            costs=['wait','ride','cost'],finite_support=['line_pairs','headway_range','headway_extrapolation'],
            quality=['zero_next_event','physical','departure_separable','next_arrival_separable','overlap_seconds','actual_holds','next_event_gain'],
            finite_coefficient_features=['7X_intercept','102X_intercept','705X_intercept','headway_imbalance'],
            aggregate_model=['n','nonzero_truth_direction_n','correct','high_choices','regret_sum'],
            shared_vs_controls=['zero_memory','per_head_level'],
            shared_regret=['n','net','improved','worsened','gain','loss'],
            training_pairs=['index','line','window','issue','admitted','age','true_gain','zero_next_event','physical'],
            pair_targets='Dictionary by arm; each list indexed by the fixed pair_index0..8',pair_stages=['packet_before','packet_after','frozen_final'],
            pair_stage_values=['delta_Qmin',*target_columns]),
        budget=p['compute_budget'], cases=cases, states=[compact_state(row,models) for row in states],missing=missing,
        display_precision='6 significant digits after full-precision direction/regret computations; full records stay resident',
        decision_gate='New heldout finite300 direction/regret and stable losses must improve before another independent seed; longQ direction is a different estimand and is reported separately.',
        limitations=['The300-second readout is passive and has no actor/Q/F/trust gradient or control path.',
            'Three critics use legal next-event TD and a mixed long-term Bellman operator; their Q difference is not a calibrated300-second cost prediction.',
            'New heldouts share a predeclared common prefix/continuation and provide development evidence, not all-day deployed-controller superiority.']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.root)
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
        require(len(payload.encode()) <= read(HERE/'protocol.json').get('summary_max_bytes', 24000), 'Resident summary exceeds transfer budget')
        with args.out.open('x') as output:
            output.write(payload)
        print(json.dumps(dict(status=result['status'], summary_bytes=len(payload.encode()))), flush=True)
    except Exception as error:
        with args.out.with_name('paired_coverage_summary_failure.json').open('x') as output:
            json.dump(dict(type=type(error).__name__, message=str(error)), output)
        raise


if __name__ == '__main__':
    main()
