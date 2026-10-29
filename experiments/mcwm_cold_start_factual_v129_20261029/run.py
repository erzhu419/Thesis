"""One server-resident offline study; no simulator, policy or controller imports."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_service_risk_contract_v126_20261029.interface import overrun_target, risk_inputs
from forecast import initialize, forecast
from metrics import score, gate
from solver import fit as solve


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_scene(root, scene):
    base = root/'experiments/mcwm_full_receiver_observation_v127_20261029'
    original = records(root/'experiments/mcwm_service_opportunity_v121_20261029/runs'/scene/'selected_records.jsonl')
    by_sequence = {row['selection']['sequence']:row for row in original}
    rows = records(base/'runs'/scene/'derived_records.jsonl')
    if len(rows) != 285 or len(by_sequence) != 285:
        raise ValueError('The original285-root selection is incomplete')
    sequences = set()
    for row in rows:
        seq = row['meta']['sequence']
        old = by_sequence[seq]
        if seq in sequences or row['meta'] != old['context']['meta'] or row['selection'] != old['selection']:
            raise ValueError('Derived root is not the immutable selected root')
        sequences.add(seq)
        if row['inputs'] is None or row['inputs'] != risk_inputs(old, row):
            raise ValueError('Current n,h inputs are incomplete or misbound')
        if row['passivity_passed'] is not True:
            raise ValueError('Passive observation was not qualified')
    rows.sort(key=lambda r:(r['meta']['observed_at'], r['meta']['sequence']))
    packets = Counter(r['selection']['packet_deadline'] for r in rows)
    if sorted(packets) != list(range(1200,18001,300)) or set(packets.values()) != {5}:
        raise ValueError('Original57 packets of5 roots were changed')
    if sum(r['selection']['packet_deadline'] <= 9000 for r in rows) != 135:
        raise ValueError('Shared initialization does not contain135 roots')

    def truth(row):
        target = overrun_target(by_sequence[row['meta']['sequence']])
        if row['target'] != target or target['value'] not in (0,1):
            raise ValueError('Target differs from original60-second evidence or is unknown')
        return target['value']

    return rows, truth


def run(root, out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    diagnostics, scores, scene_runs = [], {}, {}
    result = dict(status='technical_hold', technical_pass=False)
    fit_log = (out/'fits.jsonl').open('x')

    def record_fit(samples, dimension, metadata):
        if len(diagnostics) >= 118:
            raise RuntimeError('The frozen118-fit budget is exhausted')
        clock = time.monotonic()
        fitted = solve(samples, dimension)
        entry = dict(**metadata, samples=len(samples), positives=sum(y for _,y in samples),
                     wall_seconds=time.monotonic()-clock, **fitted)
        diagnostics.append(entry)
        fit_log.write(json.dumps(entry, separators=(',', ':'), allow_nan=False)+'\n')
        fit_log.flush()
        if not fitted['converged']:
            raise RuntimeError('Fixed numerical solver HOLD: '+fitted['reason'])
        return fitted

    try:
        prior = json.loads((root/'experiments/mcwm_full_receiver_observation_v127_20261029/summary.json').read_text())
        if not prior['technical_pass'] or not prior['shared_prefix']['all_pass']:
            raise ValueError('The source replay or shared prefix is not qualified')
        for case in prior['cases'].values():
            if not all(case['observation'][k] for k in ('collection_complete','field_coverage_complete','inputs_complete','labels_complete','passivity_all_pass')):
                raise ValueError('The source lacks required coverage')
        # The old v126 single-class HOLD is retained, not used as this new model's gate.
        loaded = {scene:load_scene(root, scene) for scene in ('stable','local_speed_drop')}
        prefixes = {scene:[r for r in rows if r['selection']['packet_deadline'] <= 9000]
                    for scene,(rows,truth) in loaded.items()}
        for a,b in zip(prefixes['stable'], prefixes['local_speed_drop']):
            normalized = lambda r: {**r,'meta':{k:v for k,v in r['meta'].items() if k!='source_group'}}
            if normalized(a) != normalized(b):
                raise ValueError('The common135 initialization rows differ')
            if loaded['stable'][1](a) != loaded['local_speed_drop'][1](b):
                raise ValueError('Original common-prefix targets differ')
        initial = initialize(prefixes['stable'], loaded['stable'][1], record_fit)
        write(out/'initialization.json', dict(full=initial['full'], hold_only=initial['hold_only'],
            rows=len(initial['history']), positives=sum(y for _,y in initial['history']),
            nonzero_n=sum(inputs[0]>0 for inputs,y in initial['history'])))
        initial_copy = dict(full=initial['full'], hold_only=initial['hold_only'], history=list(initial['history']))
        for scene,(rows,truth) in loaded.items():
            directory = out/scene
            directory.mkdir()
            with (directory/'trace.jsonl').open('x') as trace:
                def emit(event):
                    trace.write(json.dumps(event,separators=(',', ':'),allow_nan=False)+'\n')
                    trace.flush()
                def fit(samples, dimension, metadata):
                    return record_fit(samples, dimension, dict(scene=scene, **metadata))
                predictions, scene_runs[scene] = forecast(rows, initial, truth, fit, emit)
            with (directory/'scored_predictions.jsonl').open('x') as stream:
                for prediction in predictions:
                    stream.write(json.dumps(prediction,separators=(',', ':'),allow_nan=False)+'\n')
            scores[scene] = score(predictions)
            if (scene_runs[scene]['predictions'] != 150 or scene_runs[scene]['released_packets'] != 29
                    or scene_runs[scene]['final_history_rows'] != 280 or scene_runs[scene]['latest_used_deadline'] != 17700
                    or initial != initial_copy or not scene_runs[scene]['frozen_unchanged']):
                raise ValueError('Chronological panel, feedback budget or frozen state changed')
        if len(diagnostics) != 118:
            raise ValueError('The completed study must contain2 initial and116 subsequent fits')
        decision = gate(scores, True)
        result.update(status='factual_gate_pass' if decision['passed'] else 'closed_factual_gate_failed',
            technical_pass=True, gate=decision,
            initialization=dict(rows=135, positives=sum(y for _,y in initial['history']),
                nonzero_n=sum(inputs[0]>0 for inputs,y in initial['history']),
                full=initial['full'], hold_only=initial['hold_only']))
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
        result['gate'] = gate(scores, False)
    finally:
        fit_log.close()
        result.update(cases=scores, scene_runs=scene_runs,
            compute=dict(predictor_fits=len(diagnostics), converged_fits=sum(d['converged'] for d in diagnostics),
                initialization_fits=sum(d['phase']=='initialize' for d in diagnostics),
                refits=sum(d['phase']=='refit' for d in diagnostics),
                newton_steps=sum(d['newton_steps'] for d in diagnostics),
                line_search_evaluations=sum(d['line_search_evaluations'] for d in diagnostics),
                objective_evaluations=sum(d['objective_evaluations'] for d in diagnostics),
                max_final_gradient=max((d['gradient_max'] for d in diagnostics), default=None),
                max_newton_steps_per_fit=max((d['newton_steps'] for d in diagnostics), default=0),
                elapsed_seconds=time.monotonic()-started),
            limits=dict(new_native_seconds=0,new_collection=0,actor_critic_optimizer_updates=0,
                world_refits=0,new_control_actions=0,hyperparameter_sweeps=0),
            control_benefit_evidence=False, independent_confirmation=False, automatic_followup=False)
        write(out/'summary.json', result)
    raw = json.dumps(result,separators=(',', ':'),allow_nan=False)
    if len(raw.encode())+1>16000:
        raise RuntimeError('Summary exceeds the frozen small-transfer limit')
    print(raw, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thesis-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.thesis_root.resolve(), args.out.resolve())
    raise SystemExit(0 if result['technical_pass'] else 2)
