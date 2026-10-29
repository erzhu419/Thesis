"""Choose training-period branches by clock, admitted issue identity, and action support."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import ControlledCalibration

STATE_KEYS = ('sequence', 'started_at', 'line_id', 'vehicle_id', 'trip_id', 'station_id',
              's', 'raw_s', 'event', 'issue_ledger')


def read_json(path):
    return json.loads(Path(path).read_text())


def cells(protocol, *, preflight=False):
    settings = protocol['preflight'] if preflight else protocol
    seed = settings['seed'] if preflight else protocol['paired_seed']
    scenes = [settings['scenario']] if preflight else protocol['scenarios']
    lines = [settings['target_line']] if preflight else protocol['target_lines']
    for scene, line, (index, window) in product(scenes, lines, enumerate(settings['training_windows'])):
        yield dict(seed=int(seed), scenario=scene, target_line=line,
                   window_index=index, window_name=window['name']), dict(window)


def issue_candidates(summary, cell, window):
    """Do not read prediction errors or endpoint outcomes to rank issues."""
    admitted = {int(sequence) for receipt in summary['receipts']
                for sequence in receipt['calibration_sequence_ids']}
    return sorted((row for row in summary['evaluations']
        if row['role'] == 'train' and int(row['sequence']) in admitted
        and str(row['line_id']) == cell['target_line']
        and window['start'] <= float(row['issued_at']) < window['end']),
        key=lambda row: (float(row['issued_at']), int(row['sequence'])))


def executable_actions(trace, transform):
    # Reuse the real inversion rule without constructing/consuming any assignment RNG.
    calibration = ControlledCalibration.__new__(ControlledCalibration)
    calibration.transform = transform
    result = {}
    for name, hold in (('low', 0.), ('high', 7.5)):
        choice = calibration._action_for_hold(np.asarray(trace['s'], np.float32), hold)
        if choice is None:
            return None
        _proposal, action, actual = choice
        result[name] = dict(action=action.tolist(), hold_seconds=float(actual))
    return result


def source_metadata(summary, relative, cell, protocol, *, preflight=False):
    settings = dict(protocol)
    if preflight:
        settings.update(protocol['preflight'])
    assert summary['status'] == 'complete' and summary['integrity']['all_pass'] is True
    assert summary['cell']['seed'] == cell['seed']
    assert summary['cell']['scenario'] == cell['scenario']
    assert summary['initializer']['initial_updates'] == protocol['initializer_updates']
    assert summary['budget']['target_labels'] == settings['target_record_budget']
    assert summary['budget']['policy_updates'] == settings['policy_update_budget']
    expected_horizon = settings['preflight_stop_seconds'] if preflight else settings['horizon_seconds']
    if preflight:
        assert summary['budget']['native_seconds'] >= expected_horizon
    else:
        assert summary['budget']['native_seconds'] == expected_horizon
    return dict(day_relative=str(relative), external_seed=cell['seed']+protocol['target_seed_offset'],
        initializer_updates=summary['initializer']['initial_updates'],
        horizon_seconds=protocol['horizon_seconds'], full_day_budget=summary['budget'])


def choose_state(summary, trace_rows, cell, window, source, transform, protocol):
    """Select the first executable admitted training issue; preserve an empty cell."""
    base = dict(state_id=f"{cell['scenario']}/{cell['target_line']}/{window['name']}",
        cell=cell, window=window, status='selection_missing', selected=None,
        nominal_a=None, proposed_a=None, prediction=None, source_truth=None,
        nominal_hold_seconds=None, executable=None, source=dict(source),
        candidates_considered=0)
    for row in issue_candidates(summary, cell, window):
        base['candidates_considered'] += 1
        trace = trace_rows[int(row['sequence'])]
        if trace['line_id'] != cell['target_line'] or trace['started_at'] != row['issued_at']:
            raise RuntimeError('Source evaluation and native trace issue disagree')
        actions = executable_actions(trace, transform)
        if actions is None:
            continue
        # Prior packet processing can occur at this same native clock before the
        # current issue callback. Its exact list is checked again during replay.
        prior = [receipt for receipt in summary['receipts']
                 if receipt['processed_at'] <= float(row['issued_at'])]
        selected_ids = [int(sequence) for receipt in prior
                        for sequence in receipt['calibration_sequence_ids']]
        assert int(row['sequence']) not in selected_ids
        prefix = dict(prefix_receipts=prior, prefix_packets=len(prior),
            prefix_target_labels=sum(receipt['records'] for receipt in prior),
            prefix_policy_updates=len(prior)*protocol['policy_updates_per_packet'],
            selected_calibration_sequence_ids_before_issue=selected_ids)
        base.update(status='selected', selected={key: trace[key] for key in STATE_KEYS},
            nominal_a=trace['a'], proposed_a=trace['proposed_a'],
            nominal_hold_seconds=float(row['hold_seconds']),
            prediction=row['prediction'], source_truth=row['truth'], executable=actions)
        base['source'].update(prefix)
        return base
    return base


def build_manifest(root, protocol, transform, *, preflight=False):
    source_root = (Path(root)/protocol['reference_v14_root']).resolve()
    cached = {}
    rows = []
    for cell, window in cells(protocol, preflight=preflight):
        relative = (Path(protocol['reference_v14_preflight_directory'])/'day' if preflight
                    else Path('runs')/cell['scenario']/f"seed{cell['seed']}")
        key = str(relative)
        if key not in cached:
            day = source_root/relative
            summary = read_json(day/'summary.json')
            wanted = {int(row['sequence']) for row in summary['evaluations'] if row['role'] == 'train'}
            trace = {}
            # Stream the resident trace once; no trajectory is copied locally.
            with (day/'prefix_trace.jsonl').open() as stream:
                for line in stream:
                    record = json.loads(line)
                    if int(record['sequence']) in wanted:
                        trace[int(record['sequence'])] = record
            cached[key] = summary, trace
        summary, trace = cached[key]
        source = source_metadata(summary, relative, cell, protocol, preflight=preflight)
        rows.append(choose_state(summary, trace, cell, window, source, transform, protocol))
    complete = all(row['status'] == 'selected' for row in rows)
    return dict(status='complete', selection_status='all_selected' if complete else 'states_missing',
        mode='preflight' if preflight else 'training_period_diagnostic',
        source_root=str(source_root),
        selection_rule='First budget-admitted training issue in each fixed clock/line cell with jointly executable0/7.5. Outcomes and errors do not choose or rank states.',
        states=rows)


def compact_summary(value):
    rows = []
    for row in value['states']:
        selected, source = row['selected'], row['source']
        rows.append(dict(state_id=row['state_id'], cell=row['cell'], window=row['window'],
            status=row['status'], candidates_considered=row['candidates_considered'],
            sequence=selected['sequence'] if selected else None,
            issued_at=selected['started_at'] if selected else None,
            nominal_hold_seconds=row['nominal_hold_seconds'],
            executable_holds={key: arm['hold_seconds'] for key, arm in row['executable'].items()}
                if row['executable'] else None,
            external_seed=source['external_seed'], day_relative=source['day_relative'],
            prefix_packets=source.get('prefix_packets'),
            prefix_target_labels=source.get('prefix_target_labels'),
            prefix_policy_updates=source.get('prefix_policy_updates')))
    return dict(status=value['status'], mode=value['mode'], expected_states=len(rows),
        coverage=dict(expected_states=len(rows), selected_states=sum(row['status']=='selected' for row in rows),
            missing_states=[row['state_id'] for row in rows if row['status']!='selected']),
        selected_states=sum(row['status'] == 'selected' for row in rows),
        missing_states=[row['state_id'] for row in rows if row['status'] != 'selected'],
        suffix_seconds_per_branch=300, expected_branches=2*len(rows),
        maximum_new_suffix_seconds=600*len(rows), paired_labels_enter_fit=0,
        prefix_receipt_clock_rule='processed_at<=issued_at; same-clock receipts are before the next issue callback and checked by native replay',
        pre_shock_scope='The pre-shock scene pairs share the same external seed and end before the shock; repeated scene labels do not add independent physical states.',
        states=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    try:
        protocol = read_json(HERE/'protocol.json')
        metadata = read_json((HERE/protocol['reference_inputs']).resolve()/'history_metadata.json')
        manifest = build_manifest(HERE, protocol, ExecutionAction(metadata['contract']), preflight=args.preflight)
        summary = compact_summary(manifest)
        payload = json.dumps(summary, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
        if len(payload.encode()) > 20000:
            raise RuntimeError('Reduce the state-selection summary before collection')
        for output in (args.out, args.summary):
            if output.exists():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
        args.summary.write_text(payload)
        print(json.dumps(dict(status=summary['status'], selected_states=summary['selected_states'],
                              expected_states=summary['expected_states'], summary_bytes=len(payload.encode()))), flush=True)
    except Exception as error:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        (args.summary.parent/'selection_failure.json').write_text(json.dumps(dict(
            type=type(error).__name__, message=str(error)), ensure_ascii=False)+'\n')
        raise


if __name__ == '__main__':
    main()
