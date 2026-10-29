"""Audit the existing first suffix without changing its failed raw clock gate."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def trace_audit(actual_path, source_path, fork_sequence, nominal_action, horizon):
    count = 0
    checks = dict(full_prefix_exact=True, native_event_clock_exact=True,
                  raw_event_clock_exact=True, delivery_clock_exact=True)
    gaps = []
    with Path(actual_path).open() as actual, Path(source_path).open() as source:
        for line in actual:
            row = json.loads(line)
            expected_line = source.readline()
            if not expected_line:
                checks['full_prefix_exact'] = False
                break
            expected = json.loads(expected_line)
            compared = dict(row)
            if row['sequence'] == fork_sequence:
                compared['a'] = nominal_action
            checks['full_prefix_exact'] &= compared == expected
            raw_clock = row['raw_s'][0]
            # Match the original NumPy float32 canonicalization exactly.
            canonical_clock = (np.asarray([raw_clock], np.float32)/horizon)[0]
            checks['native_event_clock_exact'] &= bool(row['s'][0] == canonical_clock)
            checks['raw_event_clock_exact'] &= raw_clock == row['event'][2]
            checks['delivery_clock_exact'] &= row['started_at'] == row['issue_ledger']['time_seconds']
            gaps.append(row['started_at']-raw_clock)
            count += 1
    checks['prefix_issue_count_exact'] = count == fork_sequence+1
    return dict(checks=checks, issue_count=count,
                delivery_minus_event_seconds=dict(min=min(gaps), max=max(gaps)) if gaps else None)


def policy_source_audit(path):
    """The old checker reads clocks; actor gets the original state directly."""
    tree = ast.parse(Path(path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PhasePolicy')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__call__')
    call = next(node for node in ast.walk(method) if isinstance(node, ast.Call)
                and ast.unparse(node.func) == 'self.learner.act')
    state_writes = [node for node in ast.walk(method) if isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Store) and ast.unparse(node.value) == 'state']
    state_rebindings = [node for node in ast.walk(method) if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store) and node.id == 'state']
    return dict(actor_receives_original_state=ast.unparse(call.args[0]) == 'state[None]',
                no_state_clock_assignment=not state_writes and not state_rebindings,
                method_lines=[method.lineno, method.end_lineno])


def audit(root, source):
    root, source = Path(root), Path(source)
    spec = importlib.util.spec_from_file_location('v19_raw_analysis_for_clock_audit', root/'analyze.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    day, anchors, source_info = analysis.source_day(source)
    anchor = anchors['train/7X/late']
    protocol = read(root/'protocol.json')
    branch = root/protocol['intervention']['first_branch']['output_relative']
    value = read(branch/'summary.json')
    native = read(day/'training_branches'/str(anchor['index'])/'low'/'summary.json')
    original_checks = value['integrity']['checks']
    failed = [key for key, ok in original_checks.items() if ok is not True]
    prefix = trace_audit(branch/'prefix_trace.jsonl', day/'prefix_trace.jsonl',
                         anchor['selected']['sequence'], anchor['nominal_a'],
                         protocol['invariants']['horizon_seconds'])
    code = policy_source_audit(root/'phase_stream.py')
    checks = dict(source_contract=source_info['integrity']['all_pass'],
        only_incorrect_clock_gate_failed=value['status'] == 'integrity_failed' and failed == ['actor_real_clock'],
        original_native_complete=native['status'] == 'complete' and native['integrity']['all_pass'] is True,
        selected_and_predictions_exact=all(value[key] == native[key] == anchor[key]
            for key in ('selected', 'prefix', 'prediction', 'nominal_a', 'actions', 'features')),
        first_zero_exact=value['execution'] == native['execution'] ==
            dict(requested_hold_seconds=0., executed_hold_seconds=0.),
        fork_batch_other_controls_exact=value['native']['other_batch_actions'] == native['native']['other_batch_actions'],
        fixed_zero_exact=value['fixed_zero_prediction'] == dict(dt=0., wait=0., ride=0., cost=0.),
        suffix300=value['suffix']['seconds'] == value['truth']['fixed300']['dt'] == 300,
        suffix_frozen=bool(value['suffix']['training_delta']) and all(x == 0 for x in value['suffix']['training_delta'].values()),
        truth_not_fit=value['budget']['truth_labels_enter_training'] == value['budget']['new_target_training_labels'] == 0,
        actor_input_source=code['actor_receives_original_state'] and code['no_state_clock_assignment'],
        **prefix['checks'])
    failed_prefix = protocol['diagnostic_compute_budget']['failed_preflight_prefix']
    prior_failure = read(root/'branches/7X/low/failure.json')
    checks['failed_prefix_accounted'] = prior_failure['time'] == failed_prefix['native_seconds_replayed']
    return dict(status='complete', reusable=all(checks.values()),
        integrity=dict(all_pass=all(checks.values()), checks=checks),
        original_status=value['status'], original_failed_checks=failed,
        first_branch_relative=str(branch.relative_to(root)),
        prefix=prefix, source=source_info, actor_source_inspection=code,
        suffix_evidence=dict(per_issue_clock_records_saved=False,
            basis='Original PhasePolicy passes state[None] directly; PairedControlledStream constructs canonical state from the unmodified native event batch. The saved summary verifies the exact300 native steps, tail formula and zero training. This is source-path evidence, not a new per-issue suffix replay.',
            references=['phase_stream.py:22-33',
                'experiments/mcwm_closed_loop_response_residual_v11_20261021/paired_stream.py:120-126',
                'experiments/mcwm_closed_loop_response_residual_v11_20261021/paired_stream.py:80-81']),
        clock=dict(event='Original SUMO event detection clock used by actor.',
            delivery='Native passenger ledger issue clock; not substituted into actor state.',
            original_max_delivery_vs_actor_error_seconds=value['phase_intervention']['actor_clock_max_abs_error'],
            no_new_tolerance=True),
        budget=dict(first_branch_reused=True, new_rerun_outcomes=0,
            completed_diagnostic_outcomes=1, completed_suffix_seconds=300,
            prefix_replay_seconds=value['budget']['prefix_sim_seconds'],
            replayed_target_labels=value['budget']['replay_target_labels'],
            replayed_policy_updates=value['budget']['replay_policy_updates'],
            failed_prefix_replay=failed_prefix,
            new_training_labels=0, remaining_new_outcomes=5, total_outcome_limit=6,
            total_suffix_seconds_limit=1800),
        raw_results_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.root, args.source)
    data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(data.encode()) >= 8000:
        raise RuntimeError('Reduce the resident clock audit before export')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        output.write(data)
    print(json.dumps(dict(status=value['status'], reusable=value['reusable'], bytes=len(data.encode()))))


if __name__ == '__main__':
    main()
