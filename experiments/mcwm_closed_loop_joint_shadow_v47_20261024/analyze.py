"""Resident matched-score aggregation; download only the bounded final JSON."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.summary import (
    LINES, METRICS, POINT_COLUMNS, STAGES, STAT_COLUMNS, compact, encode_bounded,
    summarize_scores,
)


def _read_scores(path):
    with Path(path).open() as handle:
        return [json.loads(text) for text in handle if text.strip()]


def aggregate(workers, score_records, protocol):
    """workers/scoring dictionaries are indexed by scenario then M1/M8."""
    failed = [f'{scenario}/{arm}' for scenario in protocol['scenarios']
        for arm in protocol['arms'] if workers[scenario][arm].get('status') != 'complete'
        or not workers[scenario][arm].get('proof', {}).get('all_pass')]
    if failed:
        return dict(status='failed', priority='high',
            integrity=dict(all_pass=False, failed_workers=failed),
            cases={scenario: dict(workers=workers[scenario]) for scenario in protocol['scenarios']},
            compute_accounting=dict(completed_worker_updates=sum(
                worker.get('budget', {}).get('optimizer_updates', 0)
                for scenario in workers.values() for worker in scenario.values()
                if worker.get('status') == 'complete'),
                failed_worker_spend='not_inferred_from_missing_summary'),
            score_aggregation_skipped=True)
    cases, checks = {}, {}
    for scenario in protocol['scenarios']:
        supplied = workers[scenario]
        if any(supplied[arm]['status'] != 'complete' or not supplied[arm]['proof']['all_pass']
               for arm in protocol['arms']):
            raise ValueError('All four source workers must complete their declared audits')
        pair = summarize_scores(score_records[scenario]['M1'], score_records[scenario]['M8'], protocol)
        m1, m8 = supplied['M1']['wall'], supplied['M8']['wall']
        overlap = min(m1['training_finished'], m8['training_finished'])-max(
            m1['training_started'], m8['training_started'])
        checks[scenario] = dict(**pair['checks'], workers_all_pass=True,
                               actual_training_overlap=overlap > 0.)
        cases[scenario] = dict(**pair,
            actual_training_overlap_seconds=max(0., overlap),
            workers={arm: supplied[arm] for arm in protocol['arms']})
    complete = all(all(value.values()) for value in checks.values())
    result = dict(status='complete' if complete else 'failed', priority='high',
        integrity=dict(all_pass=complete, checks=checks),
        labels=dict(lines=list(LINES), stages=list(STAGES), metrics=list(METRICS),
            candidate_proposal_seconds=protocol['candidate_holds_seconds']),
        columns=dict(points=POINT_COLUMNS, statistics=STAT_COLUMNS,
            persistence=['line', 'packets', 'best_executed_disagreement_packets',
                'actor_proposal_F_alias_packets', 'actor_executed_difference_packets'],
            bank=['line', 'original_issue_clock', 'canonical_clock', 'memory_observed_at',
                  'physical_state_group', 'candidate_executed_seconds', 'candidate_alias_groups']),
        cases=cases,
        preserved_contract=dict(initializer_updates=protocol['initializer_updates'],
            training_seed=protocol['training_seed'], slots_per_arm=285, updates_per_arm=456,
            source_horizon_seconds=24000, lambda_trace=0, F_unchanged=True, trust_unchanged=True),
        new_budget=dict(formal_shadow_updates=4*456, separate_preflight_shadow_updates=456,
            new_feedback_slots=0, native_branches=0, native_seconds=0, checkpoints_downloaded=0),
        interpretation=dict(reference='M8 minusM1; advantages use each critic score atF(proposal0)',
            states='six labeled roots are three paired physical issues per scenario; repeated snapshots are not independent states',
            clocks='original issue clocks are fixed; response32-456 updates share9630 admission clock',
            persistence='exact nonzero counts require signed and absolute magnitudes for interpretation',
            endpoint='target variance and cumulative value/action diagnostics, no300s direction or actual cost claim'),
        decision=dict(automatic_native_branches=False,
            next_gate='Review whether estimator noise reduction propagates beyond value level to common-action advantages and F holding.'))
    return result


def _worker_brief(worker):
    if worker.get('status') != 'complete':
        return worker
    proof = worker['proof']
    checks = proof['checks']
    certificates = proof.get('source_certifications', [])
    return dict(status=worker['status'], arm=worker['arm'], phase=worker['phase'],
        proof=dict(all_pass=proof['all_pass'], checks_passed=sum(bool(value) for value in checks.values()),
            checks_total=len(checks), failed_checks=[name for name, value in checks.items() if not value],
            audit_counts=proof.get('audit_counts', {}),
            source_certifications=[dict(updates=item['updates'], all_pass=item['all_pass'])
                                   for item in certificates],
            boundaries=proof.get('boundaries', {})),
        budget=worker['budget'], extra_compute=worker['extra_compute'],
        actual_target_path=worker['actual_target_path'], wall=worker['wall'])


def compact_final(result, protocol):
    """Keep full aggregation resident; omit repetitive worker paths/check names."""
    value = {name: item for name, item in result.items() if name != 'cases'}
    value['cases'] = {scenario: {**{name: item for name, item in case.items() if name != 'workers'},
        'workers': {arm: _worker_brief(worker) for arm, worker in case['workers'].items()}}
        for scenario, case in result['cases'].items()}
    value = compact(value)
    for scenario, case in result['cases'].items():
        for arm, worker in case['workers'].items():
            if worker.get('status') == 'complete':
                value['cases'][scenario]['workers'][arm]['wall'] = worker['wall']
    encode_bounded(value, protocol['summary_max_bytes'])
    return value


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    out = HERE/'analysis'
    if (out/'summary.json').exists():
        completed = json.loads((out/'summary.json').read_text())
        print(json.dumps(dict(status=completed['status'], reused_existing_summary=True)), flush=True)
        return
    out.mkdir(exist_ok=False)
    try:
        workers, scores = {}, {}
        for scenario in protocol['scenarios']:
            workers[scenario], scores[scenario] = {}, {}
            for arm in protocol['arms']:
                root = HERE/'runs'/scenario/arm
                source = root/'summary.json'
                worker = json.loads((source if source.exists() else root/'failure.json').read_text())
                workers[scenario][arm] = worker
                if worker.get('status') == 'complete' and worker.get('proof', {}).get('all_pass'):
                    path = Path(worker['paths']['scores'])
                    scores[scenario][arm] = _read_scores(path if path.is_absolute() else root/path)
        result = aggregate(workers, scores, protocol)
        with (out/'result_stats.json').open('x') as handle:
            json.dump(result, handle, separators=(',', ':'), allow_nan=False)
            handle.write('\n')
        result = compact_final(result, protocol)
    except Exception as error:
        result = dict(status='failed', priority='high', integrity=dict(all_pass=False),
            failure=dict(type=type(error).__name__, message=str(error)),
            resident_analysis='analysis/result_stats.json if aggregation completed',
            no_rescoring_or_retraining=True)
        with (out/'failure.json').open('x') as handle:
            json.dump(result, handle, separators=(',', ':'), allow_nan=False)
            handle.write('\n')
    encoded = encode_bounded(result, protocol['summary_max_bytes'])
    with (out/'summary.json').open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status=result['status'], integrity=result['integrity'],
                         bytes=len(encoded.encode()))), flush=True)


if __name__ == '__main__':
    main()
