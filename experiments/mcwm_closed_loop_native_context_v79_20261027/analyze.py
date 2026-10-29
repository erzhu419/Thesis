"""Summarize the two fixed native-visibility runs without reading resident traces."""
import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEEDS = (6208, 6207)
GROUPS = ('onboard_positive', 'nonempty_queue', 'release_positive')
RUNTIME_CHECKS = (
    'exact_native_limit', 'original_horizon', 'no_terminal_fabrication',
    'frozen_initializer', 'no_optimizer_updates', 'native_modules_and_Adam_unchanged',
    'learner_objects_retained', 'no_admitted_feedback', 'no_private_outcome_draws',
    'source_learning_config_unchanged', 'actor_uses_real_clock',
    'one_original_execution_transform', 'issue_count_exact', 'observation_integrity',
)
ZERO_BUDGET = ('training_updates', 'feedback_slots', 'covariate_slots',
               'complete_outcome_slots', 'initializer_updates', 'counterfactual_branches')
SOURCE_CONTRACT = dict(covariate_slots=285, fresh_complete_outcomes=285,
    updates_per_family=456, horizon_seconds=24000, F_unchanged=True,
    trust_unchanged=True, policy_sampler_unchanged=True, learning_budget_consumed=False)


def _read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def _booleans_pass(checks):
    return bool(checks) and all(value is True for value in checks.values())


def _coverage(observation):
    if observation is None:
        return None
    coverage = observation.get('coverage', {})
    captures = observation.get('captures', {})
    groups = coverage.get('subgroups', {})
    unknown = coverage.get('membership_unknown')
    group_checks = {}
    for name in GROUPS:
        row = groups.get(name)
        group_checks[name] = (row is not None and row.get('total', 0) > 0
            and row.get('supported') == row['total']
            and row.get('observed') == row['total'])
    unknown_clear = (unknown is not None and all(unknown.get(name) == 0 for name in GROUPS))
    total, supported, observed = (captures.get(name) for name in ('total', 'supported', 'observed'))
    # The observer already excludes terminal applies from total captures.
    all_nonterminal_observed = (all(value is not None for value in (total, supported, observed))
        and total > 0 and supported == observed == total)
    return dict(captures=captures, overall=coverage.get('overall'),
        by_line=coverage.get('by_line'), subgroups=groups,
        membership_unknown=unknown, required_group_checks=group_checks,
        unknown_memberships_clear=unknown_clear,
        all_nonterminal_observed=all_nonterminal_observed,
        native_joint_supported=(all(group_checks.values()) and unknown_clear
                                and all_nonterminal_observed),
        reasons=observation.get('reasons'),
        stage_semantics=observation.get('stage_semantics'),
        next_stop_plan=observation.get('next_stop_plan'),
        checks=observation.get('checks'), all_pass=observation.get('all_pass'),
        apply_audit=observation.get('apply_audit'),
        clocks=observation.get('clocks'),
        examples=None if observation.get('examples') is None else observation['examples'][:2],
        observation_window=observation.get('window'), right_open=observation.get('right_open'))


def _cell(seed, summary, failure, protocol):
    if summary is None:
        return dict(seed=seed, status='missing_summary', runtime_pass=False,
            native_coverage=None, checks={}, failed_checks=['summary_missing'],
            actual_simulated_seconds=None if failure is None else failure.get('actual_simulated_seconds'),
            failure=failure)
    integrity = summary.get('integrity', {})
    native = summary.get('native_observation')
    native_checks = {} if native is None else native.get('checks', {})
    budget = summary.get('budget', {})
    checks = dict(status_complete=summary.get('status') == 'complete',
        correct_identity=summary.get('cell', {}).get('seed') == seed
            and summary.get('cell', {}).get('scenario') == 'stable'
            and summary.get('cell', {}).get('external_seed') == seed+protocol['external_seed_offset'],
        integrity_pass=integrity.get('all_pass') is True,
        runtime_checks_complete=all(integrity.get('checks', {}).get(name) is True
                                    for name in RUNTIME_CHECKS),
        runtime_checks_all_true=_booleans_pass(integrity.get('checks', {})),
        native_audits_all_true=native is not None and native.get('all_pass') is True
            and _booleans_pass(native_checks),
        actual_stop_exact=budget.get('actual_simulated_seconds') == protocol['native_stop_seconds'],
        actual_endpoint_exact=summary.get('endpoint', {}).get('time_seconds') == protocol['native_stop_seconds'],
        no_new_learning_or_labels=all(budget.get(name) == 0 for name in ZERO_BUDGET),
        initializer_reused=summary.get('initializer', {}).get('updates') == 50000,
        original_learning_contract_unchanged=summary.get('original_learning_contract') == SOURCE_CONTRACT,
        original_actor_clock=summary.get('control', {}).get('actor_real_clock') is True,
        original_F=summary.get('control', {}).get('shared_execution_rule_exact') is True,
        correct_observation_window=native is not None
            and native.get('window') == protocol['observation_window']
            and native.get('right_open') is True)
    return dict(seed=seed, status=summary.get('status'), runtime_pass=all(checks.values()),
        native_coverage=_coverage(native), checks=checks,
        failed_checks=[key for key, value in checks.items() if not value],
        actual_simulated_seconds=budget.get('actual_simulated_seconds'),
        budget=budget, initializer=summary.get('initializer'),
        original_learning_contract=summary.get('original_learning_contract'),
        runtime_integrity=integrity, control=summary.get('control'),
        endpoint=summary.get('endpoint'), discarded_feedback=summary.get('discarded_feedback'),
        execution_wall=summary.get('execution_wall'), wall_seconds=summary.get('wall_seconds'))


def analyze(experiment_dir=HERE):
    root = Path(experiment_dir)
    protocol = _read(root/'protocol.json')
    cells, failures = [], []
    for seed in SEEDS:
        directory = root/'runs'/f'seed{seed}'
        summary, failure = _read(directory/'summary.json'), _read(directory/'failure.json')
        if failure is not None:
            failures.append(dict(seed=seed, evidence=failure))
        cells.append(_cell(seed, summary, failure, protocol))
    complete = all(cell['status'] != 'missing_summary' for cell in cells)
    seconds = [cell['actual_simulated_seconds'] for cell in cells]
    actual_seconds = sum(seconds) if all(value is not None for value in seconds) else None
    walls = [cell.get('execution_wall') for cell in cells]
    overlap = None
    if all(wall is not None and wall.get('native_started_wall') is not None
           and wall.get('native_finished_wall') is not None for wall in walls):
        overlap = max(0., min(wall['native_finished_wall'] for wall in walls)
                      - max(wall['native_started_wall'] for wall in walls))
    protocol_checks = dict(fixed_seeds=protocol.get('seeds') == list(SEEDS),
        original_horizon=protocol.get('horizon_seconds') == 24000,
        fixed_diagnostic_stop=protocol.get('native_stop_seconds') == 16200,
        fixed_diagnostic_budget=protocol.get('maximum_native_seconds') == 32400,
        no_new_initializer=protocol.get('new_initialization_updates') == 0,
        no_new_training=protocol.get('new_training_updates') == 0,
        no_new_feedback=protocol.get('new_feedback_slots') == 0,
        original_source_contract=protocol.get('source_contract') == dict(
            covariates=285, fresh_complete_outcomes=285, updates_per_family=456),
        source_budget_unconsumed=protocol.get('source_budget_consumed') == dict(
            covariates=0, complete_outcomes=0, updates_per_family=0))
    runtime_pass = (complete and all(cell['runtime_pass'] for cell in cells)
                    and all(protocol_checks.values()) and actual_seconds == 32400
                    and overlap is not None and overlap > 0)
    native_pass = (runtime_pass and all(cell['native_coverage']['native_joint_supported']
                                      for cell in cells))
    reasons = []
    if not complete:
        reasons.append('Two fixed native summaries are not both present; missing work is not zero.')
    if complete and not runtime_pass:
        reasons.append('Runtime, original contract, exact diagnostic budget or actual parallelism failed.')
    if runtime_pass and not native_pass:
        reasons.append('Native observation finished, but required subgroup existence or complete '
                       'joint support is absent. This is a diagnostic outcome, not failed simulation.')
    if native_pass:
        reasons.append('Both seeds support every required native subgroup; only budget-matched '
                       'collection design may proceed.')
    return dict(version='v79', status=('incomplete' if not complete else
        'complete' if runtime_pass else 'integrity_failed'), complete=complete,
        runtime_integrity=dict(all_pass=runtime_pass, protocol_checks=protocol_checks,
            actual_parallel_overlap_seconds=overlap),
        cells=cells, budget=dict(planned_prefixes=2, documented_prefixes=sum(
            cell['status'] != 'missing_summary' for cell in cells),
            maximum_simulated_seconds=32400, actual_simulated_seconds=actual_seconds,
            documented_simulated_seconds=sum(value for value in seconds if value is not None),
            new_initialization_updates=0 if runtime_pass else None,
            new_training_updates=0 if runtime_pass else None,
            new_feedback_slots=0 if runtime_pass else None,
            source_285_covariates_285_outcomes_456_updates_consumed=False if runtime_pass else None,
            failed_work=failures),
        decision=dict(native_joint_coverage_verified=native_pass,
            may_design_budget_matched_collection=native_pass,
            new_collection_started=False, training_authorized=False,
            control_improvement_evidence=False, additional_seeds_authorized=False,
            reasons=reasons),
        interpretation='New frozen-initializer visibility prefixes, not replay of v75 learning '
            'or a fresh/no_fresh control comparison. Episode horizon remains24000; the diagnostic '
            'stops natively at16200 without terminal labels.',
        limitations=['Native getStops current-only or missing next-stop plans remain independently '
            'unscorable; only a scorable next-stop mismatch is a failure.',
            'Empty per-line subgroups are reported; the three required groups must exist in '
            'each seed without manufactured excitation.',
            'Visibility and unchanged-model audits establish no passenger-cost benefit.'],
        old_code_and_results_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    raw = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
    if len(raw.encode()) > 48000:
        raise ValueError('Combined native-visibility summary exceeds48KB')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=result['status'], bytes=len(raw.encode()),
        native_joint_coverage_verified=result['decision']['native_joint_coverage_verified'])))


if __name__ == '__main__':
    main()
