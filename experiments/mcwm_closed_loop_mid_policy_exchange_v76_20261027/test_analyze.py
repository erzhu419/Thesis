"""Only exact packet51 entries support the frozen-policy comparison."""
import json

import pytest

from experiments.mcwm_closed_loop_mid_policy_exchange_v76_20261027 import analyze as mod


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def mutate(path, callback):
    value = json.loads(path.read_text())
    callback(value)
    write(path, value)


def snapshot(seed, arm):
    return dict(seed=seed, source_arm=arm, packets=51, counts=mod.COUNTS,
        covariates=255, complete_outcomes=255 if arm == 'fresh' else 0,
        published_clock=16203., publication_deadline=16200, data_deadline=16200,
        max_available_at=16200., processed_at=16203., future_training_used=False)


def fixture(root, *, bad=(12000., 10000., 10500., 10000.), good=(8000., 10000., 9000., 10000.), positive_prefix_disadvantage=False):
    for seed in mod.SEEDS:
        waits = bad if seed == 6208 else good
        for entry_index, entry in enumerate(mod.ENTRIES):
            base = root/'runs'/f'seed{seed}'/entry
            clock = 16205.+2*entry_index
            prefix_wait = 100000.+(-1000 if positive_prefix_disadvantage and seed == 6207 else 1000)*entry_index
            cut = dict(time_seconds=clock, accrued_wait_seconds=prefix_wait,
                accrued_in_vehicle_seconds=200000., accrued_generalized_passenger_seconds=2*prefix_wait+200000.,
                waiting_passengers=330, in_vehicle_passengers=50)
            initial = dict(path=f'/resident/v75/inputs/h2oplus_seed{seed}.pt', initial_updates=50000,
                history_rows=32768, source_rows=4096)
            budget = dict(restoration_feedback_slots=255,
                restoration_complete_outcomes=255 if entry == 'fresh' else 0,
                restoration_packets=51, native_prefix_seconds=clock)
            budget.update({'restoration_'+key+'_updates': 408 for key in ('critic', 'value', 'actor', 'alpha')})
            prefix = dict(status='complete', cell=dict(seed=seed, scenario='stable', entry_arm=entry),
                integrity=dict(all_pass=True, checks=dict(native=True)), budget=budget,
                prefix_clock=clock, trace_rows=10000, matched_prefix_rows=10000,
                cut_snapshot=cut, initializer=initial, external_seed=seed+100000,
                actor_snapshot=snapshot(seed, entry))
            write(base/'prefix_summary.json', prefix)
            wall = {'low': dict(native_started_wall=10., native_finished_wall=30.),
                    'high': dict(native_started_wall=11., native_finished_wall=32.)}
            live = dict(all_pass=True, native_before=clock, native_after=clock,
                counters_before=mod.COUNTS, counters_after=mod.COUNTS,
                parent_rng_unchanged=True, parent_trace_cookie_unchanged=True,
                route_streams=dict(parent_offsets_unchanged=True,
                    parent_offsets_before={'route': 1}, parent_offsets_after={'route': 1}),
                overlap_seconds=19., children=wall)
            write(base/'suffixes/live_fork.json', live)
            for policy_index, (branch, policy) in enumerate(mod.BRANCHES.items()):
                w, r = waits[entry_index*2+policy_index], 30000.
                points = {}
                for time in (18300, 20100, 24000):
                    fraction = (time-clock)/(24000-clock)
                    wait, ride = cut['accrued_wait_seconds']+w*fraction, cut['accrued_in_vehicle_seconds']+r*fraction
                    points[str(time)] = dict(time_seconds=time, accrued_wait_seconds=wait,
                        accrued_in_vehicle_seconds=ride, accrued_generalized_passenger_seconds=2*wait+ride,
                        waiting_passengers=335 if policy == 'fresh' else 338, in_vehicle_passengers=0)
                windows = {}
                for period in mod.PERIODS:
                    lo = cut if period == 'remaining' else points['18300' if period == 'tail' else '20100']
                    hi = points['24000']
                    wait, ride = hi['accrued_wait_seconds']-lo['accrued_wait_seconds'], hi['accrued_in_vehicle_seconds']-lo['accrued_in_vehicle_seconds']
                    windows[period] = dict(wait=wait, ride=ride, cost=2*wait+ride,
                        denominator=15079, cost_per_passenger=(2*wait+ride)/15079)
                row = dict(status='complete', cell=dict(seed=seed, scenario='stable', entry_arm=entry, policy_arm=policy),
                    integrity=dict(all_pass=True, checks=dict(native=True)), switch_time=clock,
                    cut_snapshot=cut, endpoints=points, cost_windows=windows,
                    initializer=initial, external_seed=seed+100000, actor_snapshot=snapshot(seed, policy),
                    final_waiting=points['24000']['waiting_passengers'], zero_suffix_training=True,
                    post_budget_counts=dict(before=mod.COUNTS, after=mod.COUNTS),
                    execution_wall=wall[branch], budget=dict(suffix_native_seconds=24000-clock),
                    diagonal_reference=dict(requires_original_suffix_exact=False,
                        reason='Frozen408 versus original456; different original continuation is expected'),
                    first_issue=dict(clock_seconds=clock, executed_a=[-1.], holding_seconds=0.))
                final = points['24000']
                row['cost'] = dict(departed_passengers=15079,
                    accrued_wait_seconds=final['accrued_wait_seconds'],
                    accrued_in_vehicle_seconds=final['accrued_in_vehicle_seconds'],
                    generalized_passenger_seconds=final['accrued_generalized_passenger_seconds'],
                    generalized_seconds_per_departed_passenger=final['accrued_generalized_passenger_seconds']/15079)
                write(base/'suffixes'/branch/'summary.json', row)


def suffix(root, seed=6208, entry='fresh', branch='low'):
    return root/'runs'/f'seed{seed}'/entry/'suffixes'/branch/'summary.json'


def test_same_entry_costs_score_actual_cut_and_preserve_positive_case(tmp_path):
    fixture(tmp_path)
    result, detail = mod.analyze(tmp_path)
    assert result['status'] == 'complete'
    bad, good = result['seeds']
    assert bad['comparisons']['fresh']['switch_time'] == 16205
    assert bad['comparisons']['no_fresh']['switch_time'] == 16207
    assert bad['comparisons']['fresh']['policy_fresh_minus_no_fresh']['remaining']['cost'] == 4000
    assert bad['comparisons']['no_fresh']['policy_fresh_minus_no_fresh']['remaining']['cost'] == 1000
    assert good['comparisons']['fresh']['policy_fresh_minus_no_fresh']['remaining']['cost'] == -4000
    assert result['decision']['baseline_protection_may_be_considered'] is True
    assert result['budget']['actual_total_native_seconds'] == 127176
    assert result['budget']['restoration_updates_per_family']['actor'] == 1632
    assert result['budget']['restoration_covariate_slots'] == 1020
    assert result['budget']['restoration_complete_fresh_outcomes'] == 510
    assert detail['entries'][0]['fork_overlap_seconds'] == 19


@pytest.mark.parametrize('kind', ['future_actor', 'wrong_snapshot_steps', 'suffix_training', 'cost', 'prefix_restore', 'no_overlap'])
def test_real_contract_failures_block_mechanistic_interpretation(tmp_path, kind):
    fixture(tmp_path)
    if kind == 'future_actor':
        mutate(suffix(tmp_path), lambda r: r['actor_snapshot'].update(future_training_used=True))
    elif kind == 'wrong_snapshot_steps':
        mutate(suffix(tmp_path), lambda r: r['actor_snapshot'].update(packets=57))
    elif kind == 'suffix_training':
        mutate(suffix(tmp_path), lambda r: r['post_budget_counts']['after'].update(actor=409))
    elif kind == 'cost':
        mutate(suffix(tmp_path), lambda r: r['cost_windows']['remaining'].update(denominator=200))
    elif kind == 'prefix_restore':
        mutate(tmp_path/'runs/seed6208/fresh/prefix_summary.json', lambda r: r['budget'].update(restoration_critic_updates=407))
    else:
        mutate(tmp_path/'runs/seed6208/fresh/suffixes/live_fork.json', lambda r: r['children']['high'].update(native_started_wall=31))
    result, _ = mod.analyze(tmp_path)
    assert not result['integrity']['all_pass']
    assert result['decision']['baseline_protection_may_be_considered'] is False
    assert result['seeds'][0]['comparisons'] is None


def test_frozen_diagonal_does_not_have_to_replay_v75_future_updates(tmp_path):
    fixture(tmp_path)
    mutate(suffix(tmp_path), lambda r: r['diagonal_reference'].update(ledger_exact=False, full_remaining_trace_exact=False))
    result, _ = mod.analyze(tmp_path)
    assert result['status'] == 'complete'
    assert result['decision']['original_v75_continuation_causal_decomposition'] is False


def test_missing_branch_remains_unknown_and_failed_work_is_retained(tmp_path):
    fixture(tmp_path)
    path = suffix(tmp_path)
    path.unlink()
    write(path.with_name('failure.json'), dict(native_time=20000, consumed_budget=dict(policy_updates=408)))
    result, _ = mod.analyze(tmp_path)
    assert result['complete'] is False and result['missing'] and result['failures']
    assert result['budget']['actual_total_native_seconds'] is None
    assert len(result['cells']) == 7
    assert result['seeds'][0]['comparisons'] is None


def test_first_action_F_alias_does_not_cancel_whole_policy_comparison(tmp_path):
    fixture(tmp_path)
    result, _ = mod.analyze(tmp_path)
    assert result['status'] == 'complete'
    assert result['seeds'][0]['comparisons']['fresh']['first_action_executed_equal'] is True


@pytest.mark.parametrize('rescued,prefix_disadvantage,reason', [
    (False, False, 'no_consistent_bad_case_rescue'),
    (True, True, 'positive_case_prefix_gain_not_preserved')])
def test_no_rescue_or_lost_prefix_gain_stops_without_new_algorithm(tmp_path, rescued, prefix_disadvantage, reason):
    bad = (12000., 10000., 10500., 10000.) if rescued else (8000., 10000., 9000., 10000.)
    fixture(tmp_path, bad=bad, positive_prefix_disadvantage=prefix_disadvantage)
    result, _ = mod.analyze(tmp_path)
    assert result['status'] == 'complete'
    assert result['decision']['reason'] == reason
    assert result['decision']['baseline_protection_may_be_considered'] is False
    assert result['decision']['new_training_authorized_by_diagnostic'] is False
    assert result['decision']['automatic_seed_expansion'] is False


def test_positive_prefix_gain_can_survive_even_if_future_fresh_actor_is_worse(tmp_path):
    fixture(tmp_path, good=(12000., 10000., 10500., 10000.))
    result, _ = mod.analyze(tmp_path)
    choice = result['decision']
    assert choice['baseline_protection_may_be_considered'] is True
    assert choice['positive_case_prefix_gain_preserved_after_nofresh_handoff'] is True
    assert choice['positive_case_remaining_fresh_actor_not_worse_by_entry'] == dict(fresh=False, no_fresh=False)
    assert choice['positive_case_full_day_difference_with_nofresh_handoff']['cost'] == -2000


def test_two_exports_stay_small_and_cannot_overwrite(tmp_path):
    fixture(tmp_path)
    out = tmp_path/'results/diagnostic.json'
    receipt = mod.export(tmp_path, out)
    assert receipt['status'] == 'complete'
    assert set(receipt['files']) == {'diagnostic.json', 'evidence.json'}
    assert all(size < 48000 for size in receipt['files'].values())
    with pytest.raises(FileExistsError): mod.export(tmp_path, out)
