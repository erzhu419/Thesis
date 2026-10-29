from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mcwm_closed_loop_service_response_v88_20261028.analysis import (
    OBSERVATION_CHECKS, analyze,
)
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import service_state


def protocol():
    return json.loads(Path(__file__).with_name('protocol.json').read_text())


def audit():
    return dict(all_pass=True, checks=dict(original_actor_exact=True), source_initializer=6207,
        external_seed=716207, scenario='stable', reference_roots=12,
        reference_source_paths={scenario: '/resident/v84/'+scenario+'/reference_actor.pt'
            for scenario in ('stable', 'local_speed_drop')},
        new_fits=0, new_neural_updates=0, v87_truth_used=0, native_seconds=0)


def context(onboard, release, queue, clock):
    seats = max(0., 50.-onboard+release)
    return dict(observed=True, reasons=[], capacity=50., next_stop='next_stop',
        clocks=dict(issue=clock, event=clock-1, obs_start=clock, obs_end=clock, atomic_at_issue=True),
        counts=dict(onboard=onboard, release_next=release, compatible_next_queue=queue,
            next_queue=queue, seats_after_release=seats))


def collector():
    coverage, selection, pairs = {}, {}, []
    counts = [('7X', 'low', 10, 0, 5), ('7X', 'high', 35, 10, 20),
        ('102X', 'low', 15, 0, 25), ('102X', 'high', 40, 5, 35),
        ('705X', 'low', 5, 0, 20), ('705X', 'high', 30, 0, 55)]
    for index, (line, group, onboard, release, queue) in enumerate(counts):
        key = line+'/'+group
        clock = float(12001+index)
        root = dict(started_at=clock, sequence=8000+index, trip_id=line+'_'+str(index), station_id=line+'15')
        ctx = context(onboard, release, queue, clock)
        row = dict(line_id=line, load_bin=group, root=root, context=ctx, service=service_state(ctx),
            binding=dict(episode_id=0, sequence=root['sequence'], trip_id=root['trip_id'], line_id=line,
                station_id=root['station_id'], event_clock=clock-1, issue_clock=clock),
            observation_checks={key: True for key in OBSERVATION_CHECKS},
            ref_hold=3.3+index*.01, cand_hold=7.5, sim_step_seconds=1., action_step_seconds=1.)
        coverage[key] = row
        selection[key] = dict(selected=root, status='selected', attempted=1, reasons={})
        if line != '7X':
            continue
        reference = dict(dt=24000-clock, wait=100., ride=800., cost=1000., observed_at=24000, available_at=24030)
        candidate = dict(reference, ride=810., cost=1010.)
        pairs.append(dict(status='complete', all_pass=True, checks=dict(native_root_and_rng=True,
            frozen_reference=True, original_f_action=True, capture_before_fork=True, zero_training=True),
            line_id=line, load_bin=group, root=deepcopy(root), service=deepcopy(row['service']),
            reference_hold_seconds=row['ref_hold'], candidate_hold_seconds=row['cand_hold'],
            sim_step_seconds=1., action_step_seconds=1., overlap_seconds=120.,
            remaining=dict(reference=reference, candidate=candidate), delta=dict(wait=0., ride=10., cost=10.),
            physical=dict(reference=dict(departure_clock=clock+4, next_arrival_clock=clock+80, missing={}),
                candidate=dict(departure_clock=clock+8, next_arrival_clock=clock+84, missing={}))))
    old = dict(source='v75', source_arm='fresh', seed=6207, scenario='stable', packets=37,
        covariates=185, complete_outcomes=185, counts=dict(critic=296, value=296, actor=296, alpha=296),
        max_available_at=12000, data_deadline=12000, deterministic=True, actor_frozen=True)
    suffix_seconds = sum(2*pair['remaining']['reference']['dt'] for pair in pairs)
    return dict(status='complete', integrity=dict(all_pass=True, checks=dict(frozen_actor=True)),
        seed=6207, external_seed=716207, scenario='stable', coverage=coverage, pairs=pairs, selection=selection,
        reference_snapshot=dict(prior_run_reference=True, available_before_heldout_episode=True,
            source_snapshot_metadata=old, heldout_external_seed=716207,
            source_actor_path=audit()['reference_source_paths']['stable']),
        budget=dict(prefixes=1, suffixes=4, native_prefix_seconds=12600., suffix_native_seconds=suffix_seconds,
            total_native_seconds=12600.+suffix_seconds, labels_observed=4,
            new_initialization_updates=0, prefix_training_updates=0, suffix_updates=0,
            labels_fed_back=0, new_fits=0))


def set_label(pair, delta):
    reference = pair['remaining']['reference']
    pair['remaining']['candidate'].update(ride=reference['ride']+delta, cost=reference['cost']+delta)
    pair['delta'] = dict(wait=0., ride=delta, cost=delta)


def refresh_pair_services(value):
    for pair in value['pairs']:
        pair['service'] = deepcopy(value['coverage'][pair['line_id']+'/'+pair['load_bin']]['service'])


class NativePreflightAnalysisTest(unittest.TestCase):
    def test_complete_native_preflight_has_design_rank_but_no_fit_or_control_claim(self):
        with patch('numpy.linalg.solve', side_effect=AssertionError('Native preflight cannot fit')):
            report = analyze(collector(), protocol(), audit())
        self.assertTrue(report['preflight']['passed'])
        self.assertEqual(report['potential_design']['ranks'], dict(line=3, conditional=5))
        self.assertEqual(report['potential_design']['rows'], 6)
        self.assertEqual(len(report['coverage']), 6)
        self.assertEqual(len(report['pairs']), 2)
        self.assertEqual(report['interpretation']['model_fits'], 0)
        self.assertFalse(report['interpretation']['control_performance_evaluated'])
        self.assertNotIn('promotion', report)
        self.assertNotIn('primary', report)
        self.assertNotIn('models', report)
        self.assertNotIn('coef', report['potential_design'])
        self.assertLess(len(json.dumps(report).encode()), 64000)

    def test_positive_negative_and_zero_labels_have_identical_admission_gate(self):
        values = collector()
        baseline = analyze(values, protocol(), audit())
        for delta in (-100., 0., 100.):
            altered = deepcopy(values)
            for pair in altered['pairs']:
                set_label(pair, delta)
            report = analyze(altered, protocol(), audit())
            self.assertTrue(report['preflight']['passed'])
            self.assertEqual(report['integrity'], baseline['integrity'])
            self.assertEqual(report['potential_design'], baseline['potential_design'])
            self.assertEqual(report['pairs'][0]['label_delta']['cost'], delta)

    def test_same_line_two_states_do_not_imply_identifiable_conditional_design(self):
        values = collector()
        for row in values['coverage'].values():
            native = row['context']['counts']
            native.update(release_next=0, compatible_next_queue=0, next_queue=0,
                seats_after_release=50.-native['onboard'])
            row['service'] = service_state(row['context'])
        refresh_pair_services(values)
        report = analyze(values, protocol(), audit())
        self.assertTrue(report['integrity']['checks']['each_line_two_load_states'])
        self.assertEqual(report['potential_design']['ranks']['conditional'], 4)
        self.assertFalse(report['preflight']['passed'])

    def test_missing_slots_or_native_pair_are_retained_not_imputed(self):
        values = collector()
        values['coverage'].pop('705X/high')
        values['selection']['705X/high'].update(status='missing', selected=None)
        report = analyze(values, protocol(), audit())
        self.assertEqual(report['exceptions']['missing_covariate_slots'], ['705X/high'])
        self.assertEqual(len(report['coverage']), 5)
        self.assertFalse(report['preflight']['passed'])
        values = collector()
        values['pairs'].pop()
        report = analyze(values, protocol(), audit())
        self.assertEqual(report['exceptions']['missing_native_pairs'], ['7X/high'])
        self.assertFalse(report['preflight']['passed'])

    def test_atomic_observation_and_clock_ledger_rng_guards_are_required(self):
        for field in ('clock_unchanged', 'ledger_unchanged', 'rng_unchanged', 'context_before_action'):
            values = collector()
            values['coverage']['102X/low']['observation_checks'][field] = False
            report = analyze(values, protocol(), audit())
            self.assertFalse(report['preflight']['passed'], field)
        values = collector()
        row = values['coverage']['102X/low']
        row['context']['clocks']['obs_end'] += 1
        row['service'] = service_state(row['context'])
        report = analyze(values, protocol(), audit())
        self.assertIsNone(report['coverage']['102X/low']['service']['features'])
        self.assertFalse(report['preflight']['passed'])

    def test_issue_binding_and_service_recomputation_detect_wrong_state(self):
        values = collector()
        values['coverage']['102X/low']['binding']['sequence'] += 1
        self.assertFalse(analyze(values, protocol(), audit())['preflight']['passed'])
        values = collector()
        values['coverage']['102X/low']['service']['features'][0] += .1
        report = analyze(values, protocol(), audit())
        self.assertFalse(report['coverage']['102X/low']['checks']['service_recomputed_exact'])
        self.assertFalse(report['preflight']['passed'])

    def test_unsupported_action_and_single_load_bin_fail(self):
        values = collector()
        values['coverage']['102X/low']['ref_hold'] = 3.75
        report = analyze(values, protocol(), audit())
        self.assertFalse(report['coverage']['102X/low']['checks']['original_positive_action_support'])
        self.assertFalse(report['preflight']['passed'])
        values = collector()
        row = values['coverage']['102X/high']
        row['context']['counts'].update(onboard=10., seats_after_release=45.)
        row['service'] = service_state(row['context'])
        report = analyze(values, protocol(), audit())
        self.assertFalse(report['coverage']['102X/high']['checks']['expected_load_bin'])

    def test_physical_noop_and_no_wall_overlap_remain_invalid(self):
        values = collector()
        pair = values['pairs'][0]
        pair['physical']['candidate']['departure_clock'] = pair['physical']['reference']['departure_clock']
        set_label(pair, 0.)
        report = analyze(values, protocol(), audit())
        self.assertFalse(report['pairs'][0]['checks']['physical_departure_separation'])
        self.assertIsNone(report['pairs'][0]['label_delta'])
        self.assertFalse(report['preflight']['passed'])
        values = collector()
        values['pairs'][0]['overlap_seconds'] = 0.
        self.assertFalse(analyze(values, protocol(), audit())['preflight']['passed'])

    def test_true_terminal_maturity_and_own_root_are_required(self):
        values = collector()
        values['pairs'][0]['remaining']['candidate']['available_at'] = 24000
        self.assertFalse(analyze(values, protocol(), audit())['preflight']['passed'])
        values = collector()
        values['pairs'][0]['root']['sequence'] += 1
        self.assertFalse(analyze(values, protocol(), audit())['preflight']['passed'])

    def test_new_fits_v87_truth_training_or_excess_budget_block(self):
        frozen = audit()
        frozen['v87_truth_used'] = 1
        self.assertFalse(analyze(collector(), protocol(), frozen)['preflight']['passed'])
        for field in ('new_fits', 'suffix_updates', 'prefix_training_updates', 'labels_fed_back'):
            values = collector()
            values['budget'][field] = 1
            self.assertFalse(analyze(values, protocol(), audit())['preflight']['passed'], field)
        values = collector()
        values['budget']['native_prefix_seconds'] += 10
        values['budget']['total_native_seconds'] += 10
        self.assertFalse(analyze(values, protocol(), audit())['preflight']['passed'])
        altered = protocol()
        altered['features']['load_bin_threshold'] = .6
        self.assertFalse(analyze(collector(), altered, audit())['preflight']['passed'])


if __name__ == '__main__':
    unittest.main()
