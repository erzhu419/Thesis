"""Synthetic native-receipt contracts; no simulation or new labels."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from experiments.mcwm_closed_loop_future_rng_v102_20261028.analyze import analyze, ZERO


HERE = Path(__file__).resolve().parent


def fixture():
    protocol = json.loads((HERE/'protocol.json').read_text())
    sources = []
    for declared in protocol['roots']:
        clock = declared['clock']
        remaining = dict(wait=100000., ride=200000., cost=400000., dt=24000-clock,
                         observed_at=24000, available_at=24030)
        root = dict(started_at=clock, sequence=declared['sequence'], line_id='7X',
                    trip_id=declared['trip_id'], station_id=declared['station_id'])
        ledger = dict(time_seconds=clock, accrued_wait_seconds=1000., accrued_in_vehicle_seconds=2000.)
        pairs = []
        for future in protocol['futures']:
            effect = round(declared['old_D']*15079) if future['id'] == 'F0' else \
                -1000 if future['id'] == 'F1' else 2000
            candidate = dict(remaining, ride=remaining['ride']+effect, cost=remaining['cost']+effect)
            rng = dict(all_pass=True, checks=dict(native_physical_serialization_unchanged=True,
                other_native_rng_unchanged=True, python_numpy_torch_rng_unchanged=True,
                native_clock_unchanged=True, lane_pools_changed_as_declared=True),
                seed=future['seed'], mode=future['mode'], lane_count=64,
                other_native_streams='unchanged', changed_lane_pools=[] if future['seed'] is None else list(range(64)),
                burn_per_lane=0 if future['seed'] is None else 1000000)
            checks = dict(same_complete_root=True, same_future=True, both_real_terminal=True,
                physical_departure_separation=True, reference_before_equals_parent=True,
                candidate_before_equals_parent=True, paired_after_physical=True,
                paired_after_lane_rng=True, paired_after_other_rng=True)
            pairs.append(dict(future_id=future['id'], future_seed=future['seed'], all_pass=True,
                checks=checks, physical_noop=False, remaining=dict(reference=deepcopy(remaining), candidate=candidate),
                delta=dict(wait=0., ride=effect, cost=effect),
                rng={role:deepcopy(rng) for role in ('reference', 'candidate')},
                rng_usage={role:dict(all_pass=True, total_draws=1200, pools_advanced=8, native_clock=24000)
                           for role in ('reference', 'candidate')}, physical={}))
        budget = dict(prefixes=1, suffixes=6, native_prefix_seconds=clock,
            suffix_native_seconds=6*(24000-clock), total_native_seconds=clock+6*(24000-clock),
            total_native_seconds_is_lower_bound=False, unknown_suffix_costs=0, labels_observed=6,
            details=[], **{key:0 for key in ZERO})
        sources.append(dict(status='complete', case=declared['case'], external_seed=declared['external_seed'],
            scenario='stable', root=root, root_ledger=ledger, frozen_original_root_ledger=deepcopy(ledger),
            frozen_original_remaining=deepcopy(pairs[0]['remaining']),
            integrity=dict(all_pass=True, checks=dict(exact_root=True, error_free=True)),
            root_checks=dict(exact_root=True, exact_context=True, exact_prefix=True),
            prefix_reproduction=dict(all_pass=True, observed_issues=declared['sequence'],
                matched_issues=declared['sequence'], reference_path='frozen/prefix_trace.jsonl'),
            pairs=pairs, budget=budget, error=None))
    probe = dict(status='complete', all_pass=True, checks=dict(interventions_valid=True,
        futures_differ=True, changed_streams_are_used=True, fixed_budget=True), native_seconds=40)
    return sources, protocol, probe


class AnalyzeTest(unittest.TestCase):
    def test_actual_prefix_schema_full_budget_and_observed_sign_flip(self):
        sources, protocol, probe = fixture()
        result = analyze(sources, protocol, probe)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['decision'], 'OBSERVED_WITHIN_ROOT_SIGN_FLIP')
        self.assertEqual(result['accounting']['formal']['native_seconds'], 167955)
        self.assertEqual(result['accounting']['capability_probe']['native_seconds'], 40)
        self.assertEqual(result['accounting']['combined_native_seconds'], 167995)
        for root in result['roots']:
            self.assertTrue(root['within_root']['sign_flip_observed'])
            self.assertFalse(root['within_root']['robustness_established'])
            self.assertTrue(root['pairs'][0]['F0_reproduction']['all_pass'])

    def test_equal_D_does_not_hide_common_F0_endpoint_drift(self):
        sources, protocol, probe = fixture()
        pair = sources[0]['pairs'][0]
        for arm in pair['remaining'].values():
            arm['wait'] += 1
            arm['cost'] += 2
        result = analyze(sources, protocol, probe)
        reproduction = result['roots'][0]['pairs'][0]
        self.assertTrue(reproduction['checks']['exact_delta'])
        self.assertTrue(reproduction['checks']['original_D'])
        self.assertFalse(reproduction['F0_reproduction']['all_pass'])
        self.assertEqual(result['decision'], 'TECHNICAL_INCOMPLETE')

    def test_failed_partial_root_keeps_pairs_and_unknown_spend(self):
        sources, protocol, probe = fixture()
        source = sources[1]
        source.update(status='failed', error=dict(type='RuntimeError', message='native child failed'))
        source['pairs'] = source['pairs'][:1]
        source['budget'].update(suffixes=3, labels_observed=2, unknown_suffix_costs=1,
            suffix_native_seconds=2*(24000-12006), total_native_seconds=12006+2*(24000-12006),
            total_native_seconds_is_lower_bound=True)
        result = analyze(sources, protocol, probe)
        self.assertEqual(len(result['roots'][1]['pairs']), 1)
        self.assertTrue(result['roots'][1]['pairs'][0]['valid'])
        self.assertEqual(result['roots'][1]['missing_futures'], ['F1', 'F2'])
        self.assertEqual(result['accounting']['formal']['unknown_suffix_costs'], 1)
        self.assertIsNone(result['accounting']['formal']['native_seconds'])
        self.assertGreater(result['accounting']['formal']['known_native_seconds'], 0)
        self.assertIsNone(result['accounting']['combined_native_seconds'])
        self.assertEqual(result['decision'], 'TECHNICAL_INCOMPLETE')

    def test_physical_noop_keeps_nonzero_measured_D(self):
        sources, protocol, probe = fixture()
        pair = sources[0]['pairs'][1]
        pair['physical_noop'] = True
        pair['checks']['physical_departure_separation'] = False
        result = analyze(sources, protocol, probe)
        observed = result['roots'][0]['pairs'][1]
        self.assertTrue(observed['valid'])
        self.assertAlmostEqual(observed['D'], -1000/15079)
        self.assertEqual(result['status'], 'complete')

    def test_no_flip_does_not_establish_robustness(self):
        sources, protocol, probe = fixture()
        for source in sources:
            sign = -1 if source['case'] == 'T' else 1
            for pair in source['pairs'][1:]:
                effect = sign*1000
                ref = pair['remaining']['reference']
                pair['remaining']['candidate'] = dict(ref, ride=ref['ride']+effect, cost=ref['cost']+effect)
                pair['delta'] = dict(wait=0., ride=effect, cost=effect)
        result = analyze(sources, protocol, probe)
        self.assertEqual(result['decision'], 'NO_FLIP_OBSERVED_IN_FIXED_FUTURES')
        self.assertTrue(result['integrity']['all_pass'])
        self.assertTrue(all(not root['within_root']['robustness_established'] for root in result['roots']))


if __name__ == '__main__':
    unittest.main()
