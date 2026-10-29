"""Synthetic accounting/statistical cases; no native runtime is imported."""
from copy import deepcopy
import json
from math import sqrt
from pathlib import Path
import unittest

from .analyze import NATIVE_PAIR_CHECKS, ZERO, analyze


PROTOCOL = json.loads((Path(__file__).parent/'protocol.json').read_text())


def collectors(effects=None):
    result = []
    for seed in PROTOCOL['external_seeds']:
        for scene in PROTOCOL['scenarios']:
            declarations = [r for r in PROTOCOL['roots'] if r['external_seed'] == seed and r['scenario'] == scene]
            roots = []
            for declared in declarations:
                ds = effects if effects is not None else ([-10, -9, -8, -10, -9, -8]
                    if declared['index'] == 0 else [1, 2, 3, 1, 2, 3])
                pairs = []
                for future, value in zip(declared['futures'], ds):
                    reference = dict(wait=100000, ride=200000, cost=400000,
                        dt=PROTOCOL['horizon']-declared['root']['started_at'], observed_at=24000, available_at=24030)
                    candidate = dict(reference)
                    candidate['ride'] += value*PROTOCOL['fixed_passenger_denominator']
                    candidate['cost'] = 2*candidate['wait']+candidate['ride']
                    rng = dict(all_pass=True, checks=dict(native_physical_serialization_unchanged=True,
                        other_native_rng_unchanged=True, lane_pools_changed_as_declared=True),
                        seed=future['seed'], mode=future['mode'], lane_count=64, changed_lane_pool_count=64,
                        all_declared_lane_pools_changed=True, burn_per_lane=1000000, other_native_streams='unchanged')
                    pair = dict(future_id=future['id'], future_seed=future['seed'], all_pass=True,
                        checks={**{k: True for k in NATIVE_PAIR_CHECKS}, 'physical_departure_separation': True,
                            'both_real_terminal': True, 'trace_persisted': True, 'same_future': True},
                        action={role: dict(hold_seconds=declared[role+'_hold_seconds'], executed_action=[0.])
                            for role in ('reference', 'candidate')},
                        remaining=dict(reference=reference, candidate=candidate),
                        delta={k: candidate[k]-reference[k] for k in ('wait', 'ride', 'cost')},
                        physical_noop=False, physical={}, rng={role: deepcopy(rng) for role in ('reference', 'candidate')},
                        rng_usage={role: dict(all_pass=True, native_clock=24000, total_draws=1200, pools_advanced=64)
                            for role in ('reference', 'candidate')})
                    pairs.append(pair)
                sequence = declared['root']['sequence']
                roots.append(dict(line_id=declared['line_id'], root=deepcopy(declared['root']),
                    root_ledger=dict(time_seconds=declared['root']['started_at'], accrued_wait_seconds=1000,
                        accrued_in_vehicle_seconds=2000),
                    checks=dict(exact_root=True, exact_prefix=True, exact_context=True, frozen_actor=True),
                    frozen_predictions=dict(line3=-1, service5=0, service_history6=-2),
                    prefix_reproduction=dict(all_pass=True, matched_issues=sequence, observed_issues=sequence,
                        reference_path='old_prefix.jsonl'), pairs=pairs))
            suffix_seconds = sum(12*(24000-d['root']['started_at']) for d in declarations)
            result.append(dict(status='complete', external_seed=seed, scenario=scene,
                integrity=dict(all_pass=True, checks=dict(exact_complete_prefix=True, pairs_complete=True)),
                prefix_reproduction=dict(all_pass=True, matched_issues=9000, observed_issues=9000,
                    source_rows=9000, reference_exhausted=True, reference_path='old_prefix.jsonl'),
                roots=roots, error=None,
                budget=dict(native_prefix_seconds=12600, suffix_native_seconds=suffix_seconds,
                    total_native_seconds=12600+suffix_seconds, total_native_seconds_is_lower_bound=False,
                    unknown_suffix_costs=0, suffixes=36, labels_observed=36, **{k: 0 for k in ZERO})))
    return result


class AnalyzeTest(unittest.TestCase):
    def test_full_real_schema_single_gate_and_distinct_sd_se(self):
        result = analyze(collectors(), PROTOCOL)
        self.assertEqual(result['decision'], 'LABEL_SIGNAL_SUPPORTED_FOR_SEPARATE_FEASIBILITY')
        self.assertEqual(result['accounting']['native_seconds'], 1777752)
        self.assertEqual(result['mean_gate']['roots_with_bound_below_zero'], [PROTOCOL['roots'][0]['id']])
        row = result['roots'][0]['statistics']
        self.assertAlmostEqual(row['mean_D'], -9)
        self.assertAlmostEqual(row['single_label_sd'], sqrt(.8))
        self.assertAlmostEqual(row['mean_SE'], sqrt(.8/6))
        self.assertAlmostEqual(row['simultaneous_one_sided_95_upper'], -9+4.219309115888926*sqrt(.8/6))
        self.assertEqual(row['frozen_service_history6']['prediction_minus_mean'], 7)
        self.assertEqual(row['mean_weighted_wait_per_passenger'], 0)

    def test_signed_negative_between_variance_and_no_extra_signal_gate(self):
        result = analyze(collectors([-3, -1, 1, -3, -1, 1]), PROTOCOL)
        self.assertEqual(result['decision'], 'CLOSE_FIXED_LABEL_CANDIDATE')
        desc = result['descriptive']
        self.assertEqual(desc['sample_variance_of_root_means'], 0)
        self.assertAlmostEqual(desc['signed_denoised_between_root_variance'], -3.2/6)
        self.assertEqual(desc['split_half_covariance'], 0)
        self.assertEqual(desc['split_half_same_sign_count'], 12)
        # Common negative means can pass even without any root heterogeneity.
        supported = analyze(collectors([-10, -9, -8, -10, -9, -8]), PROTOCOL)
        self.assertEqual(supported['decision'], 'LABEL_SIGNAL_SUPPORTED_FOR_SEPARATE_FEASIBILITY')
        self.assertEqual(supported['descriptive']['split_half_covariance'], 0)

    def test_missing_future_holds_without_reducing_multiplicity_or_imputing_cost(self):
        sources = collectors()
        source = sources[0]
        source['roots'][0]['pairs'].pop()
        source['status'] = 'failed'
        source['error'] = dict(type='NativeFailure', message='partial suffix')
        source['integrity']['all_pass'] = False
        source['budget'].update(total_native_seconds_is_lower_bound=True, unknown_suffix_costs=1)
        result = analyze(sources, PROTOCOL)
        self.assertEqual(result['decision'], 'HOLD_TECHNICAL_INCOMPLETE')
        self.assertIsNone(result['mean_gate']['passed'])
        self.assertEqual(result['mean_gate']['multiplicity'], 12)
        self.assertEqual(result['mean_gate']['t_critical'], PROTOCOL['statistics']['bonferroni_t_critical'])
        row = result['roots'][0]
        self.assertEqual(row['statistics']['n'], 5)
        self.assertIsNone(row['statistics']['simultaneous_one_sided_95_upper'])
        self.assertIsNotNone(row['statistics']['mean_D'])
        self.assertEqual(row['missing_futures'], ['R6'])
        self.assertIsNone(result['accounting']['native_seconds'])
        self.assertGreater(result['accounting']['known_native_seconds'], 0)
        self.assertEqual(result['accounting']['unknown_suffix_costs'], 1)

    def test_old_F0_does_not_enter_new_mean_and_missing_collector_is_unknown(self):
        sources = collectors()
        old = deepcopy(sources[0]['roots'][0]['pairs'][0])
        old.update(future_id='F0', future_seed=None)
        sources[0]['roots'][0]['pairs'].append(old)
        result = analyze(sources, PROTOCOL)
        self.assertEqual(result['decision'], 'HOLD_TECHNICAL_INCOMPLETE')
        self.assertEqual(result['roots'][0]['statistics']['n'], 6)
        self.assertEqual(result['roots'][0]['statistics']['mean_D'], -9)
        missing = analyze(sources[:3], PROTOCOL)
        self.assertIsNone(missing['accounting']['native_seconds'])
        self.assertEqual(missing['accounting']['missing_collectors'], 1)
        self.assertIsNone(missing['accounting']['learning_counts']['new_fits'])

    def test_physical_noop_retains_nonzero_D_but_native_change_invalidates_label(self):
        sources = collectors()
        pair = sources[0]['roots'][0]['pairs'][0]
        pair['physical_noop'] = True
        pair['checks']['physical_departure_separation'] = False
        result = analyze(sources, PROTOCOL)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['roots'][0]['pairs'][0]['D'], -10)
        pair['checks']['paired_after_physical'] = False
        failed = analyze(sources, PROTOCOL)
        self.assertEqual(failed['status'], 'hold')
        self.assertEqual(failed['roots'][0]['statistics']['n'], 5)
        self.assertEqual(failed['roots'][0]['pairs'][0]['D'], -10)
        self.assertFalse(failed['roots'][0]['pairs'][0]['valid'])

    def test_bad_root_and_missing_prediction_stay_reportable(self):
        sources = collectors()
        sources[0]['roots'][0]['checks']['exact_root'] = False
        sources[0]['roots'][0]['frozen_predictions'] = None
        result = analyze(sources, PROTOCOL)
        self.assertEqual(result['decision'], 'HOLD_TECHNICAL_INCOMPLETE')
        self.assertEqual(result['roots'][0]['statistics']['n'], 0)
        self.assertEqual(len(result['roots'][0]['pairs']), 6)
        self.assertEqual(result['descriptive']['complete_root_count'], 11)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
