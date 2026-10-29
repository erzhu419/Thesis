"""Checks for legal endpoints, delayed costs, and mixed-backup accounting."""
from copy import deepcopy
import unittest

import numpy as np

from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.analyze_chain import (
    MODELS, analyze_arm, analyze_pair,
)


START = 1000.


def memory(clock):
    return dict(observed_at=clock, vector=[0.]*84)


def ledger(relative, high=False):
    wait = relative+(max(relative-5., 0.) if high else 0.)
    return dict(time_seconds=START+relative, accrued_wait_seconds=wait,
        accrued_in_vehicle_seconds=2.*relative)


def state(index):
    return [float(index)]+[0.]*228


def fixture(*, high=False, inserted=False, endpoint_relative=301.):
    specs = [(0., 'selected'), (10., 'B'), (endpoint_relative, 'C')]
    if inserted:
        specs.insert(1, (5., 'inserted'))
    rows, previous_clock = [], 0.
    for index, (completed, identity) in enumerate(specs):
        a = [-1.+(1.875/30. if high and index == 0 else 0.)]
        s, ns = state(index), state(index+1)
        ls, ln = ledger(previous_clock, high), ledger(completed, high)
        raw_reward = -(2.*(ln['accrued_wait_seconds']-ls['accrued_wait_seconds'])
            +ln['accrued_in_vehicle_seconds']-ls['accrued_in_vehicle_seconds'])
        transition = dict(sequence=80+index, started_at=START+previous_clock,
            completed_at=START+completed, dt=completed-previous_clock, terminal=False,
            s=s, ns=ns, raw_s=s, raw_ns=ns, a=a, r=raw_reward,
            memory_s=memory(START+previous_clock), memory_ns=memory(START+completed),
            line_id='7X', vehicle_id=identity, trip_id=identity, station_id=identity,
            next_vehicle_id=specs[index+1][1] if index+1 < len(specs) else 'endpoint')
        scores = {}
        for model_index, model in enumerate(MODELS):
            reward = float(np.float32(np.float32(raw_reward)*np.float32(1e-4)))
            bootstrap = float(np.float32(.31-.01*index+.001*model_index))
            q = float(np.float32(.21+.001*index+.001*model_index+(0.01 if high else 0.)))
            target = float(np.float32(np.float32(reward)+np.float32(bootstrap)))
            scores[model] = dict(sequence=transition['sequence'],
                started_at=transition['started_at'], completed_at=transition['completed_at'],
                executed_action=a[0], q_min=q, target_q_min=q+.01, reward_scaled=reward, target=target,
                value_contribution=.1*bootstrap, sac_contribution=.9*bootstrap,
                sac_q_without_entropy=bootstrap, sac_entropy_bonus=0.,
                bootstrap_contribution=bootstrap)
        rows.append(dict(kind='endpoint_actual_without_issue' if index == len(specs)-1 else 'global_transition',
            transition=transition, ledger_s=ls, ledger_ns=ln, scores=scores))
        previous_clock = completed
    ticks = []
    for relative in range(int(endpoint_relative)+1):
        ls = ledger(float(relative), high)
        wait, ride = ls['accrued_wait_seconds'], ls['accrued_in_vehicle_seconds']
        ticks.append(dict(clock_seconds=START+relative, relative_seconds=float(relative),
            wait=wait, ride=ride, cost=2*wait+ride, ledger=ls))
    selected = dict(sequence=80, started_at=START, s=state(0), memory_s=memory(START),
        issue_ledger=ledger(0.))
    h = ticks[300]
    exact = dict(status='complete', dt=300., **{key: h[key] for key in ('wait', 'ride', 'cost')})
    endpoint = dict(status='observed', clock=START+endpoint_relative,
        s=state(len(specs)), memory=memory(START+endpoint_relative), issued=False, executed=False)
    return rows, ticks, selected, exact, endpoint


class AnalyzeChainTests(unittest.TestCase):
    def test_float32_accounting_zero_duration_and_no_state_at300(self):
        rows, ticks, selected, exact, endpoint = fixture()
        result = analyze_arm(rows, ticks, selected=selected, exact300=exact, endpoint=endpoint)
        self.assertTrue(result['integrity']['all_pass'], result['integrity'])
        self.assertEqual(result['zero_duration_events'], 1)
        self.assertEqual(result['partial_to300']['clock_from'], 1010.)
        self.assertIsNone(result['partial_to300']['state'])
        self.assertIsNone(result['partial_to300']['bootstrap'])
        self.assertEqual(result['physical_300_to_endpoint']['cost'], 4.)
        for model in MODELS:
            point = result['models'][model]['endpoint']
            self.assertAlmostEqual(point['reconstructed_initial_q'], point['initial_q_min'], places=12)
            self.assertAlmostEqual(point['stitched_return'],
                point['reward_scaled_sum']+point['final_bootstrap'])
            self.assertAlmostEqual(point['continuation_gap_sum'],
                point['mixed_vs_recorded_target_q_gap_sum']+point['target_lag_gap_sum'])
        # Explicit float32 target addition is not diagnosed as a broken chain.
        self.assertGreater(result['models']['zero_memory']['maximum_native_addition_roundoff'], 0.)

    def test_breaking_real_state_or_causal_memory_fails_before_interpretation(self):
        for field in ('state', 'memory'):
            rows, ticks, selected, exact, endpoint = fixture()
            if field == 'state':
                rows[1]['transition']['s'] = state(99)
            else:
                rows[1]['transition']['memory_s']['observed_at'] += 1.
            result = analyze_arm(rows, ticks, selected=selected, exact300=exact, endpoint=endpoint)
            self.assertFalse(result['integrity']['all_pass'])
            self.assertEqual(result['status'], 'integrity_failed')
            self.assertEqual(result['models'], {})

    def test_pair_matches_semantics_not_ordinal_and_uses_fixed_ticks_for_onset(self):
        low, lt, selected, lexact, lend = fixture()
        high, ht, _, hexact, hend = fixture(high=True, inserted=True)
        result = analyze_pair(low, high, lt, ht, selected=selected,
            exact300_by_arm=dict(low=lexact, high=hexact), endpoint_by_arm=dict(low=lend, high=hend))
        self.assertTrue(result['integrity']['all_pass'])
        self.assertEqual(result['semantic_alignment']['matched'], 3)
        self.assertEqual(result['semantic_alignment']['unmatched_high'], 1)
        self.assertEqual(result['semantic_alignment']['unmatched_low'], 0)
        reward = result['semantic_alignment']['first_differences']['r']
        self.assertEqual(reward['identity'][1], 'B')
        self.assertFalse(reward['equal_issue_clock'])
        self.assertEqual(result['fixed_clock_cost']['first_differences']['cost']['relative_seconds'], 6.)
        self.assertEqual(result['fixed300']['benefit_low_minus_high'], -590.)
        self.assertEqual(result['models']['shared_level']['initial_q']['regret'], 590.)

    def test_different_actual_endpoints_are_explicit(self):
        low, lt, selected, lexact, lend = fixture()
        high, ht, _, hexact, hend = fixture(high=True, endpoint_relative=303.)
        result = analyze_pair(low, high, lt, ht, selected=selected,
            exact300_by_arm=dict(low=lexact, high=hexact), endpoint_by_arm=dict(low=lend, high=hend))
        point = result['models']['zero_memory']['endpoint']
        self.assertFalse(point['same_clock'])
        self.assertEqual(point['clocks'], [1301., 1303.])
        self.assertEqual(result['fixed300']['benefit_low_minus_high'], -590.)

    def test_missing_native_endpoint_keeps_partial_ledger_without_bootstrap(self):
        rows, ticks, selected, exact, endpoint = fixture(endpoint_relative=420.)
        # At the overrun cap no actual event exists: the pending interval is
        # not closed or turned into a terminal transition.
        rows = deepcopy(rows[:-1])
        endpoint = dict(status='missing', clock=None, s=None, memory=None, issued=False, executed=False)
        result = analyze_arm(rows, ticks, selected=selected, exact300=exact, endpoint=endpoint)
        self.assertTrue(result['integrity']['all_pass'])
        self.assertEqual(result['status'], 'endpoint_missing')
        self.assertIsNone(result['models']['zero_memory']['endpoint'])
        self.assertIsNone(result['physical_300_to_endpoint'])
        self.assertEqual(result['partial_to300']['cost'], 1160.)


if __name__ == '__main__':
    unittest.main()
