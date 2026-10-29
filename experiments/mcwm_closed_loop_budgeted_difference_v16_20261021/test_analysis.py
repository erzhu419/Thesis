"""Resident contract tests for paired-label accounting and heldout interpretation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('v16_analysis', HERE/'analyze.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def endpoint(dt, wait, ride):
    return dict(status='complete', dt=dt, wait=wait, ride=ride, cost=2*wait+ride)


def fixture(root, protocol, *, preflight=False):
    """Use real four-day dimensions and serialized runner schemas, without SUMO."""
    p = analysis.settings(protocol, preflight=preflight)
    for cell, day in analysis.expected_days(root, protocol, preflight=preflight):
        train, heldout = [], []
        for line_index, line in enumerate(p['target_lines']):
            for window_index, window in enumerate(p['training_windows']):
                index = line_index*len(p['training_windows'])+window_index
                train.append(dict(id=f"train/{line}/{window['name']}", index=index, role='train', line_id=line,
                    window_name=window['name'], time=window['start']+100+50*line_index))
            for window in p['heldout_windows']:
                heldout.append(dict(id=f"heldout/{line}/{window['name']}", index=None, role='heldout', line_id=line,
                    window_name=window['name'], time=window['start']+100+50*line_index))
        train.sort(key=lambda row: row['time'])
        receipts, admitted, ordinary, selected_ids = [], [], [], []
        for packet_index, deadline in enumerate(range(p['feedback_first_deadline'], p['feedback_last_deadline']+1,
                p['feedback_interval_seconds'])):
            ready = next((row for row in train if row['id'] not in admitted and row['time']+330 <= deadline), None)
            pair_ids = [ready['id']] if ready else []
            outcome_ids = [-(2*ready['index']+1), -(2*ready['index']+2)] if ready else []
            ordinary_ids = [100000+packet_index*5+i for i in range(5-len(outcome_ids))]
            zeros = 2 if ready is not None and ready['index'] == 0 else 0
            receipts.append(dict(deadline=deadline, processed_at=deadline+1., records=5,
                sequence_ids=outcome_ids+ordinary_ids, ordinary_records=len(ordinary_ids), paired_records=len(outcome_ids),
                ordinary_sequence_ids=ordinary_ids, pair_ids=pair_ids, pair_outcome_ids=outcome_ids,
                pair_source_sequences=[ready['index']+1] if ready else [],
                pair_available_at=[ready['time']+330.] if ready else [], max_available_at=deadline-1.,
                positive_global_records=5-zeros, learned_global_records=5-zeros, zero_global_records=zeros))
            admitted.extend(pair_ids)
            ordinary.extend(ordinary_ids)
            selected_ids.extend(outcome_ids+ordinary_ids)
        readouts = {}
        for model in ('shared', 'route'):
            features = ['7X', '102X', '705X']+(['headway_imbalance'] if model == 'shared' else [])
            readouts[model] = dict(model=model, line_only=model == 'route', ridge=1., target_scale=1000.,
                feature_names=features, targets=['wait', 'ride'], pairs=len(admitted), pair_ids=admitted,
                by_line={line: sum(row['line_id'] == line for row in train) for line in p['target_lines']},
                design=dict(rows=len(admitted), columns=len(features), rank=min(len(admitted), len(features)),
                    active_columns=features, ridge_precision_condition=7.3, raw_condition=18.4),
                coefficients={name: dict(wait=.131415926, ride=.23456789, cost=.497399742) for name in features},
                training_error=dict(wait_mae=1.23456789, ride_mae=2.3456789, cost_mae=4.81481468),
                outcome_labels=2*len(admitted), refits=len(receipts))
        sources = []
        for item in train+heldout:
            t = item['time']
            past = [row for row in receipts if row['processed_at'] < t]
            counts = dict(policy=50000+8*len(past), world=2*p['model_refit_updates']*len(past),
                trust=8*len(past), physical_source=102, physical_target=0, world_observations=40000,
                target_labels=5*len(past), shared_pairs=sum(len(row['pair_ids']) for row in past),
                route_pairs=sum(len(row['pair_ids']) for row in past), readout_refits=len(past))
            line = item['line_id']
            features = [float(line == route) for route in ('7X', '102X', '705X')]+[.314159265]
            prediction = {model: dict(status='supported', line_id=line, support_pairs=3,
                features=features if model == 'shared' else features[:3],
                gain=dict(dt=0., wait=2. if model == 'shared' else -1., ride=1., cost=5. if model == 'shared' else -1.),
                fixed300_clock=dict(low=300., high=300.)) for model in ('shared', 'route')}
            sequence = item['index']+1 if item['role'] == 'train' else 100+len(sources)
            source = dict(id=item['id'], role=item['role'], index=item['index'], status='selected',
                line_id=line, window_name=item['window_name'],
                selected=dict(sequence=sequence, started_at=t, line_id=line, vehicle_id=f'{line}_3', trip_id=2,
                    station_id='diagnostic', s=[0.]*18, raw_s=[0.]*18, event='arrival',
                    issue_ledger=dict(time_seconds=t, waiting_passengers=12., in_vehicle_passengers=48.,
                        accrued_wait_seconds=t*12., accrued_in_vehicle_seconds=t*48.)),
                nominal_a=[0.], actions=dict(low=[0.], high=[.75]), features=features, prediction=prediction,
                prefix=dict(initializer='same_fixed_initializer.pt', initial_updates=50000,
                    external_seed=cell['seed']+protocol['target_seed_offset'], counters=counts,
                    receipts=deepcopy(past), native_steps=t, issued_count=sequence+1))
            sources.append(source)
            zero = item['role'] == 'train' and item['index'] == 0
            children = []
            for arm, hold, w, r in (('low', 0., 101., 299.), ('high', 7.5, 100., 300.)):
                transition = dict(sequence=sequence, started_at=t, completed_at=t+(0 if zero else 1),
                    available_at=t+(0 if zero else 1)+30., dt=0. if zero else 1., terminal=False,
                    line_id=line, station_id='diagnostic', s=[0.]*18, a=[hold/10], ns=[0.]*18, r=-1.,
                    episode_id='new', domain='target', vehicle_id=f'{line}_3', next_vehicle_id=f'{line}_3',
                    raw_s=[0.]*18, raw_ns=[0.]*18, trip_id=2)
                branch = dict(**deepcopy(source), cell=cell, source_evidence_id=source['id'],
                    execution=dict(requested_hold_seconds=hold, executed_hold_seconds=hold),
                    truth=dict(global_next=endpoint(0. if zero else 1., 0. if zero else 3., 0. if zero else 4.),
                        same_vehicle_next=endpoint(120., 50., 200.), fixed300=endpoint(300., w, r)),
                    global_transition=transition, suffix=dict(seconds=300., training_delta={key: 0 for key in counts}),
                    native=dict(other_batch_actions=[[77, [0.]]], prefix_matched_issues=sequence+1),
                    budget=dict(prefix_sim_seconds=t, suffix_sim_seconds=300., replay_target_labels=5*len(past),
                        replay_policy_updates=8*len(past), truth_labels_enter_training=0),
                    integrity=dict(all_pass=True, checks=dict(full_native_prefix_exact=True)))
                branch['status'] = 'complete'
                children.append(branch)
                if item['role'] == 'train':
                    branch_path = day/'training_branches'/str(item['index'])/arm
                else:
                    state = dict(**cell, line_id=line, window_name=item['window_name'])
                    branch_path = analysis.eval_path(root, state, arm, preflight=preflight)
                write(branch_path/'summary.json', branch)
            if item['role'] == 'train':
                outcomes = {}
                for arm, branch in zip(('low', 'high'), children):
                    transition = deepcopy(branch['global_transition'])
                    transition['sequence'] = -(2*item['index']+(1 if arm == 'low' else 2))
                    transition['available_at'] = t+330.
                    outcomes[arm] = dict(truth=branch['truth']['fixed300'], transition=transition)
                write(day/'pair_cache'/f"{item['index']}.json", dict(id=source['id'], index=item['index'],
                    role='train', source_sequence=sequence, issued_at=t, line_id=line, features=features,
                    available_at=t+330., outcomes=outcomes))
        budget = dict(target_labels=p['target_record_budget'], packets=len(receipts), ordinary_labels=len(ordinary),
            paired_action_labels=2*len(admitted), training_pairs=len(admitted), selected_sequence_ids=selected_ids,
            admitted_pair_ids=admitted, reserved_pairs=len(train), reserved_source_sequences=list(range(1, len(train)+1)),
            excluded_factual_sequences=list(range(1, len(train)+1)), pending_pairs=0, pending_pair_ids=[],
            pairs_by_line={line: sum(row['line_id'] == line for row in train) for line in p['target_lines']},
            zero_global_admitted=2, paired_outcomes_per_pair=2,
            native_seconds=p.get('preflight_stop_seconds', p['horizon_seconds']), policy_updates=p['policy_update_budget'],
            trust_updates=p['policy_update_budget'], world_updates=2*len(receipts)*p['model_refit_updates'],
            training_stopped_at=p['feedback_last_deadline']+1.)
        write(day/'acquisitions.json', dict(records=sources))
        write(day/'summary.json', dict(status='complete', cell=cell, integrity=dict(all_pass=True),
            initializer=dict(initial_updates=p['initializer_updates'], history_rows=p['history_rows'], source_rows=p['source_rows']),
            budget=budget, receipts=receipts, readouts=readouts, missing_cells=[],
            cost=dict(full_day=dict(wait=12345678., ride=34567890., cost=59259246.), trajectory='common passive flow')))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((HERE/'protocol.json').read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_budget_common_support_and_small_summary(self):
        fixture(self.root, self.protocol)
        value = analysis.analyze(self.root, self.protocol)
        self.assertTrue(value['integrity']['all_pass'], value['integrity'])
        self.assertEqual(value['budget']['main_training_slots'], 1140)
        self.assertEqual(value['budget']['main_paired_outcomes'], 72)
        self.assertEqual(value['budget']['main_ordinary_outcomes'], 1068)
        self.assertEqual(value['budget']['training_branch_compute']['suffix_sim_seconds'], 21600.)
        self.assertEqual(value['budget']['heldout_branch_compute']['suffix_sim_seconds'], 14400.)
        self.assertEqual(value['heldout_overall']['common_n'], 24)
        self.assertEqual(value['heldout_overall']['common_metrics']['shared'][9], 0.)
        self.assertEqual(value['heldout_overall']['common_metrics']['route'][9], 24.)
        self.assertEqual(value['heldout_overall']['common_metrics']['zero'][9], 24.)
        self.assertEqual(value['decision']['independent_confirmation'], 'not_submitted')
        self.assertEqual(len(value['training_pairs']), 36)
        self.assertEqual(sum(row['duration'] == 'zero_dt' for row in value['training_pairs']), 4)
        size = len(json.dumps(value, separators=(',', ':'), allow_nan=False).encode())
        self.assertLessEqual(size, analysis.MAX_BYTES, size)

    def test_preflight_admits_pair_and_bounded_gate(self):
        fixture(self.root, self.protocol, preflight=True)
        value = analysis.analyze(self.root, self.protocol, preflight=True)
        self.assertTrue(value['integrity']['all_pass'], value)
        self.assertEqual(value['budget']['main_training_slots'], 50)
        self.assertEqual(value['budget']['main_paired_outcomes'], 2)
        self.assertEqual(value['budget']['main_ordinary_outcomes'], 48)
        self.assertLessEqual(len(json.dumps(value, separators=(',', ':')).encode()), analysis.PREFLIGHT_MAX_BYTES)

    def test_factual_third_label_and_early_release_fail(self):
        fixture(self.root, self.protocol, preflight=True)
        day_path = self.root/'day'
        day = analysis.read_json(day_path/'summary.json')
        records = analysis.read_json(day_path/'acquisitions.json')['records']
        cache = analysis.read_json(day_path/'pair_cache'/'0.json')
        receipt = next(row for row in day['receipts'] if row['pair_ids'])
        old = receipt['ordinary_sequence_ids'][0]
        receipt['ordinary_sequence_ids'][0] = records[0]['selected']['sequence']
        receipt['sequence_ids'][receipt['sequence_ids'].index(old)] = records[0]['selected']['sequence']
        audit = analysis.inspect_budget(day, records, [cache], analysis.settings(self.protocol, preflight=True))
        self.assertFalse(audit['checks']['source_factual_not_third_label'])
        self.assertFalse(audit['all_pass'])
        cache['available_at'] -= 30
        audit = analysis.inspect_budget(day, records, [cache], analysis.settings(self.protocol, preflight=True))
        self.assertFalse(audit['checks']['paired_feedback_after_native300_delay'])

    def test_missing_unsupported_zero_and_no_fitted_evaluation(self):
        fixture(self.root, self.protocol, preflight=True)
        source = analysis.read_json(self.root/'day'/'acquisitions.json')['records'][1]
        low = analysis.read_json(self.root/'pair'/'low'/'summary.json')
        high = analysis.read_json(self.root/'pair'/'high'/'summary.json')
        for value in (source, low, high):
            value['prediction']['shared'].update(status='unsupported', gain=None, support_pairs=0)
        low['truth']['global_next'] = endpoint(0., 0., 0.)
        high['truth']['global_next'] = endpoint(0., 0., 0.)
        state = dict(seed=6201, scenario='stable', line_id='7X', window_name='preflight_holdout', target_id=source['id'])
        row = analysis.inspect_pair(state, source, low, high)
        self.assertTrue(row['integrity']['all_pass'])
        self.assertEqual(row['duration'], 'zero_dt')
        self.assertIsNone(row['effects']['shared'])
        summary = analysis.summarize([row])
        self.assertEqual(summary['model_metrics']['route'][0], 1)
        self.assertEqual(summary['common_n'], 0)
        self.assertEqual(summary['model_metrics']['shared'][0], 0)
        self.assertIsNone(summary['model_metrics']['shared'][1])
        low['truth']['fixed300'] = dict(status='missing', reason='cutoff')
        missing = analysis.inspect_pair(state, source, low, high)
        self.assertIsNone(missing['truth']['fixed300']['gain'])
        self.assertIsNone(missing['effects']['route'])
        self.assertEqual(analysis.summarize([missing])['missing_truth'], 1)
        high['suffix']['training_delta']['world'] = 1
        high['execution']['executed_hold_seconds'] = 0.
        broken = analysis.inspect_pair(state, source, low, high)
        self.assertFalse(broken['integrity']['checks']['suffix_zero_training'])
        self.assertFalse(broken['integrity']['checks']['actual_commands'])
        absent = analysis.inspect_pair(state, None, dict(status='selection_missing'), dict(status='selection_missing'))
        self.assertTrue(absent['integrity']['all_pass'])
        self.assertEqual(analysis.summarize([absent])['selected_valid_n'], 0)

    def test_cache_truth_must_match_its_native_child(self):
        fixture(self.root, self.protocol, preflight=True)
        cache_path = self.root/'day'/'pair_cache'/'0.json'
        cache = analysis.read_json(cache_path)
        cache['outcomes']['high']['truth']['wait'] += 10.
        write(cache_path, cache)
        value = analysis.analyze(self.root, self.protocol, preflight=True)
        self.assertFalse(value['days'][0]['integrity']['checks']['training_cache_matches_native_truth'])
        self.assertFalse(value['integrity']['all_pass'])


if __name__ == '__main__':
    unittest.main()
