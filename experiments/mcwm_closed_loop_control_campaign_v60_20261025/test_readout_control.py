"""Verify native commands, causal paired fitting and matched intervention slots."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_control_campaign_v60_20261025 import readout_control as r
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import ExecutionAction
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from experiments.mcwm_closed_loop_matched_response_v59_20261025.test_difference import pair
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
from experiments.mcwm_closed_loop_v1.stream import EI


def event(*, line='7X', now=18400., sequence=0, station='new_station'):
    physical = np.zeros(max(EI.values())+1, np.float32)
    for key, value in dict(forward_headway_seconds=660, target_forward_headway_seconds=600,
            backward_headway_seconds=540, target_backward_headway_seconds=600).items():
        physical[EI[key]] = value
    return dict(line_id=line, started_at=now, sequence=sequence, station_id=station,
        trip_id='trip', a=np.array([-.95], np.float32), s=table(.8)['s'][0], event=physical,
        memory_s=dict(vector=list(range(84)), observed_at=now))


def selector(arm, *, gain=10., stopped=True):
    transform = ExecutionAction(TransitMDPConfigV7())
    learner = SimpleNamespace(device=torch.device('cpu'), action_transform=transform)
    model = SimpleNamespace(predict=lambda features: dict(status='supported',
        gain=dict(cost=gain), headway_extrapolation=True))
    api = SimpleNamespace(vehicle=SimpleNamespace(isAtBusStop=lambda trip: stopped,
        getActionStepLength=lambda trip: 1.), simulation=SimpleNamespace(getDeltaT=lambda: 1.))
    return r.InterventionSelector(learner, model, arm=arm, api=api,
        training_stations={line: ['old_station'] for line in r.LINES})


def test_readout_selects_high_from_positive_cost_saving_and_preserves_F():
    choose = selector('readout', gain=10.)
    issue = event()
    selected = choose(issue)
    transform = choose.learner.action_transform
    applied = transform.numpy(issue['s'], selected['proposal'])
    np.testing.assert_array_equal(applied, selected['action'])
    assert float(transform.hold_seconds(torch.as_tensor(applied))[0]) == 7.5
    assert choose.rows[-1]['selected'] is True
    assert choose.rows[-1]['training_station_seen'] is False
    assert choose.counts['training_station_unseen'] == 1
    assert choose(event(sequence=1)) is None
    assert choose.counts['skipped_used_bin'] == 1


@pytest.mark.parametrize('gain', [-1., 0.])
def test_negative_or_tied_cost_difference_executes_zero(gain):
    choose = selector('readout', gain=gain)
    selected = choose(event())
    assert choose.rows[-1]['choice'] == 0
    assert float(choose.learner.action_transform.hold_seconds(torch.as_tensor(selected['action']))[0]) == 0.


def test_q_selector_scores_executed_candidates_with_actual_issue_memory(monkeypatch):
    choose = selector('q_selector')
    captures = {}

    def explicit(learner, states, memory):
        captures['memory'] = memory.clone()
        return torch.cat((states, memory), dim=1)

    def critic(states, actions):
        assert states.shape == (2, 313)
        captures['actions'] = actions.clone()
        return torch.tensor([[0., -1.], [1., -2.]])

    monkeypatch.setattr(r, 'critic_states', explicit)
    choose.learner.critic = critic
    selected = choose(event())
    assert choose.rows[-1]['choice'] == 0
    assert choose.rows[-1]['q']['minimum'] == [0., -2.]
    assert captures['memory'].tolist() == [list(range(84))]
    assert captures['actions'].shape == (2, 1)
    np.testing.assert_array_equal(selected['action'], captures['actions'][0].numpy())


def test_actor_baseline_uses_same_first_eligible_bin_without_replacing_action():
    choose = selector('actor_mcwm')
    issue = event()
    before = deepcopy(issue)
    assert choose(issue) is None
    np.testing.assert_array_equal(issue['a'], before['a'])
    assert choose.counts['eligible_first_issues'] == 1
    assert choose.counts['selected'] == 0
    assert len(choose.used) == 1
    assert choose.rows[0]['selected'] is False


def test_native_ineligible_issue_does_not_consume_the_first_executable_slot():
    choose = selector('readout', stopped=False)
    assert choose(event()) is None
    assert len(choose.used) == 0
    choose.api.vehicle.isAtBusStop = lambda trip: True
    assert choose(event(sequence=1)) is not None
    assert len(choose.used) == 1
    assert choose.counts['skipped_native_stop_or_gap_unsupported'] == 1


def test_each_line_and300second_bin_has_one_intervention_up_to57():
    choose = selector('readout')
    assert choose(event(now=18299.)) is None
    for index in range(19):
        for line in r.LINES:
            now = r.START+index*300+1
            assert choose(event(line=line, now=now)) is not None
            assert choose(event(line=line, now=now+1)) is None
    assert choose(event(now=24000.)) is None
    assert len(choose.used) == choose.counts['selected'] == 57
    assert all(row['selected'] == 19 for row in choose.by_line.values())


def test_fitting_opens_only_paid_training_records_and_covariate_receipts(tmp_path, monkeypatch):
    collection = tmp_path/'stable'
    collection.mkdir()
    records, packets = [], []
    for index in range(9):
        line = r.LINES[index % 3]
        row = pair(index=index, line=line)
        row.update(admitted_at=900.+index, selected=dict(station_id='train_stop'))
        records.append(row)
        packets.append(dict(index=index+1, receipt=dict(processed_at=900.+index,
            pair_ids=[row['id']], paired_records=2)))
    (collection/'admitted_pair_records.json').write_text(json.dumps(records))
    seen = []

    def load(path, **kwargs):
        seen.append((path.name, kwargs))
        assert path.name == 'covariates_dataset.json'
        assert kwargs == dict(arm='no_fresh', scenario='stable', seed=6202)
        return dict(packets=packets)

    monkeypatch.setattr(r, 'load_dataset', load)
    model, stations, proof = r.fit_paid_readout(collection, 'readout_line')
    assert model.summary()['pairs'] == 9
    assert stations == {line: ['train_stop'] for line in r.LINES}
    assert proof['paid_pair_slots'] == 18
    assert proof['heldout_truth_opened'] is False
    assert seen[0][0] == 'covariates_dataset.json'
    # No heldout/truth/dataset file exists: none may be required by this fit.
    records[0]['role'] = 'heldout'
    (collection/'admitted_pair_records.json').write_text(json.dumps(records))
    with pytest.raises(ValueError, match='heldout|unadmitted'):
        r.fit_paid_readout(collection, 'readout_line')


def test_native_step_applies_replacement_and_later_batch_state_sees_real_action(monkeypatch):
    contract = TransitMDPConfigV7()
    natives = [SimpleNamespace(bus_id=trip, metadata={}) for trip in ('first', 'second')]
    env = SimpleNamespace(_current_events=natives, done=False)
    env._bridge = SimpleNamespace(consume_completed_trip_passenger_times=lambda: None)
    commands_seen, states_seen, logged = [], [], []

    def advance(commands):
        commands_seen.append(commands)
        for native in natives:
            native.metadata['applied_additional_hold_seconds'] = commands[native.bus_id][0]

    env.step_to_event = advance

    def choose(issue):
        if issue['sequence'] != 0:
            return None
        issue['intervention_row'] = dict(holding_seconds=7.5)
        return dict(action=np.array([-.75], np.float32), proposal=np.array([1.], np.float32))

    stream = r.ControlEventStream(env, np.ones(12), contract, selector=choose,
        execute_transform=False, on_issue=lambda issue: logged.append(deepcopy(issue)))
    stream.episode_id = 0
    stream.memory = SimpleNamespace(snapshot=lambda now: dict(observed_at=now, vector=[0.]*84))
    snapshot = SimpleNamespace(time_seconds=18400.)
    stream._snapshot = lambda agent=None: snapshot
    stream._close_physical = lambda event: None
    stream._close = lambda *args, **kwargs: None
    stream._canonical = lambda raw: raw
    batch = [dict(agent=trip, trip_id=trip, vehicle_id=trip, terminal=False,
        line_id='7X', station_id='station', event=np.zeros(14)) for trip in ('first', 'second')]
    stream._batch = lambda: batch

    def pack(local, index, prior, config):
        state = np.zeros(229, np.float32)
        state[0] = prior[0]
        return state

    monkeypatch.setattr(r, 'pack_event_batch_v7', pack)

    def policy(state):
        states_seen.append(state.copy())
        return np.array([1.], np.float32)

    assert stream.step(policy) == []
    assert commands_seen == [dict(first=[7.5, 1.], second=[60., 1.])]
    assert states_seen[1][0] == -.75
    assert logged[0]['a'][0] == -.75
    assert logged[1]['a'][0] == 1.
    assert stream.actual_execution_checks == 1
    assert stream.maximum_execution_gap_seconds == 0.
