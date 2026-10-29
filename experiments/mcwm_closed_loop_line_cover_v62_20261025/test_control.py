"""Check line coverage without changing feedback rights, timing or replay."""
from dataclasses import asdict
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_line_cover_v62_20261025 import control as c
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def packet_protocol():
    return dict(feedback_first_deadline=1200, feedback_last_deadline=18000,
        feedback_interval_seconds=300, feedback_records_per_packet=5)


def row(sequence, line='7X', available=1190., *, dt=1., terminal=False):
    return SimpleNamespace(sequence=sequence, line_id=line, available_at=available,
        started_at=available-30-dt, dt=dt, terminal=terminal)


@pytest.mark.parametrize('line', ['7X', '102X', '705X', '7S'])
def test_single_line_pool_is_exactly_original_latest_five(line):
    old, new = c.formal.PacketBudget(packet_protocol()), c.LineCoveragePacketBudget(packet_protocol())
    rows = [row(index, line, 1180+index) for index in range(12)]
    for budget in (old, new):
        budget.ingest(rows)
    original, old_discarded = old.take(1201)
    selected, discarded = new.take(1201)
    assert [value.sequence for value in selected] == [value.sequence for value in original]
    assert discarded == old_discarded
    assert new.rows == old.rows
    receipt = dict(new.receipts[0])
    del receipt['line_cover']
    assert receipt == old.receipts[0]
    assert new.selection_checks_pass()


def test_present_focus_lines_receive_latest_representative_then_latest_fill():
    budget = c.LineCoveragePacketBudget(packet_protocol())
    rows = [row(0, '7X', 1180), row(1, '102X', 1181), row(2, '705X', 1182),
        row(3, '7X', 1183)]+[row(index, '7S', 1180+index) for index in range(4, 12)]
    budget.ingest(rows)
    selected, _ = budget.take(1200)
    assert [value.sequence for value in selected] == [1, 2, 3, 10, 11]
    assert budget.receipts[0]['line_cover']['focus_latest_ids'] == {'7X': 3, '102X': 1, '705X': 2}
    assert budget.selection_checks_pass()
    assert sum(budget.summary()['selected_record_counts_by_line'].values()) == 5


def test_missing_focus_lines_are_not_required_and_focus_can_fill_extra_slots():
    budget = c.LineCoveragePacketBudget(packet_protocol())
    budget.ingest([row(index, '7X' if index >= 2 else '7S', 1180+index) for index in range(7)])
    selected, _ = budget.take(1200)
    assert [value.sequence for value in selected] == [2, 3, 4, 5, 6]
    assert budget.summary()['focus_available_packets'] == {'7X': 1}
    assert budget.selection_checks_pass()


def test_original_time_window_dt_terminal_and_discard_rules_are_retained():
    old, new = c.formal.PacketBudget(packet_protocol()), c.LineCoveragePacketBudget(packet_protocol())
    rows = [row(index, '7S', 1180+index) for index in range(6)]
    rows += [row(100, '102X', 900), row(101, '705X', 1195, dt=0),
        row(102, '7X', 1196, terminal=True), row(103, '102X', 1201)]
    for budget in (old, new):
        budget.ingest(rows)
    a, discarded_a = old.take(1203)
    b, discarded_b = new.take(1203)
    assert [value.sequence for value in a] == [value.sequence for value in b] == list(range(1, 6))
    assert discarded_a == discarded_b
    assert [value.sequence for value in new.rows] == [value.sequence for value in old.rows] == [103]
    assert new.summary()['focus_available_packets'] == {}
    assert new.receipts[0]['max_available_at'] <= 1200


def test_fewer_than_five_keeps_original_failure_without_wait_or_cross_window():
    for cls in (c.formal.PacketBudget, c.LineCoveragePacketBudget):
        budget = cls(packet_protocol())
        budget.ingest([row(index, available=1180+index) for index in range(4)]+[row(9, available=1201)])
        with pytest.raises(RuntimeError, match='4 eligible, 5 required'):
            budget.take(1202)
        assert budget.index == 0 and budget.receipts == []


class OutcomeDeniedRow:
    def __init__(self, index, line, available):
        self.sequence, self.line_id, self.available_at = index, line, available
        self.started_at, self.dt, self.terminal = available-35., 5., False
        self.s, self.a = table(.7)['s'][0], np.array([-.5], np.float32)

    def __getattr__(self, field):
        raise AssertionError(f'Unexpected outcome read before/no-fresh admission: {field}')


def test_selector_and_no_fresh_replay_never_read_rewards_or_next_states():
    budget = c.LineCoveragePacketBudget(packet_protocol())
    budget.ingest([OutcomeDeniedRow(index, c.FOCUS_LINES[index % 3], 1180+index)
        for index in range(9)])
    selected, _ = budget.take(1200)
    replay = c.FreshOutcomeReplay(table(.1), table(.4), TransitMDPConfigV7(),
        np.full(12, 40, np.float32), arm='no_fresh', window=285)
    replay.add_target(selected)
    assert len(selected) == replay.target_received == 5
    assert all(set(value) == {'s', 'a'} for value in replay.recent)
    target, source = replay.sample_mixed(256, .5, rng=np.random.default_rng(11))
    assert len(target['actions']) == len(source['actions']) == 128
    replacement = replay.last_sampling['replacement']
    for batch_key, table_key in c.FIELDS.items():
        expected = torch.as_tensor(replay.history[table_key][replacement], dtype=torch.float32)
        torch.testing.assert_close(target[batch_key][64:], expected.reshape_as(target[batch_key][64:]),
            rtol=0, atol=0)


def test_fresh_and_no_fresh_pay_the_same285_slots_and_keep_four_native_draws():
    budgets = [c.LineCoveragePacketBudget(packet_protocol()) for _ in range(2)]
    replays = [c.FreshOutcomeReplay(table(.1), table(.4), TransitMDPConfigV7(),
        np.full(12, 40, np.float32), arm=arm, window=285) for arm in ('fresh', 'no_fresh')]
    samplers = [np.random.default_rng(77), np.random.default_rng(77)]
    data = table(.7, count=12)
    updates = 0
    for packet, deadline in enumerate(budgets[0].deadlines):
        rows = []
        for index in range(12):
            value = {key: data[key][index] for key in c.FIELDS.values()}
            value.update(sequence=packet*12+index, line_id=c.FOCUS_LINES[index] if index < 3 else '7S',
                started_at=deadline-50+index, available_at=deadline-20+index)
            rows.append(SimpleNamespace(**value))
        for budget, replay in zip(budgets, replays):
            budget.ingest(rows)
            selected, _ = budget.take(deadline)
            replay.add_target(selected)
        assert budgets[0].receipts[-1] == budgets[1].receipts[-1]
        for _ in range(8):
            for replay, rng in zip(replays, samplers):
                replay.sample_mixed(256, .5, rng=rng)
            assert replays[0].last_sampling == replays[1].last_sampling
            assert samplers[0].bit_generator.state == samplers[1].bit_generator.state
            updates += 1
    assert updates == 456
    for budget, replay in zip(budgets, replays):
        assert budget.index == len(budget.receipts) == 57
        assert replay.target_received == len(replay.recent) == replay.recent.maxlen == 285
        assert budget.selection_checks_pass()
        assert sum(budget.summary()['selected_record_counts_by_line'].values()) == 285
        ids = [sequence for receipt in budget.receipts for sequence in receipt['sequence_ids']]
        assert len(ids) == len(set(ids)) == 285
        assert budget.summary()['label_age_seconds']['count'] == 285
        assert not hasattr(replay, 'outcome_rng')


def initializer_files(tmp_path, monkeypatch):
    (tmp_path/'history_metadata.json').write_text(json.dumps(dict(
        contract=asdict(TransitMDPConfigV7()), station_max_by_line=[40]*12)))
    np.savez(tmp_path/'history.npz', **table(.1))
    monkeypatch.setattr(c.resident, 'INPUTS', tmp_path)
    monkeypatch.setattr(c, 'load_table', lambda *args: table(.4))


@pytest.mark.parametrize('change', [dict(total_updates=49999), dict(online_critic_updates=1),
    dict(extra=dict(source_training_seed=6201))])
def test_initializer_never_substitutes_another_seed_or_update_count(tmp_path, monkeypatch, change):
    initializer_files(tmp_path, monkeypatch)
    payload = dict(total_updates=50000, online_critic_updates=0, extra=dict(source_training_seed=6203))
    payload.update(change)
    requested = []

    def load(path, **kwargs):
        requested.append(path.name)
        return payload

    monkeypatch.setattr(c.torch, 'load', load)
    with pytest.raises(ValueError, match='Wrong read-only initializer'):
        c.load_learner(6203, arm='fresh')
    assert requested == ['h2oplus_seed6203.pt']


def test_missing_initializer_does_not_fallback(tmp_path, monkeypatch):
    initializer_files(tmp_path, monkeypatch)
    requested = []

    def missing(path, **kwargs):
        requested.append(path.name)
        raise FileNotFoundError(path)

    monkeypatch.setattr(c.torch, 'load', missing)
    with pytest.raises(FileNotFoundError):
        c.load_learner(6204, arm='no_fresh')
    assert requested == ['h2oplus_seed6204.pt']
