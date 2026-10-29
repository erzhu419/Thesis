"""Catch changed covariates/RNG, hybrid labels and leakage in recent30 control."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_control_campaign_v60_20261025.control import (
    FIELDS, FreshOutcomeReplay, RecentOutcomeReplay, formal, protocol)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import (
    make_trainer, table)
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def replay(arm, *, recent=False, count=285):
    cls = RecentOutcomeReplay if recent else FreshOutcomeReplay
    kwargs = dict(outcome_seed=91230) if recent else {}
    value = cls(table(.1), table(.4), TransitMDPConfigV7(),
        np.full(12, 40, np.float32), arm=arm, **kwargs)
    data = table(.7, count=count)
    rows = [SimpleNamespace(**{key: data[key][index] for key in FIELDS.values()})
        for index in range(count)]
    value.add_target(rows)
    return value, data


def exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert left == right


def test_only_fresh_complete_outcome_quarter_is_resampled_from_last30():
    old, _ = replay('fresh')
    new, data = replay('fresh', recent=True)
    old_rng, new_rng = np.random.default_rng(904), np.random.default_rng(904)
    old_batch, old_source = old.sample_mixed(256, .5, rng=old_rng)
    batch, source = new.sample_mixed(256, .5, rng=new_rng)
    assert new.target_received == len(new.recent) == new.recent.maxlen == 285
    assert old.last_sampling == new.last_sampling
    assert old_rng.bit_generator.state == new_rng.bit_generator.state
    exact(old_source, source)
    exact(old.last_covariates, new.last_covariates)
    assert len(source['actions']) == 128 and len(batch['actions']) == 128
    for key in batch:
        exact(old_batch[key][:64], batch[key][:64])
    # Whole fresh rows move together; older s/a remain available to actor/V.
    ids = 255+np.asarray(new.last_outcome_sampling['local_indices'])
    for batch_key, table_key in FIELDS.items():
        expected = torch.as_tensor(data[table_key][ids], dtype=torch.float32)
        exact(batch[batch_key][64:], expected.reshape_as(batch[batch_key][64:]))
    assert min(new.last_sampling['recent']) < 255
    assert new.maximum_sampled_label_age <= 29


def test_no_fresh_never_reads_outcomes_and_exactly_retains_original_batch():
    class VisibleIssue:
        s, a = table(.7)['s'][0], np.array([.3], np.float32)

        def __getattr__(self, name):
            raise AssertionError(f'no_fresh read outcome field: {name}')

    old, _ = replay('no_fresh')
    new, _ = replay('no_fresh', recent=True)
    old.add_target([VisibleIssue()])
    new.add_target([VisibleIssue()])
    assert all(set(row) == {'s', 'a'} for row in new.recent)
    old_rng, new_rng = np.random.default_rng(12), np.random.default_rng(12)
    old_batch, old_source = old.sample_mixed(256, .5, rng=old_rng)
    batch, source = new.sample_mixed(256, .5, rng=new_rng)
    exact(old_batch, batch)
    exact(old_source, source)
    exact(old.last_covariates, new.last_covariates)
    assert old_rng.bit_generator.state == new_rng.bit_generator.state
    assert new.summary()['fresh_outcomes_stored'] == 0


@pytest.mark.parametrize('count', [5, 15, 30, 35, 285])
def test_private_outcome_draw_advances_equally_in_both_arms_without_global_rng(count):
    fresh, _ = replay('fresh', recent=True, count=count)
    old, _ = replay('no_fresh', recent=True, count=count)
    left, right = np.random.default_rng(18), np.random.default_rng(18)
    torch_before = torch.get_rng_state().clone()
    for _ in range(8):
        fresh.sample_mixed(256, .5, rng=left)
        old.sample_mixed(256, .5, rng=right)
    assert left.bit_generator.state == right.bit_generator.state
    assert fresh.outcome_rng.bit_generator.state == old.outcome_rng.bit_generator.state
    assert fresh.last_outcome_sampling == old.last_outcome_sampling
    assert fresh.outcome_sample_calls == old.outcome_sample_calls == 8
    assert fresh.last_outcome_sampling['window'] == min(count, 30)
    assert torch.equal(torch_before, torch.get_rng_state())


def test_no_fresh_actual_eight_updates_preserve_parameters_adam_and_rng():
    old = make_trainer('no_fresh')
    new = make_trainer('no_fresh')
    replacement = RecentOutcomeReplay(table(.1), table(.4),
        TransitMDPConfigV7(), np.full(12, 40, np.float32), arm='no_fresh', outcome_seed=712)
    data = table(.7)
    replacement.add_target([SimpleNamespace(**{key: data[key][index]
        for key in FIELDS.values()}) for index in range(len(data['s']))])
    new.replay_buffer = replacement
    for step in range(8):
        before = torch.get_rng_state().clone()
        old.continue_step(collect_diagnostics=step == 0)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        new.continue_step(collect_diagnostics=step == 0)
        assert torch.equal(after, torch.get_rng_state())
    for name in ('actor', 'value', 'critic', 'target_critic'):
        exact(getattr(old, name).state_dict(), getattr(new, name).state_dict())
    for name in ('actor', 'value', 'critic', 'alpha'):
        exact(getattr(old, name+'_optimizer').state_dict(),
            getattr(new, name+'_optimizer').state_dict())
    exact(old.log_alpha, new.log_alpha)
    assert old.rng.bit_generator.state == new.rng.bit_generator.state
    assert old.total_updates == new.total_updates == 8
    assert new.replay_buffer.outcome_sample_calls == 8
    assert asdict(old.config) == asdict(new.config)


def test_original_packet_budget_still_pays285_and_drives456_sampler_calls():
    p = protocol()
    budget = formal.PacketBudget(p)
    value, _ = replay('fresh', recent=True, count=0)
    rng = np.random.default_rng(901)
    data = table(.7, count=285)
    index = 0
    for deadline in budget.deadlines:
        rows = []
        for offset in range(5):
            values = {key: data[key][index] for key in FIELDS.values()}
            values.update(sequence=index, available_at=deadline-5+offset)
            rows.append(SimpleNamespace(**values))
            index += 1
        budget.ingest(rows)
        packet, _ = budget.take(deadline)
        value.add_target(packet)
        for _ in range(p['policy_updates_per_packet']):
            value.sample_mixed(256, .5, rng=rng)
    assert budget.index == len(budget.receipts) == 57
    assert value.target_received == len(value.recent) == 285
    assert value.outcome_sample_calls == 456
    assert sum(row['records'] for row in budget.receipts) == 285
    assert len({seq for row in budget.receipts for seq in row['sequence_ids']}) == 285
    assert value.summary()['maximum_fresh_outcome_age_in_admitted_labels'] <= 29
