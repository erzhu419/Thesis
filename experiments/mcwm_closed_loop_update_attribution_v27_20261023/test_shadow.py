"""Exercise actual resident networks/Adam updates rather than a mirrored trainer."""

from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import (
    FIELDS, make_trainer, table)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _equal, _rng, _state, compare_after_main, run_shadow_packet)


def packet():
    data = table(.71, count=5)
    return [SimpleNamespace(**{key: data[key][index] for key in FIELDS.values()},
                            line_id="7X", station_id="stop", sequence=index) for index in range(5)]


def test_shadow_leaves_origin_and_global_rng_unchanged_and_fresh_exactly_replays_real_packet():
    learner = make_trainer("fresh")
    # Nonempty Adam state is essential: a fresh optimizer copy is not equivalent.
    for _ in range(3):
        learner.continue_step()
    rows = packet()
    learner.replay_buffer.add_target(rows)
    initial, global_rng = deepcopy(_state(learner)), _rng()
    replay_initial = deepcopy(list(learner.replay_buffer.recent))
    record, reference = run_shadow_packet(learner, rows, dict(seed=6202, deadline=9300))
    assert _equal(initial, _state(learner))
    assert _equal(global_rng, _rng())
    assert _equal(replay_initial, list(learner.replay_buffer.recent))
    assert all(record["integrity"].values())
    for step in range(8):
        learner.continue_step(collect_diagnostics=step == 0)
    result = compare_after_main(learner, reference)
    assert result["exact"], result
    assert record["diagnostic_budget"]["shadow_optimizer_updates"] == 16
    assert record["arms"]["no_fresh"]["replay"]["fresh_outcomes_stored"] == 0
    assert record["arms"]["fresh"]["replay"]["fresh_outcomes_stored"] == 14


def test_fixed_state_bellman_and_execution_decompositions_use_actual_model_paths():
    learner = make_trainer("fresh")
    rows = packet()
    learner.replay_buffer.add_target(rows)
    record, reference = run_shadow_packet(learner, rows)
    for index, row in enumerate(record["probe_rows"]):
        assert row["has_transition"] and row["line_id"] == "7X"
        for name in ("before", "fresh", "no_fresh"):
            point = row[name]
            assert point["reward_scaled"] + point["bootstrap_contribution"] == pytest.approx(
                point["bellman_target"], rel=1e-6, abs=1e-6)
            assert point["reward_scaled"] == pytest.approx(float(rows[index].r)*learner.config.reward_scale)
            assert 0 <= point["holding_seconds"] <= learner.action_transform.max_hold
        states = torch.tensor(np.asarray([rows[index].s]), dtype=torch.float32)
        proposal = reference.learner.actor(states, deterministic=True)[0]
        executed = reference.learner.action_transform(states, proposal)
        expected_q = reference.learner.critic(states, executed).mean()
        assert row["fresh"]["q_on_post_actor_mean"] == pytest.approx(float(expected_q.detach()))
        expected_hold = reference.learner.action_transform.hold_seconds(executed)
        assert row["fresh"]["holding_seconds"] == pytest.approx(float(expected_hold.detach()))
    for arm in ("fresh", "no_fresh"):
        steps = record["arms"][arm]["training_steps"]
        assert len(steps) == 8
        for step in steps:
            assert [group["n"] for group in step["batches"]] == [2, 2, 4]
            for group in step["batches"]:
                assert group["reward_scaled"]["mean"] + group["bootstrap_contribution"]["mean"] == pytest.approx(
                    group["bellman_target"]["mean"], rel=1e-6, abs=1e-6)


def test_extra_real_tail_states_preserve_clock_and_are_not_ingested():
    learner = make_trainer("fresh")
    rows = packet()
    learner.replay_buffer.add_target(rows)
    extra = table(.78, count=2)["s"]
    record, reference = run_shadow_packet(learner, rows, probe_states=extra,
        probe_metadata=[dict(role="tail_heldout_state", line_id="7X"),
                        dict(role="tail_heldout_state", line_id="102X")])
    assert reference.learner.replay_buffer.target_received == learner.replay_buffer.target_received == 14
    assert len(record["probe_rows"]) == 7
    for index, row in enumerate(record["probe_rows"][5:]):
        assert not row["has_transition"] and row["role"] == "tail_heldout_state"
        assert row["issue_time"] == float(extra[index, 0])*24000
        assert row["before"]["bellman_target"] is None
        assert row["fresh"]["q_observed_mean"] is None
        assert row["fresh"]["holding_seconds"] == pytest.approx(row["fresh"]["proposal_seconds"]*.125)


def test_comparison_rejects_changed_optimizer_moments_and_no_fresh_origin():
    learner = make_trainer("fresh")
    rows = packet()
    learner.replay_buffer.add_target(rows)
    _, reference = run_shadow_packet(learner, rows)
    for _ in range(8):
        learner.continue_step()
    assert compare_after_main(learner, reference)["exact"]
    state = next(iter(learner.critic_optimizer.state.values()))
    state["exp_avg"].add_(.1)
    assert not compare_after_main(learner, reference)["checks"]["optimizers"]
    with pytest.raises(ValueError, match="fresh-origin"):
        run_shadow_packet(make_trainer("no_fresh"), rows)
