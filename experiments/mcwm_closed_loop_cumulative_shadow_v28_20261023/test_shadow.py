"""Exercise retained real networks, Adam, RNG and shared executed candidates."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import (
    FIELDS, make_trainer, table)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _equal, _restore, _rng, _state)
from experiments.mcwm_closed_loop_cumulative_shadow_v28_20261023.shadow import (
    CANDIDATES, CumulativeShadow)


def packet(index):
    data = table(.71+.01*index, count=5)
    return [SimpleNamespace(**{key: data[key][row] for key in FIELDS.values()},
                            line_id="7X", sequence=index*5+row) for row in range(5)]


def bank():
    states = table(.72, count=3)["s"]
    states[1, 0] = .7
    # Reachable completion floor: forward bus present, backward absent, light load.
    slots = states[1, 5:].reshape(16, 14)
    slots[0, 12] = 0
    states[2, 0] = .78
    metadata = [dict(bank_id=str(index), role="fixed_reference_state", line_id=line,
                     issue_time=float(state[0])*24000.)
                for index, (line, state) in enumerate(zip(("7X", "102X", "705X"), states))]
    return states, metadata


def test_two_packet_adam_and_rng_are_retained_exactly_without_changing_main():
    main = make_trainer("fresh")
    for _ in range(3):
        main.continue_step()
    rows = packet(0)
    main.replay_buffer.add_target(rows)
    shadow = CumulativeShadow(main, *bank())
    # Independent continuing no-fresh learner verifies that neither Adam nor its
    # divergent parameters are silently reinitialized at the second packet.
    manual = deepcopy(main)
    manual.outcome_arm = manual.replay_buffer.arm = "no_fresh"
    manual.replay_buffer.recent = deque(({key: row[key] for key in ("s", "a")}
                                       for row in manual.replay_buffer.recent), maxlen=285)
    manual_rng = _rng()
    for index in range(3):
        if index:
            rows = packet(index)
            main.replay_buffer.add_target(rows)
            manual.replay_buffer.add_target(rows)
        initial_main, initial_rng = deepcopy(_state(main)), _rng()
        initial_replay = deepcopy(list(main.replay_buffer.recent))
        record = shadow.advance(main, rows, dict(deadline=9000+index*300))
        assert all(record["integrity"].values()), record["integrity"]
        assert _equal(initial_main, _state(main))
        assert _equal(initial_rng, _rng())
        assert _equal(initial_replay, list(main.replay_buffer.recent))
        _restore(manual_rng)
        for _ in range(8):
            manual.continue_step()
        manual_rng = _rng()
        _restore(initial_rng)
        assert _equal(_state(manual), _state(shadow.learners["no_fresh"]))
        assert _equal(manual_rng, shadow.private_rng["no_fresh"])
        for step in range(8):
            main.continue_step(collect_diagnostics=step == 0)
        comparison = shadow.compare_after_main(main)
        assert comparison["exact"], comparison
        assert record["cumulative_updates"] == (index+1)*8
        assert record["arms"]["fresh"]["replay"]["feedback_slots_received"] == 14+index*5
        assert record["arms"]["no_fresh"]["replay"]["fresh_outcomes_stored"] == 0
        before_act = _rng()
        main.act(bank()[0], deterministic=True)
        assert _equal(before_act, _rng())
    assert not _equal(main.critic.state_dict(), manual.critic.state_dict())
    assert shadow.learners["fresh"].total_updates == main.total_updates == 27
    assert all(set(row) == {"s", "a"} for row in manual.replay_buffer.recent)


def test_candidates_are_identical_physical_actions_for_both_critics_and_keep_real_clocks():
    main = make_trainer("fresh")
    rows = packet(0)
    main.replay_buffer.add_target(rows)
    states, metadata = bank()
    shadow = CumulativeShadow(main, states, metadata)
    record = shadow.advance(main, rows)
    assert record["candidates"] == CANDIDATES
    assert len(record["initial_probe_rows"]) == len(states)
    for index, row in enumerate(record["probe_rows"]):
        assert row["canonical_clock"] == float(states[index, 0])*24000.
        assert row["issue_time"] == metadata[index]["issue_time"]
        assert row["proposal_action"][0] == record["initial_probe_rows"][index]["proposal_action"][0]
        assert row["proposal_seconds"][3:] == pytest.approx([0., 7.5], abs=1e-5)
        tensor_states = torch.tensor(np.repeat(states[index:index+1], 5, axis=0))
        actions = torch.tensor(row["proposal_action"])[:, None]
        for arm, learner in shadow.learners.items():
            with torch.no_grad():
                executed = learner.action_transform(tensor_states, actions)
                q = learner.critic(tensor_states, executed)
            assert row["executed_action"] == pytest.approx(executed[:, 0].tolist())
            assert row["q_mean"][arm] == pytest.approx(q.mean(dim=0).tolist(), abs=1e-7)
            assert row["q_min"][arm] == pytest.approx(q.min(dim=0).values.tolist(), abs=1e-7)
        if index == 1:
            assert row["holding_seconds"] == pytest.approx([60.]*5)
            for arm in ("fresh", "no_fresh"):
                assert row["q_mean"][arm] == pytest.approx([row["q_mean"][arm][0]]*5)
        if index == 2:
            assert row["holding_seconds"] == pytest.approx(np.array(row["proposal_seconds"])*.125, abs=5e-6)
    for arm in ("fresh", "no_fresh"):
        for step in record["arms"][arm]["training_steps"]:
            assert [batch["n"] for batch in step["batches"]] == [2, 2, 4]
            for batch in step["batches"]:
                assert batch["reward_scaled"]["mean"]+batch["bootstrap_contribution"]["mean"] == pytest.approx(
                    batch["bellman_target"]["mean"], abs=1e-6)


def test_comparison_detects_changed_adam_and_rejects_nonfresh_origin():
    main = make_trainer("fresh")
    rows = packet(0)
    main.replay_buffer.add_target(rows)
    shadow = CumulativeShadow(main, *bank())
    shadow.advance(main, rows)
    for _ in range(8):
        main.continue_step()
    assert shadow.compare_after_main(main)["exact"]
    next(iter(main.critic_optimizer.state.values()))["exp_avg"].add_(.1)
    assert not shadow.compare_after_main(main)["checks"]["optimizers"]
    with pytest.raises(ValueError, match="fresh-origin"):
        CumulativeShadow(make_trainer("no_fresh"), *bank())
