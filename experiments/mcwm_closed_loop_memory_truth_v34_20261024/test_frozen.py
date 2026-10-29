"""Small native reconstruction, old-prediction gates and frozen-score isolation."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import attach_memory, critic_states
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial, rows
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.probes import probe, window
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _state, _rng, _restore
from experiments.mcwm_closed_loop_memory_truth_v34_20261024 import frozen


PROTOCOL = dict(focus_lines=["7X", "102X", "705X"], diagnostic_seed=336202)


def packets():
    result = []
    for index, clock in enumerate((1500., 10000., 15000.), 1):
        packet_rows = rows(.1+clock/24000.)
        for offset, row in enumerate(packet_rows):
            row.sequence = (index-1)*5+offset
            row.line_id = PROTOCOL["focus_lines"][offset%3]
            row.started_at = clock+offset
            row.s[0], row.ns[0] = row.started_at/24000., (row.started_at+row.dt)/24000.
            row.memory_s["observed_at"] = row.started_at
            row.memory_ns["observed_at"] = row.started_at+row.dt
        result.append(dict(index=index, receipt=dict(processed_at=clock+100.), rows=packet_rows))
    return result


def record(learner, data, steps=2):
    reference, chunks, bank, seen = [], [], [], set()
    initial_updates = learner.total_updates
    for packet in data:
        learner.replay_buffer.add_target(packet["rows"])
        sampling = []
        for step in range(steps):
            metrics = learner.continue_step(collect_diagnostics=step == 0)
            sampling.append(deepcopy(learner.replay_buffer.last_sampling))
            if step == 0:
                first = dict(packet=packet["index"], processed_at=packet["receipt"]["processed_at"], **metrics)
        reference.append(dict(packet=packet["index"], receipt=packet["receipt"], metrics=frozen.plain(first),
            sampling=sampling, rng=dict(global_state=frozen.plain(_rng()),
                sampler=frozen.plain(learner.rng.bit_generator.state),
                diagnostic=frozen.plain(learner.diagnostic_rng.bit_generator.state)),
            optimizer_steps=frozen.optimizer_steps(learner), updates=learner.total_updates-initial_updates))
        for row in packet["rows"]:
            key = row.line_id, window(row.started_at)
            if key not in seen:
                seen.add(key)
                bank.append(row)
        for phase, scored in (("packet_after", packet["rows"]), ("bank_after", bank)):
            chunks.append(dict(packet=packet["index"], phase=phase,
                rows=probe(learner, scored, seed=PROTOCOL["diagnostic_seed"])))
    # Exercise the exact JSON representation used by server-resident records.
    return json.loads(json.dumps(reference)), json.loads(json.dumps(chunks))


@pytest.mark.parametrize("memory", [False, True])
def test_native_multi_packet_reconstruction_and_old_predictions_are_exact(memory):
    origin = initial()
    if memory:
        attach_memory(origin)
    expected, actual = deepcopy(origin), deepcopy(origin)
    data, start = packets(), _rng()
    reference, scores = record(expected, data)
    end = _rng()
    _restore(start)
    proof = frozen._rebuild_arm(actual, data, reference, steps_per_packet=2)
    assert proof["all_pass"] and proof["reconstruction_optimizer_updates"] == 6
    assert _equal(end, _rng())
    assert _equal(_state(expected), _state(actual))
    prediction = frozen._final_predictions(actual, data, scores, PROTOCOL)
    assert prediction["old_final_prediction_rows"] == 14
    assert _equal(end, _rng())
    frozen.freeze(actual)
    assert not actual.log_alpha.requires_grad and actual.log_alpha.grad is None
    for name in ("actor", "critic", "target_critic", "value", "ratio"):
        assert not getattr(actual, name).training
        assert all(not p.requires_grad for p in getattr(actual, name).parameters())
        assert all(p.grad is None for p in getattr(actual, name).parameters())
    assert _equal(_state(expected), _state(actual))
    assert frozen._final_predictions(actual, data, scores, PROTOCOL)["all_pass"]


def test_reconstruction_rejects_changed_sampling_rng_and_old_prediction():
    origin, data = initial(), packets()
    start = _rng()
    reference, scores = record(deepcopy(origin), data)
    for field in ("sampling", "rng"):
        changed = deepcopy(reference)
        changed[0][field] = []
        _restore(start)
        with pytest.raises(RuntimeError, match="Rebuilt packet 1 differs"):
            frozen._rebuild_arm(deepcopy(origin), data, changed, steps_per_packet=2)
    _restore(start)
    actual = deepcopy(origin)
    frozen._rebuild_arm(actual, data, reference, steps_per_packet=2)
    changed = deepcopy(scores)
    next(chunk for chunk in changed if chunk["packet"] == 3 and chunk["phase"] == "bank_after")["rows"][0]["q_min"][0] += 1.
    with pytest.raises(RuntimeError, match="old bank_after predictions differ"):
        frozen._final_predictions(actual, data, changed, PROTOCOL)


def learners():
    zero = initial()
    memory = attach_memory(deepcopy(zero))
    with torch.no_grad():
        memory.critic.action_weight.fill_(.3)
        memory.critic.level_weight.fill_(.03)
    return {"zero_memory": frozen.freeze(zero), "memory": frozen.freeze(memory)}


def test_explicit_frozen_context_changes_only_memory_q_and_never_transforms_actions_twice():
    pair = learners()
    state = packets()[-1]["rows"][0].s
    context = np.full(84, .4, np.float32)
    actions = [-1., -.9375]
    before = {arm: deepcopy(_state(learner)) for arm, learner in pair.items()}
    global_before = _rng()
    private = {arm: deepcopy(learner.diagnostic_rng.bit_generator.state) for arm, learner in pair.items()}
    scored = frozen.score_frozen(pair, state, context, actions)
    masked = frozen.score_frozen(pair, state, np.zeros_like(context), actions)
    assert _equal(global_before, _rng())
    assert scored["zero_memory"] == masked["zero_memory"]
    assert scored["memory"]["q_min"] != masked["memory"]["q_min"]
    assert scored["memory"]["holding_seconds"] == [0., 1.875]
    states = torch.tensor(state)[None].expand(2, -1)
    explicit = critic_states(pair["memory"], states, torch.tensor(context)[None].expand(2, -1))
    with torch.no_grad():
        expected = pair["memory"].critic(explicit, torch.tensor(actions)[:, None])
    assert scored["memory"]["q_min"] == expected.min(0).values.tolist()
    assert scored["memory"]["q_mean"] == expected.mean(0).tolist()
    for arm, learner in pair.items():
        assert _equal(before[arm], _state(learner))
        assert _equal(private[arm], learner.diagnostic_rng.bit_generator.state)
        assert all(parameter.grad is None or not parameter.requires_grad
                   for parameter in learner.critic.parameters())


def test_resident_full_learner_cache_roundtrip_preserves_models_and_caller_rng(tmp_path, monkeypatch):
    pair = learners()
    proof = dict(all_pass=True, budget=dict(reconstruction_optimizer_updates=912))
    with (tmp_path/"final_models.pt").open("xb") as handle:
        torch.save(dict(seed=6202, scenario="stable", learners=pair, contract="native_contract",
            stations=np.arange(12), proof=proof), handle)
    monkeypatch.setattr(frozen, "make_learner", lambda *args, **kwargs: pytest.fail("Cache retrained models"))
    before = _rng()
    loaded, contract, stations, loaded_proof = frozen.build_frozen(6202, "stable", tmp_path)
    assert _equal(before, _rng())
    assert contract == "native_contract" and loaded_proof == proof
    assert np.array_equal(stations, np.arange(12))
    for arm in pair:
        assert _equal(_state(pair[arm]), _state(loaded[arm]))
        assert all(not parameter.requires_grad for parameter in loaded[arm].critic.parameters())
