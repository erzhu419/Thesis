"""Matched budget, initial state, and causal context tests without SUMO runs."""

from collections import deque
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("formal_mcwm_run", HERE / "run.py")
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


def state():
    value = np.zeros(229, np.float32)
    value[0], value[1], value[5] = .1, 1 / 16, 1
    return value


def row(sequence, available_at, *, dt=4., terminal=False, reward=-8.):
    return SimpleNamespace(sequence=sequence, available_at=float(available_at),
        completed_at=float(available_at)-30, started_at=float(available_at)-30-dt,
        s=state(), a=np.asarray([0.], np.float32), ns=state(), r=reward, dt=dt,
        terminal=terminal)


def test_every_arm_gets_285_unique_records_from_same_prespecified_windows():
    selections = []
    for _arm in PROTOCOL["arms"]:
        budget = formal.PacketBudget(PROTOCOL)
        assert len(budget.deadlines) == 57
        with pytest.raises(ValueError, match="not arrived"):
            budget.take(1199)
        received = []
        for index, deadline in enumerate(budget.deadlines):
            first = index * 20
            eligible = [row(first + j, deadline - 100 + j) for j in range(8)]
            excluded = [row(first + 8, deadline, dt=0),
                        row(first + 9, deadline, terminal=True),
                        row(first + 10, deadline - 300)]
            future = row(first + 11, 24001)
            budget.ingest(eligible + excluded + [future])
            packet, discarded = budget.take(deadline + 2)
            assert [item.sequence for item in packet] == [first + j for j in range(3, 8)]
            assert set(discarded) == {first, first + 1, first + 2, first + 8, first + 9, first + 10}
            assert all(deadline - 300 < item.available_at <= deadline for item in packet)
            assert all(item.completed_at <= deadline and item.dt > 0 and not item.terminal for item in packet)
            received.extend(item.sequence for item in packet)
        assert len(received) == len(set(received)) == PROTOCOL["target_record_budget"] == 285
        assert len(received) // 5 * PROTOCOL["policy_updates_per_packet"] == PROTOCOL["policy_update_budget"]
        assert all(item.available_at > 18000 for item in budget.rows)
        assert all(receipt["processed_at"] == receipt["deadline"] + 2 for receipt in budget.receipts)
        assert not budget.due(24000)
        selections.append(received)
    assert selections[0] == selections[1] == selections[2]


def test_short_packet_fails_without_borrowing_future_or_invalid_outcomes():
    budget = formal.PacketBudget(PROTOCOL)
    records = [row(j, 1190 + j) for j in range(4)]
    records += [row(4, 1201), row(5, 1200, dt=0), row(6, 1200, terminal=True)]
    budget.ingest(records)
    # A late event-clock processing time does not enlarge deadline eligibility.
    with pytest.raises(RuntimeError, match="4 eligible, 5 required"):
        budget.take(1250)
    assert budget.index == 0 and budget.receipts == []
    assert budget.rows == records


def small_table():
    return dict(s=np.stack([state()] * 3), a=np.zeros((3, 1), np.float32),
                ns=np.stack([state()] * 3), r=np.asarray([-1., -2., -900000.]),
                dt=np.asarray([1., 2., 2900.]), terminal=np.asarray([0, 0, 1], np.uint8))


def test_formal_replay_passes_original_terminal_mass_to_weighted_draw():
    history = small_table()
    # The retained table has one rare tail, representing one of 100 original rows.
    history["sample_weight"] = np.asarray([49.5, 49.5, 1.0])
    replay = formal.FormalReplay(history, small_table(), formal.TransitMDPConfigV7(),
                                np.full(12, 40, np.float32), "cpu")
    captured = {}

    class Draw:
        def choice(self, population, *, size, p):
            captured.update(population=population, size=size, probability=p.copy())
            return np.asarray([0, 2])

    batch = replay.sample(2, rng=Draw())
    assert captured["probability"][2] == pytest.approx(.01)
    np.testing.assert_allclose(captured["probability"], [.495, .495, .01])
    assert "sample_weight" not in replay.history
    assert "sample_weight" in history  # Constructing an arm does not mutate common input.
    assert batch["terminals"].tolist() == [0., 1.]
    assert batch["rewards"].tolist() == [-1., -900000.]


class ConstantHead(torch.nn.Module):
    def __init__(self, global_value):
        super().__init__()
        output = torch.zeros(1124)
        output[:4] = global_value
        self.register_buffer("output", output)

    def forward(self, states, actions):
        return self.output[None].expand(len(states), -1)


def world(*, context=True, fixed_reference=False):
    result = formal.ScoredWorld.__new__(formal.ScoredWorld)
    result.device = "cpu"
    result.stations = torch.full((12,), 40.)
    result.norm = SimpleNamespace(g_mean=torch.zeros(4), g_std=torch.ones(4),
                                  local_mean=torch.zeros(8), local_std=torch.ones(8))
    result.models = dict(direct=ConstantHead(10.), residual=ConstantHead(0.))
    result.context = {family: torch.zeros(4) for family in result.models}
    result.errors = {family: deque(maxlen=128) for family in result.models}
    result.pending, result.scored = {}, []
    result.family, result.version = "direct", 0
    result.context_enabled, result.fixed_reference = context, fixed_reference
    return result


def register(model, sequence, at=10):
    model.register(dict(sequence=sequence, s=state(), a=np.asarray([0.], np.float32), started_at=at))


def test_prequential_score_uses_family_at_issue_before_truth_changes_selection():
    model = world(context=False)
    register(model, 1)
    record = row(1, 100)
    result = model.observe([record], 100)[0]
    assert model.family == "residual"
    assert result["selected_family_at_issue"] == "direct"
    assert result["selected_score"] == result["scores"]["direct"]
    assert result["selected_score"] > result["scores"]["residual"]
    frozen = world(context=False, fixed_reference=True)
    register(frozen, 1)
    frozen.observe([record], 100)
    assert frozen.family == "direct"


def test_context_after_refit_affects_future_prediction_and_keeps_saved_prediction():
    model = world()
    register(model, 1)
    saved = model.pending[1]["outputs"]["residual"].clone()
    record = row(1, 100)
    model.observe([record], 100)
    assert all(torch.equal(value, torch.zeros(4)) for value in model.context.values())
    # Refit clears contexts in v1; formal packet calibration runs afterwards.
    for value in model.context.values():
        value.zero_()
    model.recenter_context([record])
    expected = .1 * torch.tensor([np.log1p(record.dt), np.log1p(-record.r), 0., 0.])
    torch.testing.assert_close(model.context["residual"], expected.float())
    register(model, 2, at=101)
    torch.testing.assert_close(model.pending[2]["outputs"]["residual"][0, :4], expected.float())
    assert torch.equal(saved, torch.zeros_like(saved))
    assert model.scored[0]["model_version"] == 0


def test_future_label_cannot_update_selection_context_or_saved_predictions():
    model = world()
    register(model, 1)
    before = model.pending[1]["outputs"]["direct"].clone()
    with pytest.raises(ValueError, match="before it was available"):
        model.observe([row(1, 101)], 100)
    assert model.family == "direct" and model.scored == []
    assert all(torch.equal(value, torch.zeros(4)) for value in model.context.values())
    torch.testing.assert_close(model.pending[1]["outputs"]["direct"], before)


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    folder = tmp_path / "experiment"
    inputs = folder / "inputs"
    inputs.mkdir(parents=True)
    table = small_table()
    np.savez(inputs / "history.npz", **table, sample_weight=np.full(3, 1 / 3))
    np.savez(inputs / "source_9101.npz", **table)
    (inputs / "history_metadata.json").write_text(json.dumps(dict(
        contract={}, station_max_by_line=[40.] * 12)))
    protocol = folder / "protocol.json"
    protocol.write_text(json.dumps(PROTOCOL))
    monkeypatch.setattr(formal, "HERE", folder)
    payload = dict(total_updates=50000, online_critic_updates=0,
                   extra=dict(source_training_seed=6201), source_config={})
    monkeypatch.setattr(formal.torch, "load", lambda *args, **kwargs: payload)

    def args(arm):
        return SimpleNamespace(protocol=protocol, preflight=False, seed=6201,
            mode="control", arm=arm, out=tmp_path / arm, scenario="stable", device="cpu")

    return payload, args, inputs


@pytest.mark.parametrize("field,value,message", [
    ("total_updates", 49999, "initialization budget"),
    ("online_critic_updates", 1, "initialization budget"),
    ("source_training_seed", 6202, "seed mismatch"),
])
def test_wrong_initializer_is_rejected_before_any_environment(prepared_inputs, monkeypatch,
                                                             field, value, message):
    payload, args, _ = prepared_inputs
    (payload["extra"] if field == "source_training_seed" else payload)[field] = value

    def unexpected_environment(*args, **kwargs):
        pytest.fail("Invalid initialization must fail before creating the target environment")

    monkeypatch.setattr(formal, "create_target_env", unexpected_environment)
    with pytest.raises(ValueError, match=message):
        formal.run(args("replay"))


def test_all_control_arms_load_same_seed_initializer_and_policy_config(prepared_inputs, monkeypatch):
    _, args, inputs = prepared_inputs
    loaded = []

    class Loaded(Exception):
        pass

    class Trainer:
        def __init__(self, replay, *, config, device, seed):
            self.config, self.seed = config, seed

        def load_offline_initializer(self, initializer):
            loaded.append((initializer, self.config, self.seed))
            raise Loaded()

    monkeypatch.setattr(formal, "ClosedLoopTrainer", Trainer)
    for arm in PROTOCOL["arms"]:
        with pytest.raises(Loaded):
            formal.run(args(arm))
    assert [entry[0] for entry in loaded] == [inputs / "h2oplus_seed6201.pt"] * 3
    assert loaded[0][1] == loaded[1][1] == loaded[2][1]
    assert not loaded[0][1].use_ratio_correction
    assert [entry[2] for entry in loaded] == [6201] * 3


def test_cpu_optimizer_conversion_preserves_adam_state_and_allows_real_steps():
    parameters, optimizers, before = [], [], []
    for index in range(5):
        parameter = torch.nn.Parameter(torch.tensor([1. + index, -1.]))
        optimizer = torch.optim.Adam([parameter], lr=(index + 1) * 3e-4,
                                     betas=(.9, .99), weight_decay=.01)
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0].update(fused=True, capturable=True)
        before.append(dict(parameter=parameter.detach().clone(),
            state={key: value.clone() if torch.is_tensor(value) else value
                   for key, value in optimizer.state[parameter].items()},
            hyperparameters={key: value for key, value in optimizer.param_groups[0].items()
                             if key not in ("params", "fused", "capturable")}))
        parameters.append(parameter)
        optimizers.append(optimizer)
    learner = SimpleNamespace(actor_optimizer=optimizers[0], critic_optimizer=optimizers[1],
        value_optimizer=optimizers[2], alpha_optimizer=optimizers[3],
        ratio=SimpleNamespace(optimizer=optimizers[4]))

    formal.set_cpu_optimizer_runtime(learner)

    for parameter, optimizer, saved in zip(parameters, optimizers, before):
        group = optimizer.param_groups[0]
        assert group["fused"] is False and group["capturable"] is False
        assert {key: value for key, value in group.items()
                if key not in ("params", "fused", "capturable")} == saved["hyperparameters"]
        assert group["params"][0] is parameter
        assert torch.equal(parameter, saved["parameter"])
        for key, value in optimizer.state[parameter].items():
            if torch.is_tensor(value):
                assert torch.equal(value, saved["state"][key])
            else:
                assert value == saved["state"][key]
        parameter.square().sum().backward()
        optimizer.step()
        assert not torch.equal(parameter, saved["parameter"])
        assert optimizer.state[parameter]["step"].item() == saved["state"]["step"].item() + 1
