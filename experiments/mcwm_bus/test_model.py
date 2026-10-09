import numpy as np
import pytest
import torch

from data import analytic_next, within_batch
from model import BusModel, Normalizer, WIDTH, losses, targets
from control_model import ModelAugmentation


def example():
    s = np.zeros((4, 229), np.float32)
    s[:, 1] = 2 / 16
    s[:, 2] = 1 / 15
    slots = s[:, 5:].reshape(-1, 16, 14)
    slots[:, :2, 0] = 1
    slots[:, :2, 13] = -1
    ns = s.copy()
    ns[:, 2] = 0
    ns[:, 0] += 1 / 24000
    return dict(s=s, ns=ns, a=np.full((4, 1), -.5, np.float32),
                r=-np.arange(1, 5, dtype=np.float32), dt=np.ones(4, np.float32))


def test_batch_serialization():
    data = example()
    s = data["s"].copy()
    s[:, 2] = 0
    ns = analytic_next(s, data["a"])
    assert not within_batch(ns).any()
    np.testing.assert_array_equal(ns[:, 5 + 13], data["a"].reshape(-1))
    np.testing.assert_array_equal(ns[:, 0], s[:, 0])
    with pytest.raises(ValueError):
        analytic_next(ns, data["a"])


def test_inactive_slots_do_not_enter_losses():
    data = targets(example(), torch.full((12,), 46.), "cpu")
    norm = Normalizer(data)
    prediction = torch.zeros((4, WIDTH))
    original = losses(prediction, data, norm)
    for key in ("local", "flags"):
        data[key][~data["mask"]] = 99.
    data["line"][~data["mask"]] = 11
    data["station"][~data["mask"]] = 46
    torch.testing.assert_close(original, losses(prediction, data, norm))


def test_residual_initially_equals_frozen_source():
    torch.set_num_threads(2)
    data = targets(example(), torch.full((12,), 46.), "cpu")
    base = BusModel()
    residual = BusModel(base)
    torch.testing.assert_close(base(data["s"], data["a"]), residual(data["s"], data["a"]))
    norm = Normalizer(data)
    losses(residual(data["s"], data["a"]), data, norm).mean().backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert residual.head[-1].weight.grad.abs().sum() > 0


def test_generated_bus_state_has_valid_discrete_slots_and_clock():
    augmentation = ModelAugmentation.__new__(ModelAugmentation)
    augmentation.device = "cpu"
    augmentation.stations = torch.tensor([38, 36, 33, 34, 40, 41, 37, 38, 45, 46, 24, 23])
    augmentation.generator = torch.Generator().manual_seed(123)
    y = targets(example(), augmentation.stations, "cpu")
    augmentation.norm = Normalizer(y)
    ns, reward, dt = augmentation.decode(y["s"], torch.zeros(4, WIDTH))
    slots = ns[:, 5:].reshape(-1, 16, 14)
    active = slots[:, :, 0].bool()
    torch.testing.assert_close(active.sum(1), (ns[:, 1]*16).long())
    assert torch.all(slots[:, :, 1] == slots[:, :, 1].round())
    assert torch.all((slots[:, :, 2] >= 0) & (slots[:, :, 2] <= 1))
    assert torch.all(slots[:, :, :13][~active] == 0)
    assert torch.all(slots[:, :, 13] == -1)
    assert torch.all((dt >= 1) & (ns[:, 0] > y["s"][:, 0]) & (ns[:, 0] < 1))
    assert torch.all(reward <= 0)
