"""Final fixed0250 model/trust state; restoring never fits a model."""

import copy
from pathlib import Path

import numpy as np
import torch


def _module_state(module):
    return dict(
        weights={key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        training={name: child.training for name, child in module.named_modules()},
        requires_grad={name: value.requires_grad for name, value in module.named_parameters()},
    )


def _load_module(module, state):
    module.load_state_dict(state["weights"], strict=True)
    for name, child in module.named_modules():
        child.training = state["training"][name]
    for name, value in module.named_parameters():
        value.requires_grad_(state["requires_grad"][name])
    return module


def _table(table, device):
    return {key: value.detach().to(device).clone() for key, value in table.items()}


def capture_augmentation(augmentation):
    """Copy the live banks after refits, not the initial pilot_models.pt."""
    aug = augmentation
    if aug.mode != "fixed0250" or aug.protocol["arm"] != "fixed0250":
        raise ValueError("v105 restores only the final fixed0250 ContextModel")
    if aug.p != aug.protocol or aug.models is not aug.banks[aug.family]:
        raise ValueError("The selected live model bank must match its protocol")
    for source, residual in zip(aug.banks["source"], aug.banks["residual"], strict=True):
        if residual.base is not source:
            raise ValueError("Residual models must share the corresponding source model")
    trust = aug.trust
    if trust.sig_fn is not None or trust.sig_dim != 0:
        raise ValueError("The original bus trust model has no signature callback")
    return dict(
        protocol=copy.deepcopy(aug.protocol), mode=aug.mode, family=aug.family,
        history=copy.deepcopy(aug.history), stations=aug.stations.detach().cpu().clone(),
        banks={family: [_module_state(model) for model in aug.banks[family]]
               for family in ("source", "direct", "residual")},
        normalizer=_module_state(aug.norm),
        target={split: _table(aug.target[split], "cpu") for split in ("fit", "stop", "calibrate")},
        fit=_table(aug.fit, "cpu"), fit_ns=aug.fit_ns.detach().cpu().clone(),
        errors=aug.errors.detach().cpu().clone(),
        support={key: getattr(aug.support, key).detach().cpu().clone()
                 for key in ("lines", "mean", "scale", "reference", "radius")},
        trust=dict(net=_module_state(trust.net), target=_module_state(trust.target),
                   optimizer=copy.deepcopy(trust.optimizer.state_dict()),
                   obs_dim=trust.obs_dim, act_dim=trust.act_dim,
                   hidden=trust.net.net[0].out_features, gamma=trust.gamma,
                   tau_polyak=trust.tau_polyak, tau=trust._tau,
                   tau_half_life=trust._tau_half_life),
        rng=copy.deepcopy(aug.rng.bit_generator.state),
        generator=aug.generator.get_state().clone(),
    )


def restore_augmentation(state, *, model_class, device, out, costs):
    """Build the original objects without invoking ContextModel.__init__."""
    # These are the modules loaded by the original v13 implementation bootstrap.
    from model import BusModel, Normalizer
    from control_model import QDeltaBellman
    from mechanisms import TargetSupport

    if state["mode"] != "fixed0250" or state["protocol"]["arm"] != "fixed0250":
        raise ValueError("v105 restores only the final fixed0250 ContextModel")
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Network constructors draw initial weights that are immediately overwritten.
    # They must not advance the learner's already restored global RNG stream.
    with torch.random.fork_rng(devices=devices):
        aug = model_class.__new__(model_class)
        aug.device, aug.out, aug.costs = device, Path(out), costs
        aug.protocol = copy.deepcopy(state["protocol"])
        aug.p = aug.protocol
        aug.mode, aug.family = state["mode"], state["family"]
        aug.history = copy.deepcopy(state["history"])
        aug.stations = state["stations"].to(device).clone()
        aug.banks = {family: [] for family in ("source", "direct", "residual")}
        for saved in state["banks"]["source"]:
            aug.banks["source"].append(_load_module(BusModel().to(device), saved))
        for saved in state["banks"]["direct"]:
            aug.banks["direct"].append(_load_module(BusModel().to(device), saved))
        for source, saved in zip(aug.banks["source"], state["banks"]["residual"], strict=True):
            aug.banks["residual"].append(_load_module(BusModel(source).to(device), saved))
        aug.models = aug.banks[aug.family]
        aug.target = {split: _table(state["target"][split], device)
                      for split in ("fit", "stop", "calibrate")}
        aug.fit = _table(state["fit"], device)
        aug.fit_ns = state["fit_ns"].to(device).clone()
        aug.errors = state["errors"].to(device).clone()
        # Only register the four original buffers; do not recompute statistics.
        aug.norm = Normalizer.__new__(Normalizer)
        torch.nn.Module.__init__(aug.norm)
        for key in ("g_mean", "g_std", "local_mean", "local_std"):
            aug.norm.register_buffer(key, state["normalizer"]["weights"][key].to(device).clone())
        _load_module(aug.norm, state["normalizer"])
        aug.support = TargetSupport.__new__(TargetSupport)
        for key in ("lines", "mean", "scale", "reference", "radius"):
            setattr(aug.support, key, state["support"][key].to(device).clone())
        saved = state["trust"]
        aug.trust = QDeltaBellman(saved["obs_dim"], saved["act_dim"],
                                  hidden=saved["hidden"], gamma=saved["gamma"],
                                  tau_half_life=saved["tau_half_life"], device=device)
        _load_module(aug.trust.net, saved["net"])
        _load_module(aug.trust.target, saved["target"])
        aug.trust.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        aug.trust.tau_polyak, aug.trust._tau = saved["tau_polyak"], saved["tau"]
        aug.rng = np.random.default_rng(0)
        aug.rng.bit_generator.state = copy.deepcopy(state["rng"])
        aug.generator = torch.Generator(device=device)
        aug.generator.set_state(state["generator"].cpu())
    return aug
