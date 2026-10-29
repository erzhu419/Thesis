"""Complete fixed0250 endpoint; reuse the historical trainer's checkpoint API."""
from copy import deepcopy
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import numpy as np
import torch

from model_state import capture_augmentation, restore_augmentation

SCHEMA = "mcwm-fixed0250-final-v1"
CONTEXT_FIELDS = (
    "protocol", "arm", "model_fraction", "model_seed", "coefficient",
    "episode_index", "feature_indices", "coefficient_count", "coefficient_sum",
    "changed_updates", "coefficient_min", "coefficient_max", "trace",
)


def global_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state().clone(),
        cuda=torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None)


def restore_global_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def replay_binding(replay):
    return dict(dataset_path=str(replay.dataset_path), split=replay.split,
        schema_version=replay.schema_version, offline_size=replay.offline_size,
        contract_config=deepcopy(replay.contract_config),
        station_max_by_line=replay.station_max_by_line.tolist(),
        device_cache_observation_dtype=replay.device_cache_observation_dtype)


def _extra(learner, extra=None):
    if (learner.arm != "fixed0250" or learner.context_filter is not None
            or learner.bundle is not None or learner.coefficient != .25):
        raise ValueError("Only the historical fixed0250 / no-context-filter state is supported")
    if learner.model_loss is not None or learner.augmentation is None:
        raise ValueError("Save after a completed update with the fitted augmentation present")
    result = dict(extra or {})
    result["mcwm_state"] = dict(schema=SCHEMA,
        boundary="source training ended; target resumes from a new episode",
        replay_binding=replay_binding(learner.replay_buffer),
        context={name: deepcopy(getattr(learner, name)) for name in CONTEXT_FIELDS},
        blocks=deepcopy(vars(learner.blocks)),
        augmentation=capture_augmentation(learner.augmentation))
    return result


def save_final(learner, path, extra=None):
    if learner.total_updates != 70000 or learner.online_critic_updates != 20000:
        raise ValueError("Expected the original 50000 + 20000 update endpoint")
    learner.save(path, extra=_extra(learner, extra), include_online_replay=True)


def load_final(learner, path, *, model_class, out=None, costs=None):
    """Restore into the same configured trainer/replay, without fitting or collecting."""
    extra = learner.load(path, load_optimizers=True)
    saved = extra.pop("mcwm_state")
    if saved["schema"] != SCHEMA:
        raise ValueError("Unexpected MC-WM endpoint schema")
    if saved["replay_binding"] != replay_binding(learner.replay_buffer):
        raise ValueError("Resident offline replay binding differs from the saved endpoint")
    if learner.total_updates != 70000 or learner.online_critic_updates != 20000:
        raise ValueError("Checkpoint is not the historical 70000 update endpoint")
    rng = global_rng(learner.device)
    try:
        for name, value in saved["context"].items():
            setattr(learner, name, deepcopy(value))
        learner.context_filter, learner.bundle = None, None
        learner.blocks = type(learner.blocks)(saved["blocks"]["size"])
        learner.blocks.__dict__.update(deepcopy(saved["blocks"]))
        if out is not None:
            learner.output = Path(out) / "results"
        if costs is not None:
            learner.costs = costs
        learner.augmentation = restore_augmentation(saved["augmentation"],
            model_class=model_class, device=learner.device,
            out=learner.output.parent / "models", costs=learner.costs)
        learner.model_loss, learner.real_losses, learner.loss_diagnostics = None, [], {}
    finally:
        # Auxiliary network constructors consume global Torch RNG; their final
        # weights and private RNGs come from the artifact, not those draws.
        restore_global_rng(rng, learner.device)
    return extra


def export_state(learner):
    """Independent CPU copy using the existing full payload, also after a test update."""
    with TemporaryDirectory(prefix="mcwm-state-") as folder:
        path = Path(folder) / "state.pt"
        learner.save(path, extra=_extra(learner), include_online_replay=True)
        return torch.load(path, map_location="cpu", weights_only=False)


def compare_states(left, right):
    """Exact tensor/array/scalar comparison with compact, actionable differences."""
    differences, maximum = [], 0.

    def visit(a, b, path):
        nonlocal maximum
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                differences.append(path + ".keys")
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, type(a)):
            if len(a) != len(b):
                differences.append(path + ".length")
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
        elif torch.is_tensor(a) and torch.is_tensor(b):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                differences.append(path)
                if a.shape == b.shape and a.numel():
                    maximum = max(maximum, float((a.double()-b.double()).abs().max()))
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.dtype != b.dtype or not np.array_equal(a, b):
                differences.append(path)
                if a.shape == b.shape and a.size:
                    maximum = max(maximum, float(np.max(np.abs(a.astype(float)-b.astype(float)))))
        elif type(a) is not type(b) or a != b:
            differences.append(path)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                maximum = max(maximum, abs(float(a)-float(b)))

    visit(left, right, "state")
    return dict(all_pass=not differences, differing_field_count=len(differences),
        differing_fields=differences[:20], max_abs_difference=maximum)
