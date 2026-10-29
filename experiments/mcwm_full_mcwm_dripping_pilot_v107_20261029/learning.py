"""Insert complete target rows into the unchanged historical MC-WM sampler."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


class TargetReplayAdapter:
    """Preserve original replay draws, then replace the fresh offline suffix."""

    def __init__(self, replay, arm, protocol, contract):
        from bus_h2o.mdp_contract_v7 import continuous_time_discount_v7

        if arm not in ("fresh", "no_fresh"):
            raise ValueError("Expected a declared fresh or no_fresh arm")
        self.replay, self.arm, self.contract = replay, arm, contract
        self.capacity = protocol["fresh_pool_capacity"]
        self.fresh_rows = protocol["fresh_rows"]
        self.packet_size = protocol["feedback_records_per_packet"]
        self.private_rng = np.random.default_rng(protocol["private_seed"])
        self.discount = continuous_time_discount_v7
        self.target_rows = []
        self.received = self.packets = self.calls = self.draws = self.replaced = 0
        self.main_rng_unchanged = True
        self.initial_source_counters = self._source_counters()
        self._original_sample_mixed = replay.sample_mixed
        replay.sample_mixed = self.sample_mixed

    def _source_counters(self):
        return (self.replay.online_size, self.replay._online_pointer,
                self.replay._online_total_appended)

    def add_target(self, packet):
        if len(packet) != self.packet_size:
            raise ValueError("Only complete five-row feedback packets are admitted")
        # no_fresh counts slots without accessing or retaining outcome objects.
        if self.arm == "fresh":
            rows = []
            for row in packet:
                rows.append(dict(
                    observations=np.asarray(row.s, dtype=np.float32).copy(),
                    actions=np.asarray(row.a, dtype=np.float32).copy(),
                    rewards=np.float32(row.r),
                    next_observations=np.asarray(row.ns, dtype=np.float32).copy(),
                    discounts=np.float32(0.0 if row.terminal else
                        self.discount(row.dt, self.contract)),
                    terminals=np.float32(row.terminal),
                    delta_t_seconds=np.float32(row.dt)))
            self.target_rows.extend(rows)
            self.target_rows = self.target_rows[-self.capacity:]
        self.received += len(packet)
        self.packets += 1

    def sample_mixed(self, batch_size, online_fraction, *, rng=None):
        if batch_size != 256 or online_fraction != .5:
            raise ValueError("The original 256-row half-source MC-WM batch is fixed")
        # Both original sampling calls finish before any private fresh draw.
        offline, source = self._original_sample_mixed(batch_size, online_fraction, rng=rng)
        after_original_rng = deepcopy(rng.bit_generator.state) if rng is not None else None
        self.calls += 1
        pool_size = min(self.received, self.capacity)
        if not pool_size:
            return offline, source
        selected = self.private_rng.integers(0, pool_size, size=self.fresh_rows)
        self.draws += len(selected)
        if self.arm == "fresh":
            if offline["observations"].shape[0] != 128 or source is None:
                raise RuntimeError("The restored historical source replay is required")
            for key, values in offline.items():
                replacement = np.stack([self.target_rows[index][key] for index in selected])
                values[-self.fresh_rows:] = torch.as_tensor(
                    replacement, device=values.device, dtype=values.dtype)
            self.replaced += len(selected)
        if rng is not None:
            self.main_rng_unchanged &= rng.bit_generator.state == after_original_rng
        return offline, source

    def summary(self):
        return dict(arm=self.arm, packets=self.packets,
            target_received=self.received,
            fresh_outcomes=self.received if self.arm == "fresh" else 0,
            target_records_received=self.received,
            target_outcome_labels_received=self.received if self.arm == "fresh" else 0,
            target_pool_rows=len(self.target_rows),
            matched_private_pool_size=min(self.received, self.capacity),
            fresh_pool_capacity=self.capacity, fresh_rows_per_update=self.fresh_rows,
            sample_mixed_calls=self.calls, private_draws=self.draws,
            replaced_rows=self.replaced, offline_rows=self.replay.offline_size,
            source_replay_rows=self.replay.online_size,
            source_unchanged=self._source_counters() == self.initial_source_counters,
            main_rng_unchanged=self.main_rng_unchanged,
            original_sampling_before_private=True,
            source_replay_append_calls=0,
            estimand="Complete target transitions, including their V/actor/temperature inputs")


def load_learner(root, out, arm, protocol):
    """Restore the verified 70000 endpoint; no fitting or optimizer update."""
    root, out = Path(root).resolve(), Path(out).resolve()
    original = json.loads((root / protocol["v105_protocol"]).read_text())
    restoration = json.loads((root / protocol["v106_protocol"]).read_text())
    verified = json.loads((root / protocol["v106_summary"]).read_text())
    if (verified["status"] != "complete"
            or not all(row["all_pass"] for row in verified["comparisons"].values())
            or verified["terminal_updates"] != 70000
            or restoration["input_checkpoint"] != protocol["input_checkpoint"]
            or original["training_seed"] != protocol["training_seed"]):
        raise ValueError("The declared v106 endpoint verification is required")
    restore_dir = root / "experiments/mcwm_endpoint_restore_v106_20261029"
    sys.path.insert(0, str(restore_dir))
    spec = importlib.util.spec_from_file_location("v107_endpoint_restore", restore_dir / "run.py")
    restore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restore)
    implementation, state, runtime_torch = restore.bootstrap(root, original)
    runtime = dict(host=platform.node(), python=platform.python_version(),
        torch=runtime_torch.__version__, device="cpu", threads=runtime_torch.get_num_threads())
    if any(runtime[key] != original["training_runtime"][key]
           for key in ("python", "torch", "device", "threads")):
        raise RuntimeError("The historical training runtime must be preserved")
    checkpoint = root / protocol["input_checkpoint"]
    payload = runtime_torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["extra"]["seed"] != protocol["training_seed"]:
        raise ValueError("Checkpoint training seed differs from the fixed pilot")
    learner = restore.restore_learner(checkpoint, payload, implementation, state, out)
    reference = runtime_torch.load(root / original["reference_actor"],
        map_location="cpu", weights_only=False)
    actor_identity = state.compare_states(reference["actor"], learner.actor.state_dict())
    replay = learner.replay_buffer
    checks = dict(actor_exact=actor_identity["all_pass"],
        terminal_updates=learner.total_updates == 70000,
        online_critic_updates=learner.online_critic_updates == 20000,
        full_source_snapshot=replay.online_size == payload["online_replay"]["size"],
        original_source_events=payload["extra"]["source_events"] == original["expected_source_events"],
        full_offline_replay=replay.offline_size == 3284718,
        original_batch=learner.config.batch_size == 256 and learner.config.online_fraction == .5,
        no_ratio_correction=not learner.config.use_ratio_correction,
        original_mixture=learner.protocol["real_td_coefficient"] == .75 and learner.coefficient == .25,
        augmentation_present=learner.augmentation is not None)
    if not all(checks.values()):
        raise ValueError("The historical complete MC-WM endpoint changed")
    contract = implementation.base.TransitMDPConfigV7(**replay.contract_config)
    stations = replay.station_max_by_line.copy()
    receipt = dict(all_pass=all(checks.values()), checks=checks,
        checkpoint=str(checkpoint), training_seed=protocol["training_seed"],
        total_updates=learner.total_updates, online_critic_updates=learner.online_critic_updates,
        actor_exact=actor_identity["all_pass"], actor_identity=actor_identity,
        offline_rows=replay.offline_size, source_replay_rows=replay.online_size,
        original_source_events=payload["extra"]["source_events"],
        real_td_coefficient=learner.protocol["real_td_coefficient"],
        model_td_coefficient=learner.coefficient,
        model_batch_size=learner.protocol["model_batch_size"],
        model_family=learner.augmentation.family,
        actor_learning_rates=[group["lr"] for group in learner.actor_optimizer.param_groups],
        online_backup=learner.config.online_backup,
        mixed_backup_lambda=learner.config.mixed_backup_lambda,
        reward_scale=learner.config.reward_scale,
        discount_gamma_per_minute=contract.discount_gamma_per_minute,
        v106_verified=True, runtime=runtime)
    adapter = TargetReplayAdapter(replay, arm, protocol, contract)
    return learner, adapter, contract, stations, receipt
