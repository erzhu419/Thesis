"""Compute-matched attribution of fresh factual outcome supervision.

Fresh current states/actions are available in both arms. Only critic outcome
supervision differs: no_fresh substitutes complete historical transitions.
This is not an ablation of all target-state information.
"""

from collections import deque
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_diagnostic_v2_20261020 import bootstrap as resident
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_action_local_v3_20261020.learning import ActionConsistentTrainer
from experiments.mcwm_closed_loop_v1.learning import TransitTrainerConfig, load_table


FIELDS = dict(observations="s", actions="a", next_observations="ns", rewards="r",
              delta_t_seconds="dt", terminals="terminal")


def _batch(table, ids, device):
    batch = {key: torch.as_tensor(table[value][ids], dtype=torch.float32, device=device)
             for key, value in FIELDS.items()}
    batch["actions"] = batch["actions"].reshape(-1, 1)
    for key in ("rewards", "delta_t_seconds", "terminals"):
        batch[key] = batch[key].reshape(-1)
    batch["discounts"] = torch.ones(len(ids), device=device)
    return batch


def _join(left, right):
    return {key: torch.cat((left[key], right[key])) for key in left}


class FreshOutcomeReplay:
    """Source 1/2, historical target 1/4, fresh/replacement outcomes 1/4.

    Every update makes the same four sampling draws in either arm. Replacement
    indices are drawn even in the fresh arm, so the ablation does not shift the
    subsequent sampler RNG stream. No-fresh ingestion never reads outcome fields.
    """

    def __init__(self, history, source, contract, stations, *, arm, device="cpu", window=285):
        if arm not in ("fresh", "no_fresh"):
            raise ValueError(arm)
        self.arm, self.device = arm, device
        self.history = {key: history[key] for key in FIELDS.values()}
        self.source = {key: source[key] for key in FIELDS.values()}
        weights = np.asarray(history["sample_weight"], dtype=np.float64)
        self.history_weights = weights / weights.sum()
        self.contract_config = asdict(contract) if not isinstance(contract, dict) else dict(contract)
        self.station_max_by_line = stations
        self.recent = deque(maxlen=window)
        self.target_received = 0
        self.last_covariates = self.last_sampling = None

    @property
    def online_size(self):
        return len(self.source["s"])

    def add_target(self, rows):
        for row in rows:
            # Copy only s/a in the ablation, rather than keeping outcome-bearing
            # row objects and relying on later code to avoid reading them.
            values = {key: np.asarray(getattr(row, key)).copy() for key in ("s", "a")}
            if self.arm == "fresh":
                values.update({key: np.asarray(getattr(row, key)).copy()
                               for key in ("ns", "r", "dt", "terminal")})
            self.recent.append(values)
            self.target_received += 1

    def sample_mixed(self, size, online_fraction, *, rng):
        if online_fraction != .5 or size % 4:
            raise ValueError("v25 uses source/history/fresh slots in 2:1:1 proportions")
        if not self.recent:
            raise RuntimeError("Continue only after an admitted factual feedback packet")
        count = size // 4
        base_ids = rng.choice(len(self.history["s"]), size=count, p=self.history_weights)
        replacement_ids = rng.choice(len(self.history["s"]), size=count, p=self.history_weights)
        recent_ids = rng.integers(len(self.recent), size=count)
        source_ids = rng.integers(len(self.source["s"]), size=2*count)
        covariates = {key: np.asarray([row[key] for row in self.recent]) for key in ("s", "a")}
        historical = _batch(self.history, base_ids, self.device)
        self.last_covariates = dict(
            observations=torch.cat((historical["observations"], torch.as_tensor(
                covariates["s"][recent_ids], dtype=torch.float32, device=self.device))),
            actions=torch.cat((historical["actions"], torch.as_tensor(
                covariates["a"][recent_ids], dtype=torch.float32, device=self.device).reshape(-1, 1))))
        if self.arm == "fresh":
            table = {key: np.asarray([row[key] for row in self.recent]) for key in FIELDS.values()}
            outcomes = _batch(table, recent_ids, self.device)
        else:
            outcomes = _batch(self.history, replacement_ids, self.device)
        self.last_sampling = dict(history=base_ids.tolist(), replacement=replacement_ids.tolist(),
                                  recent=recent_ids.tolist(), source=source_ids.tolist())
        return _join(historical, outcomes), _batch(self.source, source_ids, self.device)

    def summary(self):
        return dict(arm=self.arm, feedback_slots_received=self.target_received,
            recent_covariates=len(self.recent), fresh_outcomes_stored=(
                len(self.recent) if self.arm == "fresh" else 0),
            td_slots=dict(source=128, historical_target=64,
                fresh_target=64 if self.arm == "fresh" else 0,
                historical_replacement=64 if self.arm == "no_fresh" else 0),
            fresh_current_s_a_use=["IQL value", "SAC actor", "temperature"],
            fresh_outcome_use=(["critic factual Bellman target"] if self.arm == "fresh" else []),
            no_fresh_replacement="complete historical (s,a,r,ns,dt,terminal), never hybrid rows",
            sampler_draw_order=["history", "historical replacement", "fresh covariates", "source"])


class FreshOutcomeTrainer(ActionConsistentTrainer):
    """Keep the resident optimizer schedule; align all policy-dependent Q with F."""

    def __init__(self, replay, *, action_transform, **kwargs):
        config = kwargs["config"]
        if (config.use_ratio_correction or config.temperature_state_source != "actor"
                or config.online_fraction != .5 or config.batch_size % 4):
            raise ValueError("v25 requires no ratio, current-state temperature, and 2:1:1 sampling")
        self.outcome_arm = replay.arm
        super().__init__(replay, arm="real1", action_transform=action_transform, **kwargs)

    def _iql_value_terms(self, batch):
        # IQL V(s) regresses Q_target(s,a). It reads neither r nor ns. Both arms
        # retain the same current-covariate rule despite differing critic data.
        return super()._iql_value_terms(self.replay_buffer.last_covariates)

    def _actor_states(self, offline_batch, online_batch):
        return torch.cat((self.replay_buffer.last_covariates["observations"],
                          online_batch["observations"]))

    def _temperature_states(self, offline_batch, online_batch):
        return self._actor_states(offline_batch, online_batch)

    def continue_step(self, collect_diagnostics=False):
        if self.world is not None:
            raise RuntimeError("v25 has no world or trust update path")
        metrics = super().continue_step(collect_diagnostics=collect_diagnostics)
        metrics["outcome_arm"] = self.outcome_arm
        return metrics


def load_learner(seed, *, arm):
    """Read server-resident inputs; never copy checkpoints or history locally."""
    torch.set_num_threads(1)
    inputs = resident.INPUTS
    metadata = json.loads((inputs/"history_metadata.json").read_text())
    with np.load(inputs/"history.npz") as archive:
        history = {key: archive[key] for key in (*FIELDS.values(), "sample_weight")}
    source = load_table(inputs/"source_9101.npz", 4096)
    contract = resident.formal.TransitMDPConfigV7(**metadata["contract"])
    stations = np.asarray(metadata["station_max_by_line"], np.float32)
    initializer = inputs/f"h2oplus_seed{seed}.pt"
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    if (payload["total_updates"] != 50000 or payload["online_critic_updates"] != 0
            or payload["extra"]["source_training_seed"] != seed):
        raise ValueError("Wrong read-only initializer")
    config = replace(TransitTrainerConfig(**payload["source_config"]), use_ratio_correction=False)
    if config.batch_size != 256:
        raise ValueError("Formal v25 quotas require batch_size=256")
    del payload
    replay = FreshOutcomeReplay(history, source, contract, stations, arm=arm)
    learner = FreshOutcomeTrainer(replay, config=config, device="cpu", seed=seed,
                                  action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    resident.formal.set_cpu_optimizer_runtime(learner)
    return learner, replay, contract, stations, initializer
