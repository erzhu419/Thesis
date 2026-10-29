"""Frozen v73 learner with new isolated initializers and unchanged resident data."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026 import learning as frozen
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import learning as native

HERE = Path(__file__).resolve().parent
SHARED_INPUTS = native.resident.INPUTS
FIELDS = frozen.FIELDS
FreshOutcomeReplay = frozen.FreshOutcomeReplay
DecoupledReplay = frozen.DecoupledReplay
ExecutedCQLTrainer = native.ExecutedCQLTrainer
ExecutionAction = native.ExecutionAction
TransitTrainerConfig = native.TransitTrainerConfig
set_cpu_optimizer_runtime = native.resident.formal.set_cpu_optimizer_runtime
cql_checks, recency_checks, target_checks = frozen.cql_checks, frozen.recency_checks, frozen.target_checks


def load_learner(seed, *, arm):
    """Retain the original construction order, Adam restoration and split replay."""
    torch.set_num_threads(1)
    metadata = json.loads((SHARED_INPUTS/'history_metadata.json').read_text())
    with np.load(SHARED_INPUTS/'history.npz') as archive:
        history = {key: archive[key] for key in (*FIELDS.values(), 'sample_weight')}
    source = native.load_table(SHARED_INPUTS/'source_9101.npz', 4096)
    contract = native.resident.formal.TransitMDPConfigV7(**metadata['contract'])
    stations = np.asarray(metadata['station_max_by_line'], np.float32)
    initializer = HERE/'inputs'/f'h2oplus_seed{seed}.pt'
    payload = torch.load(initializer, map_location='cpu', weights_only=False)
    if (payload['total_updates'] != 50000 or payload['online_critic_updates'] != 0
            or payload['extra']['source_training_seed'] != seed):
        raise ValueError('v75 requires the matching new50000-step offline-only initializer')
    config = replace(TransitTrainerConfig(**payload['source_config']), use_ratio_correction=False)
    if (config.batch_size != 256 or config.cql_n_actions != 10
            or config.online_backup != 'h2oplus' or config.mixed_backup_lambda != .1):
        raise ValueError('Frozen v73 requires original128/64/64 quotas and original mixed backup')
    del payload
    # Construct the original replay and trainer before installing the identical
    # v73 split replay. This preserves original constructor/RNG/optimizer order.
    old = native.FreshOutcomeReplay(history, source, contract, stations,
        arm=arm, outcome_seed=seed+600000)
    learner = ExecutedCQLTrainer(old, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    set_cpu_optimizer_runtime(learner)
    learner.activate_executed_cql()
    split_history = dict(old.history, sample_weight=old.history_weights.copy())
    replay = DecoupledReplay(split_history, old.source, contract, stations,
        arm=arm, outcome_seed=seed+600000)
    replay.history_weights = old.history_weights.copy()
    learner.replay_buffer = replay
    return learner, replay, contract, stations, initializer
