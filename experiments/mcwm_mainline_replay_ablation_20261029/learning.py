"""Keep the v72/v70 coupled learner; read the existing v75 initializers only."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import learning as native

HERE = Path(__file__).resolve().parent
INITIALIZERS = HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'/'inputs'
SHARED_INPUTS = native.resident.INPUTS
FIELDS = native.FIELDS
FreshOutcomeReplay = native.FreshOutcomeReplay
ExecutedCQLTrainer = native.ExecutedCQLTrainer
ExecutionAction = native.ExecutionAction
TransitTrainerConfig = native.TransitTrainerConfig
cql_checks, recency_checks, target_checks = native.cql_checks, native.recency_checks, native.target_checks


def load_learner(seed, *, arm):
    """Same construction order as v75, stopping before split-replay replacement."""
    if seed not in (6206, 6207, 6208) or arm != 'fresh':
        raise ValueError('Only the six declared coupled fresh days are in scope')
    torch.set_num_threads(1)
    metadata = json.loads((SHARED_INPUTS/'history_metadata.json').read_text())
    with np.load(SHARED_INPUTS/'history.npz') as archive:
        history = {key:archive[key] for key in (*FIELDS.values(), 'sample_weight')}
    source = native.load_table(SHARED_INPUTS/'source_9101.npz', 4096)
    contract = native.resident.formal.TransitMDPConfigV7(**metadata['contract'])
    stations = np.asarray(metadata['station_max_by_line'], np.float32)
    initializer = INITIALIZERS/f'h2oplus_seed{seed}.pt'
    payload = torch.load(initializer, map_location='cpu', weights_only=False)
    if (payload['total_updates'] != 50000 or payload['online_critic_updates'] != 0
            or payload['extra']['source_training_seed'] != seed):
        raise ValueError('Use the matching archived v75 offline-only50000-step initializer')
    config = replace(TransitTrainerConfig(**payload['source_config']), use_ratio_correction=False)
    if (config.batch_size != 256 or config.cql_n_actions != 10
            or config.online_backup != 'h2oplus' or config.mixed_backup_lambda != .1):
        raise ValueError('Keep the frozen v72/v70 batch, CQL and mixed-bootstrap recipe')
    del payload
    replay = FreshOutcomeReplay(history, source, contract, stations,
        arm=arm, outcome_seed=seed+600000)
    learner = ExecutedCQLTrainer(replay, config=config, device='cpu', seed=seed,
        action_transform=ExecutionAction(contract))
    learner.load_offline_initializer(initializer)
    native.resident.formal.set_cpu_optimizer_runtime(learner)
    learner.activate_executed_cql()
    return learner, replay, contract, stations, initializer
