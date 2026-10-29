"""Actor-only prior-run references for the new v85 heldout episodes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V84 = HERE.parent/'mcwm_closed_loop_reference_collection_v84_20261027'
V75 = HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'
HELDOUT_EXTERNAL_SEED = 216207
COUNTS = dict(critic=296, value=296, actor=296, alpha=296)


def _runtime():
    # Initialize the existing resident import paths, without constructing a
    # learner or opening its replay, initializer, or optimizer state.
    from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.bootstrap import formal
    from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.learning import (
        ExecutionAction, SHARED_INPUTS)
    from SimpleSAC.transit_v7.networks import HurdleTransitActorV7
    return HurdleTransitActorV7, formal.TransitMDPConfigV7, ExecutionAction, SHARED_INPUTS


class FrozenReference:
    def __init__(self, actor, action_transform):
        self.actor = actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.action_transform = action_transform
        self.device = torch.device('cpu')
        self.last_proposal = None

    @torch.no_grad()
    def __call__(self, state):
        state = np.asarray(state, np.float32)
        proposal = self.actor(torch.as_tensor(state)[None], deterministic=True)[0]
        self.last_proposal = proposal[0].cpu().numpy().copy()
        return self.action_transform.numpy(state, self.last_proposal)

    def snapshot_state(self):
        return {key:value.detach().cpu().clone() for key,value in self.actor.state_dict().items()}


def load_reference(scenario):
    """Load a completed prior-run actor and two small JSON metadata files only."""
    if scenario not in ('stable', 'local_speed_drop'):
        raise ValueError('Only the two fixed heldout scenarios are declared')
    Actor, MDP, Transform, shared_inputs = _runtime()
    actor_path = V84/'runs'/scenario/'reference_actor.pt'
    config_path = V75/f'runs/executed_cql/validation/seed6207/{scenario}/fresh/protocol.json'
    metadata_path = Path(shared_inputs)/'history_metadata.json'
    payload = torch.load(actor_path, map_location='cpu', weights_only=False)
    source = payload['metadata']
    if not (source['source'] == 'v75' and source['source_arm'] == 'fresh'
            and source['seed'] == 6207 and source['scenario'] == scenario
            and source['packets'] == 37 and source['covariates'] == 185
            and source['complete_outcomes'] == 185 and source['counts'] == COUNTS
            and source['data_deadline'] == 12000
            and source['max_available_at'] <= 12000 <= source['published_clock']
            and source['future_training_used'] is False):
        raise ValueError('Reference must be the matching completed v84 packet37 snapshot')
    config = json.loads(config_path.read_text())['policy_config']
    historical_metadata = json.loads(metadata_path.read_text())
    # Network initialization consumes Torch randomness; loading the fixed
    # reference must leave the caller's future native random stream intact.
    with torch.random.fork_rng(devices=[]):
        actor = Actor((config['hidden_dim'], config['hidden_dim']),
                      layer_norm=config['actor_layer_norm'], activation=config['activation'])
        actor.load_state_dict(payload['actor'])
    mdp = MDP(**historical_metadata['contract'])
    stations = np.asarray(historical_metadata['station_max_by_line'], np.float32)
    controller = FrozenReference(actor, Transform(mdp))
    metadata = dict(prior_run_reference=True, available_before_heldout_episode=True,
        heldout_available_at=0, heldout_external_seed=HELDOUT_EXTERNAL_SEED,
        source_snapshot_metadata=deepcopy(source), scenario=scenario,
        source_training_seed=6207, source_offline_updates=50000,
        source_response_updates=296, source_total_updates=50296,
        new_optimizer_updates=0, source_actor_path=str(actor_path),
        source_config_path=str(config_path), source_contract_path=str(metadata_path),
        deterministic=True, actor_frozen=True,
        prefix_scope='New episode controlled from0 by this prior-run frozen actor; not a v75 adaptive-prefix replay')
    return controller, mdp, stations, metadata
