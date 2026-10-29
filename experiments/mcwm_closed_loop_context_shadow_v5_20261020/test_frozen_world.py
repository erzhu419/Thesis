"""Distinguish frozen neural weights from frozen family, trust or policy updates."""

import copy
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/'mcwm_closed_loop_diagnostic_v2_20261020'))
from experiments.mcwm_closed_loop_context_shadow_v5_20261020 import bootstrap
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.frozen_world import FrozenWorld
from experiments.mcwm_closed_loop_action_local_v3_20261020.learning import ActionConsistentWorldMixin
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.world_diagnostics import DiagnosticWorld
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_probes import learner
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_world import issue_and_row


class UpdatingWorld(ActionConsistentWorldMixin, DiagnosticWorld):
    pass


def test_identical_initial_fit_then_frozen_parameters_and_matched_refit_draws():
    item = learner()
    item.replay_buffer.device = 'cpu'
    torch.manual_seed(917)
    updating = UpdatingWorld(copy.deepcopy(item.replay_buffer), seed=41, updates=1)
    updating.action_transform = item.action_transform
    torch.manual_seed(917)
    frozen = FrozenWorld(copy.deepcopy(item.replay_buffer), seed=41, updates=1,
                         action_transform=item.action_transform)
    initial = copy.deepcopy({name: model.state_dict() for name, model in updating.models.items()})
    for name, model in frozen.models.items():
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[name][key], rtol=0, atol=0)
    assert frozen.rng.bit_generator.state == updating.rng.bit_generator.state
    assert frozen.family == updating.family
    for sequence in (7, 8):
        _, row = issue_and_row(sequence)
        for world in (updating, frozen):
            world.replay.add_target([row])
            world.refit(2)
        assert frozen.rng.bit_generator.state == updating.rng.bit_generator.state
    assert any(not torch.equal(value, initial[name][key])
               for name, model in updating.models.items() for key, value in model.state_dict().items())
    evidence = frozen.summary()['frozen_parameters']
    assert evidence['initial_fit_updates'] == 1
    assert evidence['skipped_refit_packets'] == 2
    assert evidence['skipped_neural_gradient_updates'] == 8
    assert evidence['online_neural_gradient_updates'] == 0
    assert all(row['changed_tensor_count'] == row['max_absolute_change'] == 0
               for row in evidence['models'].values())


def test_fresh_labels_select_family_and_eight_policy_trust_updates_leave_world_frozen():
    item = learner()
    item.replay_buffer.device = 'cpu'
    world = FrozenWorld(item.replay_buffer, seed=41, updates=1,
                        action_transform=item.action_transform)
    item.world = world
    issue, row = issue_and_row()
    world.register(issue)
    initial_family = world.family
    alternate = next(name for name in world.models if name != initial_family)
    # Make the saved initial-family prediction demonstrably worse on the
    # released label; the ordinary prequential selector must remain active.
    world.pending[row.sequence]['outputs'][initial_family].fill_(100.)
    world.observe([row], row.available_at)
    assert world.family == alternate
    world.replay.add_target([row])
    world.refit(2)
    trust_before = copy.deepcopy(world.trust.net.state_dict())
    actor_before = copy.deepcopy(item.actor.state_dict())
    before_updates = item.total_updates
    first = item.continue_step(collect_diagnostics=True)
    for _ in range(7):
        item.continue_step()
    assert item.total_updates-before_updates == 8
    assert first['td_diagnostic']['real_scale'] == .75
    assert first['td_diagnostic']['model_scale'] == .25
    assert first['td_diagnostic']['model_batches']
    assert any(not torch.equal(value, trust_before[key])
               for key, value in world.trust.net.state_dict().items())
    assert any(not torch.equal(value, actor_before[key])
               for key, value in item.actor.state_dict().items())
    evidence = world.summary()['frozen_parameters']
    assert evidence['trust_update_calls'] == 8
    assert evidence['family_selection_adapts'] is True
    assert evidence['local_readout_enabled'] is False
    assert not hasattr(world, 'local_readouts')
    assert all(row['changed_tensor_count'] == row['max_absolute_change'] == 0
               for row in evidence['models'].values())
