"""Real successor indexing, component closure and frozen CRN isolation."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial, rows
from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import attach_shared_level
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _state, _equal, _rng
from experiments.mcwm_closed_loop_action_attribution_v56_20261025 import credit


def fixture():
    learner = attach_shared_level(initial())
    learner.sync_actor, learner.sync_transform = deepcopy(learner.actor), deepcopy(learner.action_transform)
    records = [deepcopy(rows()[i % 5]) for i in range(271)]
    for record in records:
        record.line_id = '102X'
    tensor = lambda field: torch.tensor(np.asarray([getattr(row, field) for row in records]),dtype=torch.float32)
    states = lambda field: torch.cat((tensor(field),torch.tensor(np.asarray([
        getattr(row,'memory_'+field)['vector'] for row in records]),dtype=torch.float32)),1)
    learner.physical_batch = dict(observations=states('s'),next_observations=states('ns'),
        actions=tensor('a'),rewards=tensor('r'),discounts=torch.ones(271),
        terminals=torch.zeros(271))
    learner.physical_coefficients = torch.full((270,), .99)
    with torch.no_grad():
        learner.target_critic.level_weight.fill_(.002)
        learner.target_critic.action_weight[0].fill_(.004)
        learner.target_critic.action_weight[1].fill_(-.003)
    return learner, records


def test_true_successor_reward_excludes_root_and_stops_at_zero_link():
    atoms = torch.tensor([100., 2., 3., 4.],dtype=torch.float64)
    actual = credit.future_correction(atoms,torch.tensor([.5,0.,.25]),eta=1.)
    assert actual.tolist() == [1.,0.,1.,0.]
    # Root's own large reward cannot masquerade as future physical credit.
    atoms[0]=-1000.
    assert torch.equal(actual,credit.future_correction(atoms,torch.tensor([.5,0.,.25]),eta=1.))


def test_frozen_m8_components_repeat_exactly_and_leave_parameters_Adam_RNG_unchanged():
    learner, _ = fixture()
    before, rng = deepcopy(_state(learner)),_rng()
    one = credit.frozen_block(learner,credit.DEFAULT_SEEDS[0])
    two = credit.frozen_block(learner,credit.DEFAULT_SEEDS[0])
    assert torch.equal(one['atoms'],two['atoms'])
    assert torch.equal(one['corrections'],two['corrections'])
    assert one['decomposition_max_abs'] < 1e-5
    assert one['native_M8_reference_max_abs'] <= one['native_M8_float32_bound']
    assert _equal(before,_state(learner)) and _equal(rng,_rng())
    # The final real row has no admitted successor correction.
    assert torch.count_nonzero(one['corrections'][-1]) == 0


def test_common_Q_V_level_offset_cancels_but_Q_only_exposes_mixed_backup():
    learner, records = fixture()
    result=credit.compare_targets(dict(one_step=learner,trace=deepcopy(learner)),records,
                                  seeds=credit.DEFAULT_SEEDS[:1])
    offset=result['arms']['trace']['offsets']
    assert offset['joint_trace_correction_max_abs'] == 0.
    assert offset['Q_only_trace_correction']['mean'] < 0.
    assert result['trace_minus_single']['all']['total']['mean_abs'] == 0.
    assert result['trace_minus_single']['all']['reward']['mean_abs'] == 0.
    records[2].s[0] += .02
    with pytest.raises(RuntimeError,match='same admitted chain'):
        credit.compare_targets(dict(one_step=learner),records,seeds=credit.DEFAULT_SEEDS[:1])
