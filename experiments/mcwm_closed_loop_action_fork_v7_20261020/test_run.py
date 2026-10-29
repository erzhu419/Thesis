"""Exercise the native string-device prefix update using the real learner."""

import copy
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("THESIS_H2O_ROOT", str(HERE.parent / "sumo_eval/runtime/H2Oplus"))
sys.path.insert(0, str(HERE))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.run import train_packet
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    _learner_state, restore_rng, rng_state,
)
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_world import issue_and_row
from experiments.mcwm_closed_loop_paired_world_v6_20261020.run import action_probe
from experiments.mcwm_closed_loop_paired_world_v6_20261020.test_paired_world import paired_learner
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def test_native_prefix_packet_matches_v6_learning_state_and_rng_after_all_eight_steps():
    measured = paired_learner()
    assert measured.replay_buffer.device == "cpu"  # Native replay input type.
    world = measured.world
    world.paired_probe_deadlines = []
    world.context_enabled = False
    packet = []
    for sequence in range(100, 105):
        issue, row = issue_and_row(sequence)
        world.register(issue)
        packet.append(row)
    discarded_issue, _ = issue_and_row(99)
    world.register(discarded_issue)
    reference = copy.deepcopy(measured)
    replay = measured.replay_buffer
    contract = TransitMDPConfigV7()
    before = dict(policy=measured.total_updates, critic=measured.online_critic_updates,
        rows=len(replay.recent), fits=world.refits, gradient=world.model_updates,
        scored=len(world.paired_issue_records), trust=world.trust_updates)
    initial_rng = rng_state()

    train_packet(measured, replay, world, packet, [99], {"deadline": 9000.},
                 now=9031., updates=1, contract=contract)
    final_rng = rng_state()

    # The reference is the v6 packet path, with its passive probes disabled at
    # this deadline. It uses the actual contract in both action measurements.
    restore_rng(initial_rng)
    ref_world, ref_replay = reference.world, reference.replay_buffer
    ref_world.pending.pop(99)
    assert len(ref_world.observe(packet, 9031.)) == 5
    action_probe(reference, packet, contract, reference.action_transform)
    ref_replay.add_target(packet)
    ref_world.refit(1)
    ref_world.recenter_context(packet)
    assert ref_world.diagnose_packet(reference, packet, 9000., 9031.) is None
    for step in range(8):
        ref_world.capture_counterfactual = step == 0
        reference.local_counterfactual = reference.head_counterfactual = None
        reference.continue_step(collect_diagnostics=step == 0)
        ref_world.capture_counterfactual = False
        ref_world.counterfactual_batch = None
        ref_world.head_counterfactual_batches = None
        ref_world.head_counterfactual_weights = None
        ref_world.head_counterfactual_outputs = None
    action_probe(reference, packet, contract, reference.action_transform)

    assert measured.total_updates - before["policy"] == 8
    assert measured.online_critic_updates - before["critic"] == 8
    assert len(replay.recent) - before["rows"] == 5
    assert world.refits - before["fits"] == 1
    assert world.model_updates - before["gradient"] == 2  # Direct + residual.
    assert len(world.paired_issue_records) - before["scored"] == 5
    assert world.trust_updates - before["trust"] == 8
    assert len(world.trust_records) == 8
    assert world.paired_processed_at == 9031.
    assert 99 not in world.pending
    assert all(row.sequence not in world.pending for row in packet)
    assert_same(_learner_state(measured), _learner_state(reference))
    assert_same(world.twin_trust.net.state_dict(), ref_world.twin_trust.net.state_dict())
    assert_same(world.twin_trust.target.state_dict(), ref_world.twin_trust.target.state_dict())
    assert_same(world.twin_trust.optimizer.state_dict(), ref_world.twin_trust.optimizer.state_dict())
    assert_same(rng_state(), final_rng)
    assert world.trust_records == ref_world.trust_records
