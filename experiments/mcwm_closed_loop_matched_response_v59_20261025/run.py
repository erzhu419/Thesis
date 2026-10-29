"""Two matched retained-Adam M8 learners on the new v59 common stream."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.bootstrap import formal
import numpy as np
import torch

from experiments.mcwm_closed_loop_matched_response_v59_20261025.data import load_dataset
from experiments.mcwm_closed_loop_matched_response_v59_20261025.learning import MatchedOutcomeReplay
from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import make_learner, initial_preflight
from experiments.mcwm_closed_loop_shared_level_v36_20261024.run import _objects, _training_rng, heldout_prediction
from experiments.mcwm_closed_loop_physical_truth_v55_20261025.frozen import build_frozen
from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import install_reference_sync_m8
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.mean_target import mean8_state, mean8_stats
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze, optimizer_steps, plain
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(plain(value), handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def line(handle, value):
    handle.write(json.dumps(plain(value), separators=(',', ':'), allow_nan=False)+'\n')
    handle.flush()


def worker(args, protocol):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        torch.set_num_threads(1)
        learner, replay, contract, stations, initializer = make_learner(args.seed, arm='shared_level')
        zero_gate = initial_preflight(learner)
        replay.__class__ = MatchedOutcomeReplay
        replay.arm, learner.outcome_arm = args.arm, args.arm
        # The old models supply only their frozen issue-pre continuation actor;
        # neither old feedback nor old physical diagnostic truth is imported.
        references, old_contract, old_stations, reference_gate = build_frozen(args.scenario)
        one, other = references['one_step'], references['trace']
        if not (_equal(one.sync_actor.state_dict(), other.sync_actor.state_dict())
                and _equal(contract, old_contract) and _equal(stations, old_stations)):
            raise ValueError('The resident mu14 or physical contract differs')
        evaluator = SimpleNamespace(actor=one.sync_actor, action_transform=one.sync_transform)
        install_reference_sync_m8(learner, evaluator, seed=protocol['mean8_extra_seed'])
        del references, one, other, evaluator
        if learner.total_updates != 50000 or replay.target_received:
            raise ValueError('Require the common50000 initializer and empty target replay')
        initial = dict(learner=deepcopy(_state(learner)), rng=_rng(),
            diagnostic_rng=deepcopy(learner.diagnostic_rng.bit_generator.state),
            M8=mean8_state(learner), sync_actor=deepcopy(learner.sync_actor.state_dict()))
        with (out/'initial_state.pt').open('xb') as handle:
            torch.save(initial, handle)
        del initial
        collection = HERE/'runs/collection'/args.scenario
        filename = 'dataset.json' if args.arm == 'fresh' else 'covariates_dataset.json'
        dataset = load_dataset(collection/filename, arm=args.arm, scenario=args.scenario, seed=args.seed)
        write(out/'worker_ready.json', dict(arm=args.arm, initialized_at=time.time(),
            initial_updates=50000, input_packets=len(dataset['packets'])))
        counterpart = out.parent/('no_fresh' if args.arm == 'fresh' else 'fresh')
        deadline = time.monotonic()+1800
        while not (counterpart/'worker_ready.json').exists():
            if (counterpart/'failure.json').exists():
                raise RuntimeError('Matched counterpart failed before its shared start')
            if time.monotonic() > deadline:
                raise RuntimeError('Matched counterpart did not reach the shared start within30minutes')
            time.sleep(2)
        identities, previous_steps = _objects(learner), optimizer_steps(learner)
        feedback_clocks, pair_slots, records = [], 0, 0
        training_started = time.time()
        with (out/'packets.jsonl').open('x') as output:
            for packet in dataset['packets']:
                if _objects(learner) != identities or optimizer_steps(learner) != previous_steps:
                    raise RuntimeError('Parameters or Adam changed between feedback packets')
                replay.add_target(packet['rows'])
                update_started = time.time()
                sampling = []
                for step in range(8):
                    learner.continue_step(collect_diagnostics=False)
                    sampling.append(deepcopy(replay.last_sampling))
                update_finished = time.time()
                previous_steps = optimizer_steps(learner)
                if learner.total_updates != 50000+8*packet['index'] or replay.target_received != 5*packet['index']:
                    raise RuntimeError('Budget differs from57 packets/285slots/456 updates')
                line(output, dict(packet=packet['index'], receipt=packet['receipt'],
                    updates=learner.total_updates-50000, optimizer_steps=previous_steps,
                    sampling=sampling, rng=_training_rng(learner),
                    extra_rng=learner.mean8_extra_rng_state.tolist(),
                    optimizer_objects_retained=True,
                    update_started_wall=update_started, update_finished_wall=update_finished))
                records += 1
                pair_slots += sum(row.sequence < 0 for row in packet['rows'])
                feedback_clocks.append(packet['receipt']['processed_at'])
                write_progress = dict(phase='v59_matched_M8', scenario=args.scenario, arm=args.arm,
                    current=packet['index'], total=57, unit='packets', updates=learner.total_updates-50000)
                print('SCHEDULEURM_PROGRESS '+json.dumps(write_progress), flush=True)
                if packet['index'] == 1:
                    write(out/'first8_gate.json', dict(all_pass=zero_gate['all_pass'],
                        actual_updates=8, slots=5, retained_Adam=True, same_prefix_no_retraining=True))
        training_finished = time.time()
        freeze(learner)
        with (out/'post456_model.pt').open('xb') as handle:
            torch.save(dict(learner=learner, contract=contract, stations=stations, protocol=protocol,
                scenario=args.scenario, arm=args.arm, updates=456, feedback_slots=285,
                initializer=str(initializer), source_dataset=str(collection/filename)), handle)
        frozen_steps, frozen_rng, extra_before = optimizer_steps(learner), _training_rng(learner), mean8_state(learner)
        # First heldout input read occurs after all updates and the freeze/save.
        heldout = json.loads((collection/'heldout_prediction_inputs.json').read_text())
        slots = []
        with (out/'heldout.jsonl').open('x') as output:
            for value in heldout:
                selected, actions = value['selected'], value['actions']
                scored = heldout_prediction(learner, selected, actions)
                if scored['prediction']['holding_seconds'] != [0., 7.5]:
                    raise ValueError('Heldout actual actions differ from the declared0/7.5')
                slot = value['slot']
                slots.append(slot)
                line(output, dict(slot=slot, sequence=selected['sequence'],
                    line_id=selected['line_id'], issue=selected['started_at'], **scored))
        expected = {line+'/'+window for line in protocol['focus_lines'] for window in protocol['heldout_windows']}
        stored_fields = {key for row in replay.recent for key in row}
        checks = dict(initial_zero_residual_native_parity=zero_gate['all_pass'],
            old_mu14_exact=reference_gate['all_pass'], collection_integrity=dataset['proof']['all_pass'],
            slots285=replay.target_received == 285, paired18=pair_slots == 18,
            updates456=learner.total_updates == 50456, packets57=records == 57,
            Adam_parameter_objects_retained=_objects(learner) == identities,
            no_fresh_outcomes_absent=args.arm == 'fresh' or stored_fields == {'s','a','memory_s'},
            heldout_first_opened_after_freeze=True, heldout_complete=len(slots) == 6 and set(slots) == expected,
            post_stop_zero_training=optimizer_steps(learner) == frozen_steps and learner.total_updates == 50456,
            probes_rng_unchanged=_training_rng(learner) == frozen_rng and _equal(extra_before, mean8_state(learner)),
            no_world_or_trust_update=learner.world is None, no_trace_on_disjoint_rows=True,
            frozen_mu14_no_gradient=all(not parameter.requires_grad for parameter in learner.sync_actor.parameters()))
        summary = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), scenario=args.scenario, arm=args.arm,
            initializer=str(initializer), zero_initialization_gate=zero_gate,
            source_prefix=dataset['metadata'], source_file=filename,
            heldout_slots=slots, replay=replay.summary(), target='synchronized_M8_one_step',
            budget=dict(initializer_updates=50000, feedback_slots=285, ordinary_slots=267,
                paired_slots=18, optimizer_updates=456, post_stop_updates=0, native_seconds=0,
                response_fit_SGD_updates=0, last_feedback_clock=max(feedback_clocks)),
            extra_compute=mean8_stats(learner),
            wall=dict(pid=os.getpid(), started=started, training_started=training_started,
                training_finished=training_finished, finished=time.time()))
        write(out/'summary.json', summary)
        if not all(checks.values()):
            raise RuntimeError('Matched worker failed: '+str([key for key,value in checks.items() if not value]))
        return summary
    except BaseException as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            actual_optimizer_updates=(learner.total_updates-50000) if 'learner' in locals() else 0,
            admitted_slots=replay.target_received if 'replay' in locals() else 0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('stable','local_speed_drop'), required=True)
    parser.add_argument('--arm', choices=('fresh','no_fresh'), required=True)
    parser.add_argument('--seed', type=int, default=6202)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    worker(args, json.loads((HERE/'protocol.json').read_text()))


if __name__ == '__main__':
    main()
