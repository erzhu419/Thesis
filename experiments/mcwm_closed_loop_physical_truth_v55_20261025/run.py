"""New common prefixes and paired native heldout truth for frozen v54 critics."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import torch
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.bootstrap import formal
from experiments.mcwm_closed_loop_physical_truth_v55_20261025.frozen import (
    build_frozen, executable_candidates, score_frozen, optimizer_steps, plain)
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.selection import FirstEligible, MemoryForkStream
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.suffix import NativeTruthPair, selected_state
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.collect import StochasticController
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_v1.stream import create_target_env
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(plain(value), handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


class HeldoutPair(NativeTruthPair):
    """Keep the common private actor stream explicit across the native fork."""
    def __init__(self, *args, controller, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.actor_rng_before = controller.rng.clone()
        self.actor_draws_before = controller.draws
        self.actor_inherited = None

    def fork(self, event, stream, trace, prepare_child=None):
        def prepare(arm):
            self.actor_inherited = (torch.equal(self.actor_rng_before, self.controller.rng)
                and self.controller.draws == self.actor_draws_before)
            if not self.actor_inherited:
                raise RuntimeError('Suffix private common actor RNG differs from its root')
        replacement = super().fork(event, stream, trace, prepare_child=prepare)
        if not self.in_child:
            unchanged = (torch.equal(self.actor_rng_before, self.controller.rng)
                and self.controller.draws == self.actor_draws_before)
            self.receipt['parent_private_actor_rng_unchanged'] = unchanged
            self.receipt['children_private_actor_rng_inherited'] = all(
                value['integrity']['checks']['private_actor_rng_inherited'] for value in self.values.values())
            self.receipt['all_pass'] &= unchanged and self.receipt['children_private_actor_rng_inherited']
            write(self.out/'private_actor_fork.json', self.receipt)
            if not self.receipt['all_pass']:
                raise RuntimeError('Private actor RNG or paired native fork failed')
        return replacement

    def build_summary(self, stream):
        result = super().build_summary(stream)
        checks = result['integrity']['checks']
        checks['private_actor_rng_inherited'] = self.actor_inherited is True
        result['private_action_rng'] = dict(seed=self.controller.seed,
            inherited_root_draw=self.actor_draws_before, final_draw=self.controller.draws)
        result['integrity']['all_pass'] = all(checks.values())
        result['status'] = 'complete' if result['integrity']['all_pass'] else 'integrity_failed'
        return result


def spent(out, stream):
    """Preserve observed native work on failure; never fill missing truth with zero."""
    clock = (float(stream.now) if stream is not None and stream.initial_snapshot is not None else 0.)
    suffixes = []
    for state_root in sorted((out/'states').glob('*')):
        for arm in ('low', 'high'):
            branch = state_root/arm
            if not branch.exists():
                continue
            summary, failure = branch/'summary.json', branch/'failure.json'
            if summary.exists():
                value = json.loads(summary.read_text())
                suffixes.append(dict(path=str(branch),status=value['status'],
                    native_seconds=value['budget']['suffix_sim_seconds'],exact=True))
            elif failure.exists():
                value = json.loads(failure.read_text())
                seconds = value.get('failure_budget', {}).get('suffix_sim_seconds')
                suffixes.append(dict(path=str(branch),status='failed',native_seconds=seconds,exact=seconds is not None))
            else:
                suffixes.append(dict(path=str(branch),status='started_without_complete_receipt',
                    native_seconds=None,exact=False))
    known = [row['native_seconds'] for row in suffixes if row['native_seconds'] is not None]
    return dict(new_native_prefixes=int(stream is not None), prefix_native_seconds=clock,
        observed_suffixes=len(suffixes), observed_suffix_native_seconds=sum(known),
        unknown_suffix_costs=sum(row['native_seconds'] is None for row in suffixes),
        new_training_labels=0, new_optimizer_updates=0, suffixes=suffixes)


def run(args, protocol):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.time()
    env = stream = trace = hook = memory = pair = None
    controller = None
    try:
        learners, contract, stations, model_gate = build_frozen(args.scenario)
        one, trace_model = learners['one_step'], learners['trace']
        mu_checks = dict(shared_frozen_actor_exact=_equal(one.sync_actor.state_dict(), trace_model.sync_actor.state_dict()),
            F_contract_exact=_equal(one.sync_transform.contract, trace_model.sync_transform.contract)
                and _equal(one.sync_transform.contract, one.action_transform.contract))
        model_gate['continuation_mu14_gate'] = dict(all_pass=all(mu_checks.values()), **mu_checks,
            actor_updates=50024, provenance='v54 saved sync_actor from v53 actual post24_from14')
        if not all(mu_checks.values()):
            raise RuntimeError('The two frozen critics require one identical issue-pre continuation actor/F')
        write(out/'model_gate.json', model_gate)
        before = {arm: deepcopy(_state(learner)) for arm,learner in learners.items()}
        before_steps = {arm: optimizer_steps(learner) for arm,learner in learners.items()}
        continuation = SimpleNamespace(actor=one.sync_actor, action_transform=one.sync_transform,
            device=one.device, total_updates=50024)
        controller = StochasticController(continuation, seed=protocol['private_action_seed'])
        controller.snapshot = 'v53_actual_post24_from14'
        selection, records, recorded = FirstEligible(protocol), [], set()
        clock_checks, clock_exact = 0, True

        def counts():
            value = dict(new_optimizer_updates=0, new_truth_training_labels=0,
                world_updates=0, trust_updates=0, controller_updates=50024)
            for arm, learner in learners.items():
                value[arm+'_updates'] = learner.total_updates
                value[arm+'_feedback'] = learner.replay_buffer.target_received
                for name, step in optimizer_steps(learner).items():
                    value[arm+'_'+name] = step['max'] or 0.
            return value

        def policy(state):
            nonlocal clock_checks, clock_exact
            clock = float(env._current_events[0].sim_time)
            expected = np.float32(np.float32(clock)/protocol['horizon_seconds'])
            clock_checks += 1
            clock_exact &= bool(state[0] == expected)
            return controller(state)

        def selector(event):
            nonlocal pair
            if pair is not None and pair.in_child:
                return None
            slot = selection.active(event)
            if slot is None:
                return None
            choices = {arm: executable_candidates(learner.action_transform, event['s'],
                protocol['paired_holds_seconds']) for arm,learner in learners.items()}
            if choices['one_step'] != choices['trace']:
                raise RuntimeError('The frozen models changed candidate execution F')
            candidates = choices['one_step']
            if not candidates['physically_distinct'] or not all(candidates['supported']):
                selection.slots[slot]['attempted'] += 1
                selection.slots[slot]['reasons']['requested_F_actions_unsupported'] += 1
                return None
            api = importlib.import_module(env._bridge.__class__.__module__).traci
            try:
                stop = bool(api.vehicle.isAtBusStop(event['trip_id']))
                delta = float(api.simulation.getDeltaT())
                action_step = float(api.vehicle.getActionStepLength(event['trip_id']))
            except Exception:
                stop = delta = action_step = None
            if not selection.consider(slot, event, candidates['holding_seconds'],
                    at_stop=stop, simulation_step=delta, action_step=action_step):
                return None
            event['proposed_a'] = np.asarray(controller.last['proposal'], np.float32)
            selected = selected_state(event, stream._snapshot())
            rng_before = _rng()
            predictions = score_frozen(learners, event['s'], event['memory_s'],
                candidates['executed_actions'])
            if not _equal(rng_before, _rng()):
                raise RuntimeError('Frozen heldout scoring changed the native prefix RNG')
            trace.observe(event, stream._snapshot())
            recorded.add(event['sequence'])
            state_out = out/'states'/slot.replace('/', '_')
            expected = dict(selected=selected,
                actions={arm: [candidates['executed_actions'][index]] for index,arm in enumerate(('low','high'))},
                holds=dict(zip(('low','high'), candidates['holding_seconds'])))
            pair = HeldoutPair(state_out, expected, counts, protocol['native_external_seed'], controller=controller)
            replacement = pair.fork(event, stream, trace)
            if pair.in_child:
                return replacement
            record = dict(slot=slot, selected=selected, prediction=predictions,
                actions=expected['actions'], holds=expected['holds'], candidates=candidates,
                native_feasibility=dict(at_stop=stop, simulation_step=delta, action_step=action_step),
                pair_receipt=pair.receipt, comparison=pair.comparison,
                branch_paths={arm: str(state_out/arm/'summary.json') for arm in ('low','high')})
            write(state_out/'state_summary.json', record)
            records.append(record)
            (out/'progress.json').write_text(json.dumps(dict(status='running',
                native_time=stream.now,states=len(records),last_slot=slot,new_optimizer_updates=0))+'\n')
            if len(records) == 1:
                write(out/'preflight_gate.json', dict(all_pass=True,
                    old_model_reproduction=model_gate['all_pass'], first_pair=slot,
                    native_first_pair_reused=True, duplicate_prefixes=0, duplicate_suffixes=0,
                    actual_parallel=pair.receipt['all_pass'], new_optimizer_updates=0))
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v55_paired_truth',
                current=stream.now,total=24000,unit='sim_seconds',slot=slot,states=len(records))),flush=True)
            return None

        def issue(event):
            event['proposed_a'] = np.asarray(controller.last['proposal'], np.float32)
            if pair is not None and pair.in_child:
                pair.observe_issue(event, stream)
            elif event['sequence'] not in recorded:
                trace.observe(event, stream._snapshot())

        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000, seed=protocol['native_external_seed'])
        stream = MemoryForkStream(env, stations, contract, domain='target', selector=selector,
            fork_duration_seconds=300, delay_seconds=30., execute_transform=False, on_issue=issue)
        stream.reset()
        memory = stream.memory = ExecutionMemory(env._bridge)
        hook = install_scenario(env, args.scenario, shock_at=9000., factor=.65, metric_every=300.)
        trace = NativePrefixTrace(out/'prefix_trace.jsonl')
        frozen_counts = counts()
        native_started = time.time()
        progress = time.monotonic()
        while not stream.done:
            stream.step(policy)
            stream.release_intervals()
            if pair is not None and pair.in_child:
                pair.observe_progress(stream)
                if stream.cutoff_reached:
                    pair.finish_child(stream)
                continue
            if time.monotonic()-progress >= 45:
                (out/'progress.json').write_text(json.dumps(dict(status='running',
                    native_time=stream.now,states=len(records),new_optimizer_updates=0))+'\n')
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v55_common_prefix',
                    current=stream.now,total=24000,unit='sim_seconds',states=len(records))),flush=True)
                progress = time.monotonic()
        native_finished = time.time()
        hook.capture_final()
        checks = dict(frozen_v54_final_predictions_exact=model_gate['all_pass'],
            original285_slots456_updates=all(learner.total_updates == 50456
                and learner.replay_buffer.target_received == 285 for learner in learners.values()),
            models_and_Adam_unchanged=all(_equal(before[arm], _state(learner))
                and optimizer_steps(learner) == before_steps[arm] for arm,learner in learners.items()),
            common_frozen_mu14=model_gate['continuation_mu14_gate']['all_pass'],
            prefix_and_suffix_zero_training=counts() == frozen_counts,
            full24000_parent=stream.done and stream.now == 24000.,
            full_common_prefix_trace=trace.observed_issues == stream.issued_count,
            actor_real_clock=clock_exact and clock_checks > 0,
            new_external_seed=protocol['native_external_seed'] != protocol['training_external_seed'],
            paired_prefix_seeds_actions_parallel=all(record['pair_receipt']['all_pass'] for record in records),
            causal_issue_memory=memory.summary()['integrity']['all_pass'],
            no_truth_admission=True)
        selection_summary = selection.summary()
        result = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), scenario=args.scenario,
            seed=args.seed, initializer_seed=args.seed,
            native_external_seed=protocol['native_external_seed'], private_action_seed=protocol['private_action_seed'],
            controller_snapshot=controller.snapshot, frozen_model_gate=model_gate, model_rebuild=model_gate,
            selection=selection_summary,
            states=[dict(slot=record['slot'], path=str(out/'states'/record['slot'].replace('/', '_')/'state_summary.json'))
                    for record in records],
            prefix=dict(trace_rows=trace.observed_issues, world_updates=0, trust_updates=0),
            counts=dict(before=frozen_counts, after=counts()),
            budget=dict(reconstruction_updates=0,new_native_prefixes=1,native_prefix_seconds=24000,
                states=len(records),suffixes=2*len(records),suffix_sim_seconds=600*len(records),
                declared_states=4,missing_states=4-len(records),
                new_training_labels=0,new_prefix_optimizer_updates=0,suffix_optimizer_updates=0,
                source_feedback_slots_per_model=285, source_updates_per_model=456,
                duplicate_preflight_prefixes=0,duplicate_preflight_suffixes=0),
            wall=dict(native_started=native_started,native_finished=native_finished,total_seconds=time.time()-started))
        write(out/'summary.json', result)
        (out/'progress.json').write_text(json.dumps(dict(status=result['status'],
            native_time=stream.now,states=len(records),new_optimizer_updates=0))+'\n')
        if not all(checks.values()):
            raise RuntimeError('Native frozen paired diagnostic integrity failed')
        return result
    except BaseException as error:
        failure = dict(status='failed',type=type(error).__name__,message=str(error),scenario=args.scenario,
            native_time=float(stream.now) if stream is not None and stream.initial_snapshot is not None else None)
        if pair is not None and pair.in_child:
            failure['failure_budget'] = dict(suffix_sim_seconds=max(0., float(stream.now)-pair.expected['selected']['started_at']),
                new_training_labels=0,new_optimizer_updates=0)
            write(pair.out/pair.arm/'failure.json', failure)
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        failure['failure_budget'] = spent(out, stream)
        write(out/'failure.json', failure)
        raise
    finally:
        if pair is not None and pair.in_child:
            os._exit(1)
        if memory is not None: memory.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=6202)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    arguments = parser.parse_args()
    if arguments.seed != 6202:
        parser.error('v55 retains the source6202/50000 initializer')
    run(arguments, json.loads((HERE/'protocol.json').read_text()))
