"""Independent v95 calibration-cohort collection with the frozen v98 bias identity."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_supervision_plan_v84_20261027.run import (
    snapshot, fork_children, NativePrefixTrace, install_scenario, _equal,
    create_target_env, selected_state, contract, physical)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.action_pair import choose_pair
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.stream import CollectionStream
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.stream import NativeContextReader, issue_binding
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import action_support, service_state
from experiments.mcwm_closed_loop_issue_conditions_v91_20261028.conditions import CorridorHistory
from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.selection import (
    LINES, context_reason, frozen_predictions, lock_first, model_support)
from experiments.mcwm_closed_loop_v1.stream import EI
import numpy as np
import torch

ROLES = {'low':'reference', 'high':'candidate'}
ROOT_KEYS = ('started_at','sequence','trip_id','station_id')


def pair_integrity(checks):
    """A measured physical no-op retains its label; technical failures still stop."""
    return all(value for name, value in checks.items() if name != 'physical_departure_separation')


def exact_tail_boundary(metrics):
    """Persist an existing actual18300 ledger only; never select a nearby tick."""
    return next((deepcopy(row) for row in metrics if row['time_seconds'] == 18300.), None)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.writing')
    with pending.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',',':'), allow_nan=False)
        handle.write('\n')
    if path.exists():
        raise FileExistsError(path)
    pending.rename(path)


def run(out, scenario, external_seed, gate_path=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root_out = out
    p = json.loads((HERE/'protocol.json').read_text())
    source = json.loads((HERE/'source_protocol.json').read_text())
    model = json.loads((HERE/'source_model.json').read_text())
    source_fit = json.loads((HERE/'source_fit.json').read_text())
    audit = json.loads((HERE/'freeze_audit.json').read_text())
    if not (p['source_initializer'] == source['frozen_reference']['initializer_seed'] == 6207
            and p['cohort'] == 'C'
            and external_seed in p['external_seeds'] == source['independent_cohorts']['C']
            and scenario in p['scenarios']
            and audit['all_pass'] and all(audit['checks'].values())
            and model['integrity']['all_pass']
            and model['common_support'] == source['frozen_common_support']
            and source_fit['integrity']['all_pass'] and all(source_fit['integrity']['checks'].values())
            and source_fit['primary_model'] == 'service_history6'
            and source_fit['bias_model']['q_range'] == p['q_range_for_scaling_only']
            and source_fit['candidate_deployment_domain'] == p['candidate_deployment_domain']):
        raise RuntimeError('Frozen v95 C cohort, reference, base model and v98 bias must agree')
    frozen_bias = dict(source_fit=p['source_fit'], primary_model=source_fit['primary_model'],
        bias_model=deepcopy(source_fit['bias_model']), source_summary=source_fit['source_summary'],
        source_protocol=source_fit['source_protocol'])
    seed = 6207
    gate_path = Path(gate_path or HERE/'first_pair_gate.json')
    producer = external_seed == p['external_seeds'][0] and scenario == 'stable'
    scenario_cap = p['budget']['scenario_cap']
    if not producer:
        deadline = time.monotonic()+21600
        while not gate_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('First formal pair gate unavailable; no native prefix started')
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_first_pair', current=0,
                total=1, unit='technical_gate', external_seed=external_seed, scenario=scenario)), flush=True)
            time.sleep(30)
        if not json.loads(gate_path.read_text())['all_pass']:
            value = dict(status='not_started', scenario=scenario, seed=seed, external_seed=external_seed,
                prefix_id=f'v99/{external_seed}/{scenario}', integrity=dict(all_pass=False,
                checks=dict(first_pair_gate=False)), first_pair_gate=json.loads(gate_path.read_text()),
                selection={line:dict(selected=None, status='missing', reasons={}) for line in LINES},
                coverage={}, pairs=[], training_started=False, models_fitted=False,
                frozen_bias=frozen_bias,
                predictions_used_for_selection=False, labels_used_for_selection=False,
                budget=dict(prefixes=0, suffixes=0, native_prefix_seconds=0, suffix_native_seconds=0,
                    total_native_seconds=0, total_native_seconds_is_lower_bound=False,
                    unknown_suffix_costs=0, labels_observed=0,
                    new_initialization_updates=0, prefix_training_updates=0, suffix_updates=0,
                    labels_fed_back=0, new_fits=0, new_neural_updates=0, v87_truth_used=0, details=[]))
            write(out/'collector_summary.json', value)
            return value
    env = hook = stream = trace = probe = controller = context_reader = history = None
    branch = root = choice = pair_path = native_info = service_row = None
    pairs, coverage = [], {}
    selection = {line:dict(selected=None, attempted=0, reasons={}, status='missing') for line in LINES}
    observation_counts = dict(reads=0, observed=0, admissible=0, unsupported=0)
    observation_checks = dict(clock_unchanged=True, ledger_unchanged=True, rng_unchanged=True,
        issue_clock_exact=True, event_clock_exact=True, context_before_action=True)
    clock_exact = continuation_exact = True
    collection_halted = False
    root_actions = 0
    native_started_wall = None
    local_first_pair_gate = None
    reference_reproduction = {}
    started = time.monotonic()

    def counts():
        return dict(actor_updates=0, critic_updates=0, value_updates=0, alpha_updates=0,
            D_refits=0, world_updates=0, trust_updates=0, admitted_labels=0)

    def progress(phase):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase,
            current=stream.now if stream else 0., total=24000 if branch else 12600,
            unit='sim_seconds', external_seed=external_seed, scenario=scenario,
            locked_lines=len(coverage), pairs_complete=len(pairs), elapsed_seconds=time.monotonic()-started)), flush=True)

    def declare_gate(record=None, reason=None):
        nonlocal local_first_pair_gate
        value = dict(status='passed' if record and record['all_pass'] else 'held',
            all_pass=bool(record and record['all_pass']), checks=record['checks'] if record else {},
            external_seed=external_seed, scenario=scenario, line_id=record['line_id'] if record else None,
            root=record['root'] if record else None, reason=reason,
            new_pair_in_formal_cohort=record is not None, labels_used_for_gate=False,
            cost_sign_used_for_gate=False, first_pair_reused=record is not None)
        if local_first_pair_gate is None:
            local_first_pair_gate = value
            write(root_out/'first_pair_gate.json', value)
        if producer and not gate_path.exists():
            write(gate_path, value)
        return value

    def policy(state):
        nonlocal clock_exact
        event_clock = float(env._current_events[0].sim_time)
        clock_exact &= bool(state[0] == np.float32(np.float32(event_clock)/24000))
        return controller(state)

    def selector(event):
        nonlocal branch, out, trace, probe, root, choice, pair_path, native_info, service_row
        nonlocal root_actions, native_started_wall, collection_halted
        line = event['line_id']
        if (branch is not None or line not in LINES or selection[line]['selected'] is not None
                or not 12000 <= event['started_at'] < 12600):
            return None
        slot = selection[line]
        slot['attempted'] += 1
        api = importlib.import_module(env._bridge.__class__.__module__).traci
        try:
            step = float(api.simulation.getDeltaT())
            action_step = float(api.vehicle.getActionStepLength(event['trip_id']))
            at_stop = bool(api.vehicle.isAtBusStop(event['trip_id']))
        except Exception:
            step = action_step = None
            at_stop = False
        if not at_stop or step is None or action_step is None:
            reason = 'native_stop_or_clock_unavailable'
            slot['reasons'][reason] = slot['reasons'].get(reason, 0)+1
            return None
        proposed = choose_pair(controller.action_transform, event['s'], controller.last_proposal,
            simulation_step=step, action_step=action_step)
        support = action_support(proposed['reference_hold_seconds'], proposed['candidate_hold_seconds'], step, action_step)
        if not support['supported']:
            reason = support['reason']
            slot['reasons'][reason] = slot['reasons'].get(reason, 0)+1
            return None
        identity = {name:event[name] for name in ROOT_KEYS}
        lock_first(selection, line, identity)
        if not np.array_equal(event['a'], np.asarray(proposed['reference_action'], np.float32)):
            raise RuntimeError('Original actor/F reference differs from issued action')
        before = dict(clock=float(api.simulation.getTime()), ledger=snapshot(stream._snapshot()), rng=_rng())
        ctx = context_reader(event)
        state = service_state(ctx)
        condition = history.snapshot(event['started_at'])
        binding = issue_binding(event)
        after = dict(clock=float(api.simulation.getTime()), ledger=snapshot(stream._snapshot()), rng=_rng())
        checked = dict(clock_unchanged=before['clock'] == after['clock'],
            ledger_unchanged=before['ledger'] == after['ledger'], rng_unchanged=_equal(before['rng'], after['rng']),
            issue_clock_exact=before['clock'] == after['clock'] == event['started_at'],
            event_clock_exact=state['event_clock'] == binding['event_clock'], context_before_action=True)
        for key, value in checked.items():
            observation_checks[key] &= value
        observation_counts['reads'] += 1
        observation_counts['observed'] += int(state['observed'])
        if not all(checked.values()):
            raise RuntimeError('Issue observation changed native clock, ledger or RNG')
        reason = context_reason(state, condition, line, model, event['started_at'])
        training_support = model_support(model, line, condition['value'])
        predictions = None if reason else frozen_predictions(model, line, state, condition['value'],
            proposed['reference_hold_seconds'], proposed['candidate_hold_seconds'])
        service_row = dict(line_id=line, root=identity, context=ctx, service=state, condition=condition,
            binding=binding, observation_checks=checked, predictions=predictions, training_support=training_support,
            prediction_recorded_at=event['started_at'], before_suffixes=True,
            ref_hold=proposed['reference_hold_seconds'], cand_hold=proposed['candidate_hold_seconds'],
            sim_step_seconds=step, action_step_seconds=action_step, status='unsupported' if reason else 'supported',
            reason=reason)
        coverage[line] = deepcopy(service_row)
        if reason:
            observation_counts['unsupported'] += 1
            slot.update(status='unsupported', reason=reason)
            slot['reasons'][reason] = slot['reasons'].get(reason, 0)+1
            collection_halted = True
            if not pairs:
                declare_gate(reason='first_locked_slot_unsupported:'+reason)
            return None
        observation_counts['admissible'] += 1
        if collection_halted:
            slot['status'] = 'not_queried_after_hold'
            return None
        slot['status'] = 'selected'
        choice = proposed
        root = selected_state(event, stream._snapshot())
        root.update(reference_proposal=controller.last_proposal.tolist(), service_binding=binding,
                    service=state, history_condition=condition, predictions=predictions,
                    training_support=training_support)
        native_info = dict(sim_step_seconds=step, action_step_seconds=action_step, at_stop=at_stop)
        pair_path = root_out/'pairs'/line
        write(pair_path/'root.json', dict(root=root, reference=reference, choice=choice,
            frozen_bias=frozen_bias,
            context=ctx, service=state, condition=condition, predictions=predictions,
            training_support=training_support,
            prediction_recorded_at=event['started_at'], observation_checks=checked,
            before_suffixes=True, **native_info))

        def prepare_child(role):
            nonlocal branch, out, trace, probe, native_started_wall
            branch, out = role, pair_path/role
            old_rows = trace.observed_issues
            trace.close()
            trace = NativePrefixTrace(out/'trace.jsonl')
            trace.observed_issues = old_rows
            stream.enter_suffix()
            probe = physical.install(env._bridge, root)
            native_started_wall = time.time()

        receipt = fork_children(pair_path, trace, lambda:stream.now, counts, prepare_child, timeout_seconds=3600)
        if isinstance(receipt, str):
            root_actions += 1
            return np.asarray(choice[ROLES[branch]+'_action'], np.float32)
        arms = {ROLES[role]:json.loads((pair_path/role/'summary.json').read_text()) for role in ROLES}
        ref, cand = arms['reference'], arms['candidate']
        departures = [value['physical']['departure']['clock_seconds'] for value in arms.values()]
        observed = all(value is not None for value in departures)
        checks = dict(same_complete_root=ref['root'] == cand['root'] == root,
            same_fixed_reference=ref['reference'] == cand['reference'] == reference,
            same_frozen_bias=ref['frozen_bias'] == cand['frozen_bias'] == frozen_bias,
            same_external_seed=ref['external_seed'] == cand['external_seed'] == external_seed,
            original_actor_root_exact=ref['executed_action'] == choice['reference_action'],
            predeclared_candidate_exact=cand['executed_action'] == choice['candidate_action'],
            command_support=support['supported'], actual_departures_observed=observed,
            physical_departure_separation=bool(observed and abs(departures[1]-departures[0]) >= step),
            actual_next_arrivals=all(value['physical']['coverage']['next_arrival_observed'] for value in arms.values()),
            both_real_terminal=all(value['final_clock'] == 24000 for value in arms.values()),
            nonzero_duration=ref['remaining']['dt'] == cand['remaining']['dt'] > 0,
            label_maturity=all(value['remaining']['available_at'] == 24030 for value in arms.values()),
            zero_suffix_training=all(value['suffix_updates'] == 0 for value in arms.values()),
            zero_truth_feedback=all(value['suffix_admitted_labels'] == 0 for value in arms.values()),
            native_fork=receipt['all_pass'], actual_parallel=receipt['overlap_seconds'] > 0,
            parent_actor_frozen=_equal(actor_before, controller.snapshot_state()),
            service_context_before_action=all(service_row['observation_checks'].values()),
            same_issue_context=all(value['root']['service'] == service_row['service'] for value in arms.values()),
            same_historical_condition=all(value['root']['history_condition'] == condition for value in arms.values()),
            history_strictly_before_issue=condition['historical_only'] and condition['latest_completed_at'] < root['started_at'],
            no_replacement_root=True)
        initial = ref['root']['issue_ledger']
        full_wait = initial['accrued_wait_seconds']+ref['remaining']['wait']
        full_ride = initial['accrued_in_vehicle_seconds']+ref['remaining']['ride']
        record = dict(status='complete', all_pass=pair_integrity(checks), checks=checks, line_id=line,
            physical_noop=bool(observed and not checks['physical_departure_separation']),
            scenario=scenario, external_seed=external_seed, root={name:root[name] for name in ROOT_KEYS},
            context=ctx, service=service_row['service'], condition=condition, predictions=predictions,
            source_fit=frozen_bias['source_fit'],
            training_support=training_support,
            prediction_recorded_at=root['started_at'], before_suffixes=True, **native_info,
            reference_hold_seconds=choice['reference_hold_seconds'], candidate_hold_seconds=choice['candidate_hold_seconds'],
            observation_checks=service_row['observation_checks'], overlap_seconds=receipt['overlap_seconds'],
            remaining={role:value['remaining'] for role,value in arms.items()},
            tail_boundary={role:value['tail_boundary'] for role,value in arms.items()},
            tail_boundary_status={role:value['tail_boundary_status'] for role,value in arms.items()},
            delta={name:cand['remaining'][name]-ref['remaining'][name] for name in ('wait','ride','cost')},
            physical={role:dict(departure_clock=value['physical']['departure']['clock_seconds'],
                next_arrival_clock=value['physical']['next_same_vehicle_arrival']['clock_seconds'],
                missing=value['physical']['missing']) for role,value in arms.items()},
            reference_full_endpoint=dict(wait=full_wait, ride=full_ride, cost=2*full_wait+full_ride))
        write(pair_path/'pair_summary.json', record)
        pairs.append(record)
        if local_first_pair_gate is None:
            declare_gate(record=record)
        if not record['all_pass']:
            raise RuntimeError('Formal pair failed physical integrity; no replacement')
        slot['status'] = 'complete'
        progress('formal_pair_complete_parent_resumes')
        return None

    def issue(event):
        nonlocal continuation_exact
        event['proposed_a'] = controller.last_proposal.copy()
        if branch is not None and event['sequence'] != root['sequence']:
            continuation_exact &= bool(np.array_equal(event['a'], controller.action_transform.numpy(event['s'], controller.last_proposal)))
        trace.observe(event, stream._snapshot())
        history.observe(event)

    def spent_budget():
        rows = []
        for line in LINES:
            for role in ROLES:
                directory = root_out/'pairs'/line/role
                if not directory.exists():
                    continue
                if (directory/'summary.json').exists():
                    value = json.loads((directory/'summary.json').read_text())
                    rows.append(dict(line_id=line, role=ROLES[role], status=value['status'], seconds=value['remaining']['dt']))
                elif (directory/'failure.json').exists():
                    value = json.loads((directory/'failure.json').read_text())
                    rows.append(dict(line_id=line, role=ROLES[role], status='failed', seconds=value.get('suffix_native_seconds')))
                else:
                    rows.append(dict(line_id=line, role=ROLES[role], status='unresolved', seconds=None))
        prefix = stream.now if stream is not None else 0.
        suffix = sum(row['seconds'] for row in rows if row['seconds'] is not None)
        unknown = sum(row['seconds'] is None for row in rows)
        return dict(prefixes=1 if stream else 0, suffixes=len(rows), native_prefix_seconds=prefix,
            suffix_native_seconds=suffix, total_native_seconds=prefix+suffix,
            total_native_seconds_is_lower_bound=unknown > 0, unknown_suffix_costs=unknown,
            labels_observed=sum(row['status'] == 'complete' for row in rows), new_initialization_updates=0,
            prefix_training_updates=0, suffix_updates=0, labels_fed_back=0, new_fits=0,
            new_neural_updates=0, v87_truth_used=0, details=rows)

    def finish_parent(error=None):
        spent = spent_budget()
        if local_first_pair_gate is None:
            declare_gate(reason='no_supported_first_pair' if error is None else 'native_failure_before_first_pair')
        checks = dict(three_locked_slots=len(coverage) == 3, three_formal_pairs=len(pairs) == 3,
            all_slots_supported=all(value.get('status') != 'unsupported' for value in coverage.values()),
            pair_integrity=all(row['all_pass'] for row in pairs),
            actor_frozen=controller is not None and _equal(actor_before, controller.snapshot_state()),
            original_reference_actions_reproduced=bool(reference_reproduction) and all(reference_reproduction.values()),
            same_reference_path=controller is not None and reference['source_actor_path'] == model['reference_source_paths'][scenario],
            freeze_audit_pass=audit['all_pass'], issue_observation_unchanged=all(observation_checks.values()),
            parent_prefix_limit=stream is not None and stream.now == 12600 and stream.prefix_limit_reached,
            native_budget=spent['total_native_seconds'] <= scenario_cap,
            known_spend=spent['unknown_suffix_costs'] == 0, zero_training_and_admission=True,
            all_issues_logged=trace is not None and stream is not None
                and trace.observed_issues == stream.issued_count > 0,
            actor_real_clock=clock_exact, first_pair_gate=local_first_pair_gate['all_pass'], error_free=error is None)
        value = dict(status='failed' if error is not None else 'complete' if all(checks.values()) else 'held',
            integrity=dict(all_pass=all(checks.values()), checks=checks), scenario=scenario, seed=seed,
            external_seed=external_seed, prefix_id=f'v99/{external_seed}/{scenario}', departed_passengers=15079,
            reference_snapshot=reference if controller else None, reference_reproduction=reference_reproduction,
            frozen_bias=frozen_bias,
            coverage=coverage, selection=selection, observation_counts=observation_counts,
            observation_checks=observation_checks, budget=spent, pairs=pairs, training_started=False,
            models_fitted=False, predictions_used_for_selection=False, labels_used_for_selection=False,
            collection_halted_on_unsupported=collection_halted,
            first_pair_gate=json.loads(gate_path.read_text()), local_first_pair_gate=local_first_pair_gate,
            history_summary=history.summary() if history else None, error=error,
            prefix_reproduction=dict(all_pass=checks['all_issues_logged'],
                checks=dict(all_issues_logged=checks['all_issues_logged']), reference_trace_path=None,
                observed_issues=trace.observed_issues if trace else 0, matched_issues=0))
        if error is not None:
            value['failure'] = error
        write(root_out/'collector_summary.json', value)
        progress('calibration_collection_finished_no_fit')
        return value

    try:
        torch.set_num_threads(1)
        controller, mdp, stations, reference = load_reference(scenario)
        reference['heldout_external_seed'] = external_seed
        actor_before = controller.snapshot_state()
        for version in ('mcwm_closed_loop_reference_collection_v84_20261027', 'mcwm_closed_loop_cost_advantage_v85_20261027'):
            for line in LINES:
                declaration = json.loads((HERE.parent/version/'runs'/scenario/'pairs'/line/'root.json').read_text())
                actual = controller(np.asarray(declaration['root']['s'], np.float32))
                reference_reproduction[version+'/'+line] = bool(
                    np.array_equal(actual, np.asarray(declaration['choice']['reference_action'], np.float32))
                    and np.array_equal(controller.last_proposal, np.asarray(declaration['root']['reference_proposal'], np.float32)))
        if not all(reference_reproduction.values()):
            raise RuntimeError('Frozen actor/F fails existing issue-input reproduction')
        controller.last_proposal = None
        trace = NativePrefixTrace(root_out/'prefix_trace.jsonl')
        history = CorridorHistory(EI)
        env = create_target_env(os.environ['THESIS_SUMO_ROOT'],
            Path(os.environ['THESIS_H2O_ROOT'])/'bus_h2o/network_data/a_sorted_busline_edge.xml',
            episode_seconds=24000, seed=external_seed)
        stream = CollectionStream(env, stations, mdp, domain='target', selector=selector,
            delay_seconds=30, execute_transform=False, on_issue=issue, prefix_limit_seconds=12600)
        stream.reset(); stream.enable_prefix_limit()
        context_reader = NativeContextReader(env)
        hook = install_scenario(env, scenario, shock_at=9000, factor=.65, metric_every=300)
        last_progress = time.monotonic()
        while not stream.done and not stream.prefix_limit_reached:
            stream.step(policy); stream.release_intervals()
            if time.monotonic()-last_progress >= 45:
                progress('frozen_suffix' if branch else 'new_frozen_prefix')
                last_progress = time.monotonic()
        if branch is None:
            return finish_parent()
        remaining = contract.remaining_cost(root['issue_ledger'], snapshot(stream.final_snapshot))
        tail_boundary = exact_tail_boundary(hook.metrics)
        observation = probe.summary(endpoint_reached=True)
        applied = observation['native_action']
        role = ROLES[branch]
        hold = choice[role+'_hold_seconds']
        checks = dict(real_terminal=stream.now == 24000 and stream.done,
            root_action_once=applied is not None and root_actions == 1,
            actual_command_exact=applied is not None and abs(applied['applied_hold_seconds']-hold) <= 1e-5,
            own_frozen_continuation=continuation_exact, actor_real_clock=clock_exact,
            actor_frozen=_equal(actor_before, controller.snapshot_state()), zero_native_fits=counts()['D_refits'] == 0,
            zero_admission=counts()['admitted_labels'] == 0, native_nonzero_duration=remaining['dt'] > 0,
            issue_context_saved_before_action=all(service_row['observation_checks'].values()),
            history_condition_strictly_past=service_row['condition']['historical_only'],
            label_not_released=not contract.label_available(remaining['available_at'], stream.now),
            all_issues_logged=trace.observed_issues == stream.issued_count)
        value = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), root=root, reference=reference, role=role,
            frozen_bias=frozen_bias,
            external_seed=external_seed, executed_action=choice[role+'_action'], executed_hold_seconds=hold,
            **native_info, final_clock=stream.now, suffix_updates=0, suffix_admitted_labels=0,
            remaining=remaining, physical=observation, tail_boundary=tail_boundary,
            tail_boundary_status='observed_exact' if tail_boundary is not None else 'missing_exact18300',
            execution_wall=dict(pid=os.getpid(), native_started_wall=native_started_wall, native_finished_wall=time.time()))
        write(out/'summary.json', value)
        if not all(checks.values()):
            raise RuntimeError('Native calibration-cohort child integrity failed')
        progress('complete'); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    except Exception as error:
        value = dict(type=type(error).__name__, message=str(error), role=ROLES.get(branch),
            native_time=stream.now if stream else None, labels_fed_back=0)
        if branch is None:
            value['consumed_budget'] = spent_budget()
        else:
            value['suffix_native_seconds'] = max(0., stream.now-root['started_at'])
        write(out/'failure.json', value)
        if branch is not None:
            sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
        if not (root_out/'collector_summary.json').exists():
            finish_parent(error=value)
        raise
    finally:
        if probe is not None: probe.close()
        if trace is not None: trace.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenario', choices=('stable','local_speed_drop'), required=True)
    parser.add_argument('--external-seed', type=int, required=True)
    parser.add_argument('--gate-path', type=Path)
    args = parser.parse_args()
    run(args.out, args.scenario, args.external_seed, args.gate_path)
