"""Inspect archived fields and source interfaces; never connect to a simulator."""
import ast
from collections import Counter
import json
import os
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET


V75 = 'mcwm_closed_loop_frozen_validation_v75_20261027'
V76 = 'mcwm_closed_loop_mid_policy_exchange_v76_20261027'
RUNTIME = Path('sumo_eval/runtime/H2Oplus')
PASSENGER_FILES = (
    '3_modified_period_workday_passenger_VC_0805.rou.xml',
    '3_modified_transfer_passenger_0805.rou.xml',
)


def _source_lines(path, tokens, scope=None):
    if not path.is_file():
        return dict(path=str(path), available=False, evidence=[])
    lines = path.read_text().splitlines()
    first_line = 1
    if scope is not None:
        function = next(node for node in ast.walk(ast.parse('\n'.join(lines)))
                        if isinstance(node, ast.FunctionDef) and node.name == scope)
        first_line = function.lineno
        lines = lines[function.lineno-1:function.end_lineno]
    evidence = []
    for token in tokens:
        found = next(((index, line) for index, line in enumerate(lines, first_line)
                      if token in line), None)
        evidence.append(dict(token=token, line=None if found is None else found[0],
                             text=None if found is None else found[1].strip()))
    return dict(path=str(path), available=True, evidence=evidence)


def _event_clock_index(runtime):
    """v7 imports v5, whose named event prefix is inherited unchanged from v4."""
    path = runtime/'bus_h2o/mdp_contract_v4.py'
    tree = ast.parse(path.read_text())
    names = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name)
                         and target.id == 'RAW_EVENT_STATE_NAMES'
                         for target in node.targets))
    return names.index('sim_time_seconds'), str(path)


def _ride_source(path):
    if not path.is_file():
        return dict(path=str(path), available=False, rides=None)
    counts = Counter()
    for _, element in ET.iterparse(path, events=('end',)):
        if element.tag == 'ride':
            counts['rides'] += 1
            destination = bool(element.get('busStop', '').strip())
            lines = element.get('lines', '').split()
            explicit_lines = bool(lines) and all(line not in ('ANY', '*') for line in lines)
            counts['rides_with_busStop'] += destination
            counts['rides_with_explicit_lines'] += explicit_lines
            counts['rides_with_both'] += destination and explicit_lines
        element.clear()
    total = counts['rides']
    return dict(path=str(path), available=True,
                **{key: counts[key] for key in ('rides', 'rides_with_busStop',
                   'rides_with_explicit_lines', 'rides_with_both')},
                every_ride_explicit=bool(total) and counts['rides_with_both'] == total,
                interpretation='Planned route attributes only; no observed current stage or issue snapshot.')


def _api_sources():
    roots = []
    spec = importlib.util.find_spec('traci')
    if spec is not None and spec.origin:
        roots.append(Path(spec.origin).parent)
    if os.environ.get('SUMO_HOME'):
        roots.append(Path(os.environ['SUMO_HOME'])/'tools/traci')
    roots.append(Path('/usr/share/sumo/tools/traci'))
    base = next((path for path in roots if (path/'_person.py').is_file()), roots[-1])
    required = {'_person.py': ('getStage',),
                '_vehicle.py': ('getLine', 'getPersonIDList'),
                '_vehicletype.py': ('getPersonCapacity',),
                '_busstop.py': ('getPersonIDs',), '_simulation.py': ('getTime',)}
    rows = []
    for filename, methods in required.items():
        path = base/filename
        definitions = {}
        if path.is_file():
            tree = ast.parse(path.read_text())
            definitions = {node.name: node.lineno for node in ast.walk(tree)
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and node.name in methods}
        rows.append(dict(path=str(path), available=path.is_file(),
                         methods={name: definitions.get(name) for name in methods}))
    vehicle_source = base/'_vehicle.py'
    inherited_capacity = vehicle_source.is_file() and any(
        isinstance(node, ast.ClassDef) and node.name == 'VehicleDomain'
        and any(isinstance(parent, ast.Name) and parent.id == 'VTypeDomain' for parent in node.bases)
        for node in ast.walk(ast.parse(vehicle_source.read_text())))
    return dict(source_getter_definitions_available=inherited_capacity and all(
                    row['available'] and all(row['methods'].values()) for row in rows),
                vehicle_capacity_inherited_from_VTypeDomain=inherited_capacity,
                definitions=rows, native_getter_calls=0,
                interpretation='Interface source availability; current native values were not queried.')


def _prefix(path, clock_index, seed, arm):
    result = dict(seed=seed, arm=arm, path=str(path), available=path.is_file(),
                  total_trace_rows=0, window_issue_rows=0,
                  joint_context_observations=0, missing_joint_context=None)
    if not path.is_file():
        return result
    fields, deltas, coverage = set(), Counter(), Counter()
    for line in path.open():
        row = json.loads(line)
        result['total_trace_rows'] += 1
        started = row['started_at']
        if not 12000 <= started < 16200:
            continue
        result['window_issue_rows'] += 1
        fields.update(row)
        for name in ('event', 's', 'raw_s', 'a', 'proposed_a', 'issue_ledger'):
            coverage[name] += name in row
        # No service timestamp join: a same-clock service can be post-command.
        if row.get('event') is not None:
            deltas[float(started)-float(row['event'][clock_index])] += 1
        for name in ('capacity_context', 'joint_context', 'current_stage',
                     'next_stop_queue', 'context_z', 'context_mask'):
            coverage[name] += name in row
    total = result['window_issue_rows']
    result.update(recorded_fields=sorted(fields), field_coverage=dict(coverage),
        event_clock_rows=sum(deltas.values()),
        started_minus_event_clock=dict(count=sum(deltas.values()),
            min=min(deltas) if deltas else None, max=max(deltas) if deltas else None,
            mean=sum(value*count for value, count in deltas.items())/sum(deltas.values())
                if deltas else None,
            values=[dict(seconds=value, count=count) for value, count in sorted(deltas.items())]),
        missing_joint_context=total,
        joint_context_evidence='NativePrefixTrace stores no explicitly ordered pre-action '
            'joint stage0/next-stop inventory observation. No archived state was backfilled.')
    return result


def _v76_prefix(path, seed, arm):
    result = dict(seed=seed, arm=arm, path=str(path), available=path.is_file())
    if path.is_file():
        row = json.loads(path.read_text())
        result.update(status=row.get('status'), prefix_clock=row.get('prefix_clock'),
            trace_rows=row.get('trace_rows'), matched_prefix_rows=row.get('matched_prefix_rows'),
            integrity_all_pass=row.get('integrity', {}).get('all_pass'),
            checks={name: row.get('integrity', {}).get('checks', {}).get(name) for name in
                    ('full_prefix_exact', 'cut_input_exact', 'prefix_receipts_exact',
                     'prefix_update_values_exact')})
    return result


def audit_sources(experiments_root):
    """Read four original prefixes, small v76 receipts, static XML and source only."""
    root = Path(experiments_root)
    runtime = root/RUNTIME
    index, index_source = _event_clock_index(runtime)
    prefixes, v76 = [], []
    for seed in (6208, 6207):
        for arm in ('fresh', 'no_fresh'):
            prefixes.append(_prefix(root/V75/f'runs/executed_cql/validation/seed{seed}'
                /'stable'/arm/'prefix_trace.jsonl', index, seed, arm))
            v76.append(_v76_prefix(root/V76/f'runs/seed{seed}'/arm/'prefix_summary.json', seed, arm))
    sources = dict(
        bridge=_source_lines(runtime/'bus_h2o/sumo_env/rl_bridge.py',
            ('self._collect_new_events(simulation_current_time)', 'traci.simulationStep()',
             'self.passenger_time_ledger.advance_to(self.current_time)',
             'def fetch_events(', 'self._advance_until_event()')),
        environment=_source_lines(runtime/'bus_h2o/envs/sumo_gym_env.py',
            ('def step_to_event(', 'self._bridge.apply_action(ev,',
             'self._bridge.fetch_events()'), scope='step_to_event'),
        stream=_source_lines(root/'mcwm_closed_loop_v1/stream.py',
            ('snapshot = self._snapshot(controls[0]["agent"])',
             'proposed = np.asarray(policy(state)', 'started_at=float(snapshot.time_seconds)',
             'self.on_issue(issued)', 'self.env.step_to_event(commands)')),
        service=_source_lines(root/'mcwm_closed_loop_tail_policy_exchange_v71_20261026/service_trace.py',
            ('result = collect(native_time)',
             "self._history(bus_id, bus, 'service_ready', {station: history}, native_time)",
             'observed_at=float(observed_at)',
             "queued_passenger_ids=None if queue is None else sorted(set(queue))")),
        archived_prefix=_source_lines(root/'mcwm_closed_loop_response_residual_v11_20261021/prefix_trace.py',
            ("'s', 'raw_s', 'event', 'a', 'proposed_a'",)))
    bridge_lines = {row['token']: row['line'] for row in sources['bridge']['evidence']}
    collect = bridge_lines.get('self._collect_new_events(simulation_current_time)')
    advance = bridge_lines.get('traci.simulationStep()')
    api = _api_sources()
    route_dir = root/'sumo_eval/assets/SUMO_ruiguang/e_passenger_rou'
    routes = [_ride_source(route_dir/filename) for filename in PASSENGER_FILES]
    total = sum(row['window_issue_rows'] for row in prefixes)
    available = all(row['available'] for row in prefixes)
    return dict(status='complete' if available else 'incomplete_archived_prefixes',
        source_api_visibility_possible=api['source_getter_definitions_available'],
        source_api_current_values_possible=api['source_getter_definitions_available'],
        native_joint_coverage_verified=False, historical_joint_available=False,
        joint_context_observations=0, window_issue_rows=total, missing_joint_context=total,
        window=dict(lo=12000, hi=16200, interval='[lo,hi)', clock='issue.started_at'),
        event_clock_field=dict(array='event', named_field='sim_time_seconds',
                               index=index, named_schema_source=index_source),
        archived_prefixes=prefixes, v76_restoration_receipts=v76,
        static_passenger_routes=routes, api_source=api, source_evidence=sources,
        issue_provenance=dict(collect_precedes_simulation_step=
            collect is not None and advance is not None and collect < advance,
            same_timestamp_proves_preaction=False,
            interpretation='collect(t) and its service observation precede simulationStep to t+1. '
                'The returned policy issue uses ledger t+1. A service observed at that issue '
                'timestamp may come from the next collect after applying the issue command. '
                'Event time and issue time are retained separately; no joint values are inferred.'),
        permitted_next_step='Only consider a separately authorized pre-action native visibility '
            'sample before learning; this audit installs no hook and advances no simulator.',
        new_simulated_seconds=0, new_training_updates=0, new_feedback_slots=0,
        final_state_reads=0, service_timestamp_joins=0, raw_states_exported=0,
        old_code_and_results_modified=False)
