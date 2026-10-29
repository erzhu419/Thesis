"""Keep native coverage separate from a decision about future trace targets."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLUMNS = ['updates', 'line', 'arm', 'first_reward_links', 'full44_overlap', 'full44_L_geom',
    'full44_L_actual', 'first_reward_overlap', 'first_reward_L_geom', 'first_reward_L_actual',
    'behavior_p_mean', 'target_p_mean', 'latent_mean_shift', 'log_std_shift',
    'full44_atom_overlap', 'full44_continuous_overlap', 'full44_overlap_min']
BRANCH_COLUMNS = ['line', 'arm', 'requested_hold', 'actual_hold', 'native_seconds', 'zero_duration_rows',
    'integrity_pass', 'failed_checks', 'departure_status', 'departure_clock', 'arrival_status', 'arrival_clock',
    'action_step_length', 'at_bus_stop', 'actual_parallel_seconds', 'parent_clock_unchanged',
    'parent_route_offsets_unchanged']


def read(path):
    return json.loads(path.read_text())


def compact(value):
    if isinstance(value, float):
        return float(f'{value:.8g}')
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    return value


def before9000(path):
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['started_at'] >= 9000.:
                return
            yield row


def compare_prefixes(runs):
    from itertools import zip_longest
    paths = [runs/case/'prefix_trace.jsonl' for case in ('stable', 'local_speed_drop')]
    count = 0
    if not all(path.exists() for path in paths):
        return dict(status='missing', exact=None, rows=None)
    for a, b in zip_longest(*(before9000(path) for path in paths)):
        if a != b:
            return dict(status='different', exact=False, rows_compared=count,
                first_sequence=[None if row is None else row['sequence'] for row in (a, b)])
        count += 1
    return dict(status='exact' if count else 'missing', exact=True if count else None, rows=count)


def summarize_case(root):
    completed = root/'pilot_summary.json'
    if not completed.exists():
        failed = root/'pilot_failure.json'
        return dict(status='failed', failure=read(failed) if failed.exists() else None)
    source = read(completed)
    native, learning = source['native'], source['learning']
    rows = []
    for packet in learning['credit_curve']:
        for chain in packet['chains']:
            full, first = chain['prefixes']['full44'], chain['prefixes'].get('first_reward', {})
            drift = chain['drift']
            rows.append([packet['updates'], chain['line_id'], chain['arm'], chain['first_reward_event'],
                full['overlap_mean'], full['L_geom'], full['L_old'], first.get('overlap_mean'),
                first.get('L_geom'), first.get('L_old'), drift['behavior_p_hold_mean'],
                drift['target_p_hold_mean'], drift['latent_mean_shift'], drift['log_std_shift'],
                full['atom_overlap_mean'], full['continuous_overlap_mean'], full['overlap_min']])
    branches = []
    for branch in native['response']['branches']:
        detail = read(Path(branch['summary_path']))
        overlap = read(Path(branch['parallel_receipt']))
        branches.append(dict(line=branch['line_id'], arm=branch['arm'],
            requested_hold=detail['execution']['requested_hold_seconds'],
            actual_hold=detail['execution']['native_applied_hold_seconds'],
            native_seconds=branch['native_seconds'], zero_duration_rows=branch['zero_duration_rows'],
            integrity_pass=detail['integrity']['all_pass'],
            failed_checks=[key for key, value in detail['integrity']['checks'].items() if not value],
            physical=dict(departure=[detail['physical']['departure'][key] for key in ('status', 'clock_seconds')],
                next_arrival=[detail['physical']['next_same_vehicle_arrival'][key] for key in ('status', 'clock_seconds')],
                action_step_length=detail['physical']['apply_before']['action_step_length_seconds'],
                at_bus_stop=detail['physical']['apply_before']['is_at_bus_stop']),
            actual_parallel_seconds=overlap['overlap_seconds'],
            parent_clock_unchanged=overlap['native_before'] == overlap['native_after'],
            parent_route_offsets_unchanged=overlap['route_streams']['parent_offsets_unchanged']))
    return dict(status=source['status'], integrity=source['integrity'],
        native_checks=native['integrity'], learning_checks=learning['integrity'],
        budget=native['budget'], behavior_saved_at=native['behavior_saved_at'],
        initializer=native['initializer'], native_external_seed=source['native_external_seed'],
        admission=learning['admission'], branches=[[
            branch['line'], branch['arm'], branch['requested_hold'], branch['actual_hold'],
            branch['native_seconds'], branch['zero_duration_rows'], branch['integrity_pass'], branch['failed_checks'],
            *branch['physical']['departure'], *branch['physical']['next_arrival'],
            branch['physical']['action_step_length'], branch['physical']['at_bus_stop'],
            branch['actual_parallel_seconds'], branch['parent_clock_unchanged'],
            branch['parent_route_offsets_unchanged']] for branch in branches], credit_rows=rows,
        initial_preflight=read(root/'preflight_gate.json'),
        wall_seconds=source['wall']['finished']-source['wall']['started'])


def main():
    runs, out = HERE/'runs', HERE/'analysis'
    cases = {case: summarize_case(runs/case) for case in ('stable', 'local_speed_drop')}
    prefix = compare_prefixes(runs)
    complete = all(case.get('integrity', {}).get('all_pass') for case in cases.values())
    value = dict(status='complete' if complete else 'partial', priority='high', cases=cases,
        common_before_shock=prefix, integrity=dict(all_pass=complete and prefix['exact'] is True),
        schema=dict(credit_columns=COLUMNS, branch_columns=BRANCH_COLUMNS, root_excluded=True, continuation_links=44,
            scores_every8_updates=True, transferred_updates=[24,32,48,80,144,272,400,456],
            mixed_path_log_weight='links*log(0.9), separate from listed L fields; native environment gamma=1',
            first_reward='first ordinal with differing actual r; event clocks can differ; missing is null',
            L_geom='sum of conditional E_mu log c on observed new states, not a trajectory expectation'),
        decision=dict(trace_training_started=False, further_collection_started=False,
            next_gate='Review physical coverage and actual post24-to-current overlap at the first reward and full44; compare lines and stable/shock before proposing a trace target. No new stage launches automatically.'),
        retained_contract=dict(slots_per_case=285, updates_per_case=456, horizon=24000,
            release_clock=9630, initialization=50000, F_unchanged=True, trust_unchanged=True),
        artifacts='All full chains, models, per-packet scores and update/RNG logs remain resident.')
    encoded = json.dumps(compact(value), separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > 24000:
        raise RuntimeError('Reduce resident summary before the24KB pull; completed pilot is not rerun')
    out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(encoded)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'])))


if __name__ == '__main__':
    main()
