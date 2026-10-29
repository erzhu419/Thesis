"""Read-only support for the proposed exact full-day action-difference target."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'experiments/mcwm_service_opportunity_v121_20261029/runs'


def main():
    cases, prefix = {}, {}
    for scene in ('stable', 'local_speed_drop'):
        records = [json.loads(line) for line in
                   (SOURCE / scene / 'selected_records.jsonl').read_text().splitlines()]
        groups = {}
        for row in records:
            m, a = row['context']['meta'], row['action']
            if row['errors'] or a is None:
                raise ValueError('The retained factual record is incomplete')
            key = (m['sequence'], m['observed_at'], m['receiver'], m['station'])
            groups.setdefault(key, set()).add((a['applied_at'], a['hold_seconds']))
        prefix[scene] = {
            (r['context']['meta']['sequence'], r['context']['meta']['observed_at'],
             r['context']['meta']['receiver'], r['context']['meta']['station']): r['action']
            for r in records if r['selection']['packet_deadline'] <= 9000}
        cases[scene] = dict(records=len(records), distinct_recorded_roots=len(groups),
            roots_with_multiple_recorded_executed_actions=sum(len(v) > 1 for v in groups.values()),
            first_issue=min(r['selection']['window_start'] for r in records),
            last_issue=max(r['selection']['window_start'] for r in records))
    if prefix['stable'].keys() != prefix['local_speed_drop'].keys():
        raise ValueError('Cannot reuse the declared common-prefix grouping')
    same_action = sum(a == prefix['local_speed_drop'][k] for k, a in prefix['stable'].items())
    result = dict(status='complete', decision='EXACT_SAME_DAY_PAIRED_TARGET_NOT_SUPPORTED',
        cases=cases,
        repeated_prefix=dict(roots=len(prefix['stable']), same_executed_action=same_action,
            different_executed_action=len(prefix['stable'])-same_action,
            scope='Common issue identities, not certification of identical full future dynamics. Different later scenarios are not action interventions.'),
        recorded_root_context_groups=sum(c['distinct_recorded_roots'] for c in cases.values())-len(prefix['stable']),
        exact_action_difference_labels_in_this_panel=0,
        timing=dict(full_day_endpoint=24000, feedback_delay=30,
            hypothetical_complete_label_available_at=24030, last_feedback_deadline=18000,
            first_feedback_deadline=1200, interval=300, packets=57, max_per_packet=5,
            admissible_packets_for_complete_full_day_target=0,
            meaning='Timing is conditional on both potential outcomes existing; this passive panel has no second-action outcome.'),
        inference='This rules out literal exact full-day paired supervision under the same-day contract, not bootstrapped or other approximate online learning.',
        new_native_seconds=0, new_collection=0, fitted_models=0, control_changes=0)
    if any(c['roots_with_multiple_recorded_executed_actions'] for c in cases.values()) or same_action != len(prefix['stable']):
        raise ValueError('Unexpected action variation needs separate same-state/continuation review')
    (HERE / 'resident_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    raw = json.dumps(result, separators=(',', ':'))
    if len(raw.encode()) > 8000:
        raise ValueError('Keep this support summary under8KB')
    print(raw)


if __name__ == '__main__':
    main()
