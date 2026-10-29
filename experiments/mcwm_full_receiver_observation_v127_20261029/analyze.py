"""Server-side readout of the frozen selection and passive receiver captures."""
from collections import Counter
import json
from pathlib import Path

from experiments.mcwm_service_risk_contract_v126_20261029.interface import (
    overrun_target, risk_inputs,
)


def _load(record, capture_dir):
    sequence = record['selection']['sequence']
    saved = json.loads((Path(capture_dir) / (str(sequence) + '.json')).read_text())
    observation = saved['observation']
    inputs = risk_inputs(record, observation)
    target = overrun_target(record)
    capture = observation['capture']
    passive = (bool(saved['passivity'])
        and all(value is True for value in saved['passivity'].values())
        and all(value is True for value in capture['checks'].values())
        and capture['native_clock_before'] == capture['native_clock_after']
            == observation['meta']['observed_at'])
    derived = dict(meta=observation['meta'], selection=record['selection'],
        inputs=inputs, target=target, receiver=observation['receiver'],
        capture_status=capture['status'], missing=capture['missing'],
        errors=capture['errors'], passivity_passed=passive,
        reference_field_checks=saved['reference_field_checks'])
    return derived, observation


def _field_name(name):
    return 'person/current_stage0' if name.endswith('/current_stage0') else name


def _counts(rows):
    packets = Counter(row['selection']['packet_deadline'] for row in rows)
    classes = Counter('unknown' if row['target']['value'] is None
                      else str(row['target']['value']) for row in rows)
    n = Counter(str(row['inputs'][0]) for row in rows if row['inputs'] is not None)
    cross = Counter(('unknown' if row['inputs'] is None else 'n=0' if row['inputs'][0] == 0 else 'n>0',
                     'unknown' if row['target']['value'] is None else str(row['target']['value']))
                    for row in rows)
    fields = ('onboard_count', 'current_leg_dest_count',
              'estimated_service_ready_time', 'estimated_service_ready_age_seconds')
    complete = sum(row['capture_status'] == 'complete' for row in rows)
    reference = {field: dict(Counter('unknown' if row['reference_field_checks'][field] is None
        else str(row['reference_field_checks'][field]).lower() for row in rows))
        for field in ('original_onboard_count', 'original_ready_estimate')}
    return dict(slots=len(rows), packets=len(packets),
        packet_size_histogram=dict(sorted(Counter(map(str, packets.values())).items())),
        classes={value: classes[value] for value in ('0', '1', 'unknown')},
        fields=dict(complete=complete, missing=len(rows) - complete,
            known={field: sum(row['receiver'][field] is not None for row in rows) for field in fields},
            current_stage0_complete=sum(row['receiver']['current_stage0_complete'] is True for row in rows),
            missing_fields=dict(Counter(_field_name(field) for row in rows for field in row['missing'])),
            read_errors=dict(Counter(_field_name(error['field']) for row in rows for error in row['errors']))),
        inputs_complete=sum(row['inputs'] is not None for row in rows),
        n=dict(histogram=dict(sorted(n.items(), key=lambda item: int(item[0]))),
            unknown=sum(row['inputs'] is None for row in rows),
            unit='selected root snapshots, not unique passengers'),
        n_target_classes={group: {value: cross[(group, value)] for value in ('0', '1', 'unknown')}
                          for group in ('n=0', 'n>0', 'unknown')},
        departure_censored=sum(row['target']['departure_time_censored'] is True for row in rows),
        censored_binary_identifiable=sum(row['target']['departure_time_censored'] is True
            and row['target']['value'] is not None for row in rows),
        passivity_passed=sum(row['passivity_passed'] for row in rows),
        reference_field_checks=reference)


def summarize_scene(records, capture_dir):
    """Write resident derived rows; return only compact coverage/class counts."""
    capture_dir = Path(capture_dir)
    rows = [_load(record, capture_dir)[0] for record in records]
    prefix = [row for row in rows if row['selection']['packet_deadline'] <= 9000]
    suffix = [row for row in rows if row['selection']['packet_deadline'] > 9000]
    total, before, after = map(_counts, (rows, prefix, suffix))
    derived_path = capture_dir.parent / 'derived_records.jsonl'
    with derived_path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    complete = (total['slots'] == 285 and total['packets'] == 57
        and total['packet_size_histogram'] == {'5': 57}
        and before['slots'] == 135 and after['slots'] == 150)
    return dict(collection_complete=complete,
        field_coverage_complete=complete and total['fields']['missing'] == 0,
        inputs_complete=complete and total['inputs_complete'] == 285,
        labels_complete=complete and total['classes']['unknown'] == 0,
        prefix_has_both_classes=before['classes']['0'] > 0 and before['classes']['1'] > 0,
        passivity_all_pass=complete and total['passivity_passed'] == 285,
        total=total, prefix=before, suffix=after, derived_records=str(derived_path),
        split='original packet_deadline <= 9000 versus > 9000',
        target_scope='Original 60-second evidence only; original records are unchanged.',
        fits=0, performance_evidence=False)


def _prefix(records, capture_dir):
    result = {}
    count = 0
    for record in records:
        if record['selection']['packet_deadline'] > 9000:
            continue
        count += 1
        derived, observation = _load(record, capture_dir)
        meta = {key: value for key, value in observation['meta'].items() if key != 'source_group'}
        # New dictionaries preserve the original observation and selected record.
        result[record['selection']['sequence']] = dict(
            meta={key: value for key, value in record['context']['meta'].items() if key != 'source_group'},
            selection=record['selection'], action=record['action'], labels=record['labels'],
            observation={**observation, 'meta': meta}, inputs=derived['inputs'], target=derived['target'])
    return count, result


def compare_prefixes(stable_records, shock_records, stable_capture_dir, shock_capture_dir):
    """Compare all common initialization facts, omitting only meta.source_group."""
    stable_count, stable = _prefix(stable_records, stable_capture_dir)
    shock_count, shock = _prefix(shock_records, shock_capture_dir)
    shared = sorted(stable.keys() & shock.keys())
    differences = [(sequence, [field for field in stable[sequence]
                              if stable[sequence][field] != shock[sequence][field]])
                   for sequence in shared if stable[sequence] != shock[sequence]]
    missing_stable, missing_shock = sorted(shock.keys() - stable.keys()), sorted(stable.keys() - shock.keys())
    return dict(all_pass=(stable_count == shock_count == len(stable) == len(shock) == 135
                         and not missing_stable and not missing_shock and not differences),
        slots=dict(stable=stable_count, local_speed_drop=shock_count),
        shared_sequences=len(shared), exact_roots=len(shared) - len(differences),
        mismatched_roots=len(differences),
        mismatches=[dict(sequence=sequence, fields=fields) for sequence, fields in differences[:3]],
        missing_stable=dict(count=len(missing_stable), examples=missing_stable[:3]),
        missing_shock=dict(count=len(missing_shock), examples=missing_shock[:3]),
        compared=['meta_except_source_group', 'selection', 'action', 'labels',
                  'entire_observation_except_meta_source_group', 'inputs', 'target'])
