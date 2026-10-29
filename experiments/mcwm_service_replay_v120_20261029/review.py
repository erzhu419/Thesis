"""Adjudicate actual route-end write ordering, then finish passive data readout."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from route_end_order import compare_route_end_blocks
from samples import select_and_label
from analyze_data import analyze_data


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def review(root):
    protocol = json.loads((HERE/'protocol.json').read_text())
    output = HERE/'review'
    output.mkdir(exist_ok=False)
    cases = {}
    for scene in protocol['scenarios']:
        run = HERE/'runs'/scene
        reference = root/protocol['reference_root']/scene/'no_fresh'
        raw = json.loads((run/'summary.json').read_text())
        if raw['status'] != 'integrity_failed':
            raise ValueError('This adjudication is specific to the retained strict-order failure')
        other_checks = {k:v for k,v in raw['integrity']['checks'].items() if k != 'complete_native_logs'}
        files = raw['reproduction']['files']
        preserved_exact = (all(other_checks.values()) and raw['reproduction']['observer']['all_pass']
            and all(v['all_pass'] for k,v in files.items() if k != 'visit_events.jsonl'))
        order = compare_route_end_blocks(run/'visit_events.jsonl', reference/'visit_events.jsonl')
        folder = output/scene
        folder.mkdir()
        checks = dict(other_native_logs_and_accounting_exact=preserved_exact,
                      route_end_blocks_equal=order['all_pass'])
        value = dict(status='review_hold', integrity=dict(all_pass=all(checks.values()), checks=checks),
            raw_status=raw['status'], raw_summary=str(run/'summary.json'),
            raw_strict_visit_rows_equal=False, visit_order_adjudication=order,
            final_ledger=raw['final_ledger'], observation=raw['observation'],
            budget=dict(original_native_seconds=raw['budget']['native_seconds'],
                exact_native_seconds=(24000 if raw['budget']['full_day_completed'] and
                    raw['final_ledger']['time_seconds']==24000 else None),
                additional_native_seconds=0, optimizer_updates=0, feedback_admitted=0),
            interpretation='Original strict failure and source logs remain unchanged. '
                'Only simultaneous consecutive route_end order is treated as unspecified. '
                'No other native record, timestamp or accounting field is relaxed.')
        if value['integrity']['all_pass']:
            # Existing full logs stay in place; only new selection outputs go here.
            for name in ('contexts.jsonl', 'commands.jsonl', 'visit_events.jsonl',
                         'exchanges.jsonl', 'observer_summary.json'):
                (folder/name).symlink_to(run/name)
            selected = select_and_label(folder, protocol['feedback_contract'],
                                        horizon=24000, window_seconds=60)
            value.update(status='complete' if selected['all_pass'] else 'service_data_hold',
                         service_records=selected)
        write(folder/'summary.json', value)
        cases[scene] = value
    all_pass = all(c['status'] == 'complete' for c in cases.values())
    profile = analyze_data(output) if all_pass else None
    result = dict(status='complete' if all_pass else 'hold', cases=cases,
        new_native_seconds=0, original_native_seconds=sum(c['budget']['original_native_seconds'] for c in cases.values()),
        profile=profile, old_raw_results_unchanged=True, no_training=True)
    write(output/'review_summary.json', result)
    payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(payload.encode()) > 40000:
        raise ValueError('Keep full review resident; compact receipt exceeded40KB')
    print(payload)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thesis-root', type=Path, required=True)
    result = review(parser.parse_args().thesis_root)
    raise SystemExit(0 if result['status']=='complete' else 1)
