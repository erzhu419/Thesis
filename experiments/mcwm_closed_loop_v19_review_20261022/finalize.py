"""Compare the six authorized suffixes, retaining the original clock failure."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
OLD = HERE.parent/'mcwm_closed_loop_execution_phase_v19_20261022'
spec = importlib.util.spec_from_file_location('v19_comparison', OLD/'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def compare(source, root, original):
    audit = json.loads((root/'first_arm_clock_audit.json').read_text())
    if audit.get('reusable') is not True:
        raise RuntimeError('The retained first arm has not passed the native-event clock audit')
    first = original/'branches_retest/7X/low'
    original_loader = analysis.load_branch

    def audited_load(path):
        value = original_loader(path)
        if Path(path).resolve() != first.resolve():
            return value
        failed = [key for key, ok in value['integrity']['checks'].items() if not ok]
        if value['status'] != 'integrity_failed' or failed != ['actor_real_clock']:
            raise RuntimeError('The retained first arm differs from its audited failure')
        # This is an explicit in-memory interpretation of the separately audited
        # false predicate. The original result remains integrity_failed on disk.
        value = deepcopy(value)
        value['original_status'] = value['status']
        value['status'] = 'complete'
        value['integrity']['checks']['actor_real_clock'] = True
        value['integrity']['all_pass'] = True
        value['phase_intervention']['actor_real_clock'] = True
        return value

    analysis.load_branch = audited_load
    try:
        value = analysis.analyze(source, root)
    finally:
        analysis.load_branch = original_loader
    value['first_arm_clock_audit'] = dict(
        original_status='integrity_failed', original_failed_check='actor_real_clock',
        audit_file='first_arm_clock_audit.json', reused_without_rerun=True,
        prefix_verification='resident elementwise trace and native event clock',
        suffix_verification='unchanged source path plus native 300 steps and zero-training counters',
        suffix_per_issue_clock_records=False)
    value['new_suffix_clock_verification'] = 'Native event clock checked online in each of the remaining five suffixes; actor inputs are unchanged.'
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    while True:
        value = compare(args.source, args.root, args.original)
        if value['branch_statuses'].get('failed') or value['branch_statuses'].get('integrity_failed'):
            raise RuntimeError('A remaining branch failed; preserve it for review')
        if value['status'] == 'complete' or not args.wait:
            break
        time.sleep(20)
    data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(data.encode()) > 30000:
        raise RuntimeError('Reduce the resident summary before collection')
    args.out.write_text(data)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'], bytes=len(data.encode()))))


if __name__ == '__main__':
    main()
