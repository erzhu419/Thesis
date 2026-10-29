"""Read only frozen selection metadata, retaining the full root manifest remotely."""
import argparse
import json
from pathlib import Path


def selection_budget(root, out):
    source = root / 'experiments/mcwm_service_opportunity_v121_20261029/runs'
    cases = {}
    manifest = {}
    for scene in ('stable', 'local_speed_drop'):
        roots = []
        with (source / scene / 'selected_records.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                roots.append(dict(meta=row['context']['meta'], selection=row['selection']))
        roots.sort(key=lambda r: (r['selection']['window_start'], r['selection']['sequence']))
        deadlines = sorted({r['selection']['packet_deadline'] for r in roots})
        counts = {d: sum(r['selection']['packet_deadline'] == d for r in roots) for d in deadlines}
        prefix = [r for r in roots if r['selection']['packet_deadline'] <= 9000]
        if len(roots) != 285 or len(deadlines) != 57 or set(counts.values()) != {5} or len(prefix) != 135:
            raise ValueError('Selection differs from the frozen v121 budget: ' + scene)
        if any(r['meta']['observed_at'] != r['selection']['window_start'] or
               r['meta']['sequence'] != r['selection']['sequence'] for r in roots):
            raise ValueError('Root identity and selection clock disagree: ' + scene)
        final = roots[-1]
        cases[scene] = dict(roots=len(roots), packets=len(deadlines), common_prefix_slots=len(prefix),
            suffix_slots=len(roots)-len(prefix), first_packet=deadlines[0], last_packet=deadlines[-1],
            first_root_clock=roots[0]['meta']['observed_at'],
            final_root=final['meta'], native_seconds_from_zero=final['meta']['observed_at'])
        manifest[scene] = roots
    a = [r for r in manifest['stable'] if r['selection']['packet_deadline'] <= 9000]
    b = [r for r in manifest['local_speed_drop'] if r['selection']['packet_deadline'] <= 9000]
    # Scenario identity necessarily differs; every shared-prefix selection and other meta field must match.
    common_equal = all(x['selection'] == y['selection'] and
        {k:v for k,v in x['meta'].items() if k != 'source_group'} ==
        {k:v for k,v in y['meta'].items() if k != 'source_group'} for x,y in zip(a,b))
    if not common_equal:
        raise ValueError('Shared initialization roots disagree')
    result = dict(status='METADATA_BUDGET_CONFIRMED', cases=cases,
        shared_prefix_selection_equal=common_equal,
        future_native_seconds=sum(c['native_seconds_from_zero'] for c in cases.values()),
        current_native_seconds=0, current_collection=0, current_fits=0,
        manifest='selected_roots_metadata.json', labels_read_for_analysis=False,
        note='Later passive replay stops before applying the last selected root; no future labels are recollected.')
    out.mkdir(parents=True, exist_ok=True)
    (out/'selected_roots_metadata.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'selection_budget.json').write_text(json.dumps(result, indent=2)+'\n')
    raw = json.dumps(result, separators=(',', ':'))
    if len(raw.encode()) > 6000:
        raise ValueError('Metadata receipt exceeds the fixed small-pull bound')
    print(raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thesis-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection_budget(args.thesis_root, args.out)
