"""Record only atlas sheets actually viewed by this reviewer."""
from pathlib import Path
import json
import sys

A = Path(__file__).resolve().parent
dest = A / 'visual_review_3.json'
index = json.loads((A / 'native_atlas/atlas_index.json').read_text())
r = json.loads(dest.read_text()) if dest.exists() else {
    'scope_first_math_index': 3201, 'scope_last_math_index': 4825,
    'expected_math_objects': 1625, 'reviewed_sheets': [], 'issues': [], 'capture_gaps': [],
    'method': 'Latest native Word original-position PrintWindow atlas, every target tile viewed; suspicious target content checked against original viewport.',
}
by_file = {s['file']: s for s in index['sheets']}
seen = {s['file'] for s in r['reviewed_sheets']}
for filename in sys.argv[1:]:
    s = by_file[filename]
    assert all(3201 <= i <= 4825 for i in s['math_indices'])
    if filename not in seen:
        r['reviewed_sheets'].append({'file': filename, 'math_indices': s['math_indices'], 'visual_inspection_complete': True})
        seen.add(filename)
placements = {}
for p in index['placements']:
    placements.setdefault(p['math_index'], []).append(p)
reviewed = [i for i in range(3201, 4826) if i in placements and all(p['sheet'] in seen for p in placements[i])]
r['reviewed_math_indices'] = reviewed
r['reviewed_math_objects'] = len(reviewed)
r['pending_math_indices'] = sorted(set(range(3201, 4826)) - set(reviewed))
r['all_1625_visually_inspected'] = len(reviewed) == 1625
r['atlas_index'] = str((A / 'native_atlas/atlas_index.json').resolve())
dest.write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
print(f'Reviewed {len(reviewed)}/1625 actual formulas on {len(seen)} sheets')
