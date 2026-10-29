"""Record only sheets already inspected visually by this reviewer."""
from pathlib import Path
import json
import sys

A = Path(__file__).resolve().parent
dest = A / 'visual_review_003201_004738.json'
index = json.loads((A / 'native_atlas/atlas_index.json').read_text())
report = json.loads(dest.read_text()) if dest.exists() else {
    'scope_first_math_index': 3201, 'scope_last_math_index': 4738,
    'expected_math_objects': 1538, 'method': 'Actual original-position Word UI images, inspected at native scale; surrounding context is not treated as the target formula.',
    'reviewed_sheets': [], 'issues': [], 'capture_gaps': [],
}
available = {s['file']: s for s in index['sheets']}
seen = {s['file'] for s in report['reviewed_sheets']}
for filename in sys.argv[1:]:
    sheet = available[filename]
    assert all(3201 <= i <= 4738 for i in sheet['math_indices'])
    if filename not in seen:
        report['reviewed_sheets'].append({'file': filename, 'math_indices': sheet['math_indices'],
                                        'visual_inspection_complete': True})
        seen.add(filename)
placements = {}
for placement in index['placements']:
    placements.setdefault(placement['math_index'], []).append(placement)
reviewed = [i for i in range(3201, 4739) if i in placements
            and all(p['sheet'] in seen for p in placements[i])]
report['reviewed_math_indices'] = reviewed
report['reviewed_math_objects'] = len(reviewed)
report['pending_math_indices'] = sorted(set(range(3201, 4739))-set(reviewed))
report['all_1538_visually_inspected'] = len(reviewed) == 1538
report['atlas_index'] = str(A / 'native_atlas/atlas_index.json')
dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(f'Reviewed {len(reviewed)}/1538 actual formulas, {len(seen)} sheets')
