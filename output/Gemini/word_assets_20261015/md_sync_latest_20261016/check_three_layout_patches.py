"""Append content checks for the three approved native equation line breaks only."""
from pathlib import Path
import importlib.util
import json
import sys

T = Path(__file__).resolve().parent
helper = T.parent / 'exact_library_v2_20261016/prepare_math_verification.py'
spec = importlib.util.spec_from_file_location('settled_math_canonical', helper)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
native = Path(sys.argv[1])
selected = {'B-25', 'B-39', 'E-46'}
expected = {
    row['label']: row
    for row in json.loads((T / 'math_latest_display_expected.json').read_text())
    if row['label'] in selected
}
actual = {row['label']: row for row in h.word_rows(native) if row['label'] in selected}
assert set(expected) == set(actual) == selected
checks = []
for label in ('B-25', 'B-39', 'E-46'):
    wanted, found = expected[label], actual[label]
    content = json.loads(json.dumps(h.content(found['maths'], wanted['tex']), ensure_ascii=False))
    check = {
        'label': label,
        'ordinal': found['ordinal'],
        'native_oMath_count': len(found['maths']),
        'content_matches_existing_md_donor': content == wanted['expected_content'],
    }
    if not check['content_matches_existing_md_donor']:
        check.update(expected_content=wanted['expected_content'], actual_content=content)
    checks.append(check)
report_path = T / 'qa_math_latest_native.json'
report = json.loads(report_path.read_text())
report['layout_patch_native'] = str(native)
report['layout_patch_checks'] = checks
report['layout_patch_correct'] = all(row['content_matches_existing_md_donor'] for row in checks)
report['correct'] = report['correct'] and report['layout_patch_correct']
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'correct': report['correct'], 'layout_patch_checks': checks}, ensure_ascii=False))
