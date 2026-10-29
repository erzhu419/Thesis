"""Temporary display-equation layout supply; authoritative MD is not written."""
from pathlib import Path
from zipfile import ZipFile
import importlib.util
import json
import re

from lxml import etree as E
import pypandoc

T = Path(__file__).resolve().parent
helper = T.parent / 'exact_library_v2_20261016/prepare_math_verification.py'
spec = importlib.util.spec_from_file_location('settled_math_canonical', helper)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
selected = {'B-25', 'B-39', 'E-46'}
entries = [row for row in json.loads((T / 'math_latest_display_expected.json').read_text())
           if row['label'] in selected]
assert len(entries) == 3
batch = []
for row in entries:
    tex = re.sub(r'\\tag\{[^}]+\}', '', row['tex']).strip()
    if row['label'] == 'B-25':
        before = r' + \sum_{h_{\mathrm{obs}}}'
        assert tex.count(before) == 1
        tex = tex.replace(' = ', ' &= ', 1).replace(before, '\\\\\n &\\quad + \\sum_{h_{\\mathrm{obs}}}', 1)
        tex = '\\begin{aligned}\n' + tex + '\n\\end{aligned}'
    elif row['label'] == 'B-39':
        before = r' - Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{O}'
        assert tex.count(before) == 1
        tex = tex.replace(before, '\\\\\n &\\quad' + before, 1)
    else:
        before = r" + \gamma_{\Delta}\sum_{x'}"
        assert tex.count(before) == 1
        tex = tex.replace(before, '\\\\\n &\\quad' + before, 1)
    row['layout_tex'] = tex
    batch.extend([f'EXPECT_SLOT_{row["ordinal"]:04d}', '$$\n' + tex + '\n$$'])
markdown = '\n\n'.join(batch) + '\n'
(T / 'math_three_layout_donor.md').write_text(markdown)
donor = T / 'math_three_layout_donor.docx'
pypandoc.convert_text(markdown, 'docx', format='markdown+tex_math_dollars', outputfile=str(donor))
with ZipFile(donor) as z:
    root = E.fromstring(z.read('word/document.xml'))
by = {}
slot = None
for block in root.find(h.W + 'body'):
    marker = re.fullmatch(r'EXPECT_SLOT_(\d+)', h.text(block))
    if marker:
        slot = int(marker[1])
        by[slot] = []
    elif slot is not None:
        by[slot].extend(block.iter(h.M + 'oMath'))
checks = []
for row in entries:
    content = json.loads(json.dumps(h.content(by[row['ordinal']], row['layout_tex']), ensure_ascii=False))
    checks.append({
        'label': row['label'], 'ordinal': row['ordinal'],
        'layout_tex': row['layout_tex'],
        'source_tex': row['tex'],
        'content_matches_md_donor': content == row['expected_content'],
        'oMath_count': len(by[row['ordinal']]),
    })
report = {'donor': str(donor), 'correct': all(row['content_matches_md_donor'] for row in checks), 'checks': checks}
(T / 'math_three_layout_donor_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False))
assert report['correct']
