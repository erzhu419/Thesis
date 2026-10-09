"""Check actual reference spans and the copied front layout against the template."""
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workpapers/模板迁移_20261002'))
from verify_format import Doc, normp, normr
sys.path.insert(0, str(ROOT / 'workpapers/全文深度校审_20261003'))
from extract_word import NS, content

W = '{' + NS['w'] + '}'
base = ROOT / 'output/排版完成.docx'
template = next(ROOT.glob('中南大学博士学位论文_gty*docx'))
target, source = Doc(base), Doc(template)
front_errors = []
source_front, target_front = list(source.body)[:62], list(target.body)[:62]

for i, (a, b) in enumerate(zip(source_front, target_front)):
    ap = [a] if a.tag == W+'p' else a.findall('.//w:p', NS)
    bp = [b] if b.tag == W+'p' else b.findall('.//w:p', NS)
    assert len(ap) == len(bp)
    for j, (pa, pb) in enumerate(zip(ap, bp)):
        _, pp_a, _, ra, _ = source.resolve(pa)
        _, pp_b, _, rb, _ = target.resolve(pb)
        if normp(pp_a) != normp(pp_b):
            front_errors.append({'block': i, 'p': j, 'paragraph': [normp(pp_a), normp(pp_b)]})
        expected = [normr(source, r) for txt, r in ra if txt.strip()]
        for txt, r in rb:
            actual = normr(target, r)
            if txt.strip() and actual not in expected:
                front_errors.append({'block': i, 'p': j, 'text': txt, 'format': actual, 'expected': expected})
    if a.tag == W+'tbl':
        old_grid = [n.get(W+'w') for n in a.findall('w:tblGrid/w:gridCol', NS)]
        new_grid = [n.get(W+'w') for n in b.findall('w:tblGrid/w:gridCol', NS)]
        if old_grid != new_grid:
            front_errors.append({'block': i, 'grid': [old_grid, new_grid]})

with ZipFile(base) as z:
    doc = E.fromstring(z.read('word/document.xml'))
active, names, ranges = {}, {}, {}
for n in doc.iter():
    if n.tag == W+'bookmarkStart':
        k = n.get(W+'id')
        active[k] = []
        names[k] = n.get(W+'name')
    elif n.tag == W+'bookmarkEnd':
        k = n.get(W+'id')
        ranges[names[k]] = ''.join(active.pop(k))
    elif n.tag == W+'t':
        for values in active.values():
            values.append(n.text or '')

fields, mismatches = [], []
for p in doc.findall('.//w:p', NS):
    pending, depth = [], 0
    for n in p:
        depth += len(n.xpath('.//w:fldChar[@w:fldCharType="begin"]', namespaces=NS))
        if depth:
            pending.append(n)
        ends = n.xpath('.//w:fldChar[@w:fldCharType="end"]', namespaces=NS)
        if ends:
            depth -= len(ends)
            if depth == 0:
                code = ''.join(''.join(v.xpath('.//w:instrText/text()', namespaces=NS)) for v in pending)
                match = re.match(r'\s*REF\s+(\S+)', code)
                if match:
                    shown = ''.join(content(v) for v in pending)
                    wanted = ranges.get(match[1])
                    row = {'target': match[1], 'shown': shown, 'bookmark_span': wanted}
                    fields.append(row)
                    if wanted is None or re.sub(r'\s+', '', wanted) != re.sub(r'\s+', '', shown):
                        mismatches.append(row)
                pending = []

result = {'front_format_errors': front_errors, 'reference_fields': len(fields),
          'reference_field_mismatches': mismatches, 'sections': len(doc.xpath('.//w:sectPr', namespaces=NS)),
          'proofs': json.loads((HERE/'proof_status.json').read_text())}
(HERE / 'current_integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
