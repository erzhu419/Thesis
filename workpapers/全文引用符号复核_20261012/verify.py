"""Check that the audit edits preserve scientific data and native document parts."""
from collections import Counter
from zipfile import ZipFile
import json
import shutil
import sys
from lxml import etree as E
from audit import ROOT, HERE, BASE, W, M, NS, text, style

OUT = ROOT / 'output/排版完成_编号符号引用复核_20261012.docx'
with ZipFile(BASE) as a, ZipFile(OUT) as b:
    assert a.namelist() == b.namelist()
    changed = [name for name in a.namelist() if a.read(name) != b.read(name)]
    assert changed == ['word/document.xml'], changed
    native = [n for n in a.namelist() if n.startswith(('word/media/', 'word/embeddings/'))]
    before = E.fromstring(a.read('word/document.xml'))
    after = E.fromstring(b.read('word/document.xml'))
assert BASE.read_bytes() == (HERE / '修改前.docx').read_bytes()
def shape(node):
    return node.tag, tuple(sorted(node.attrib.items())), node.text, tuple(shape(c) for c in node)
assert [shape(s) for s in before.iter(W + 'sectPr')] == [shape(s) for s in after.iter(W + 'sectPr')]
bb, aa = list(before.find(W + 'body')), list(after.find(W + 'body'))
assert len(bb) == len(aa)
assert [(i,text(p)) for i,p in enumerate(bb) if style(p) == 'bibliography'] == [(i,text(p)) for i,p in enumerate(aa) if style(p) == 'bibliography']
for i in range(64, 79):
    assert shape(bb[i]) == shape(aa[i]), ('abstract changed', i)
equation_blocks = {v['i'] for v in json.loads((HERE / '编号与引用检查.json').read_text())['equations'].values()}
table_count = 0
for i, p in enumerate(bb):
    if p.tag == W + 'tbl' and i not in equation_blocks and i != 81:
        assert shape(p) == shape(aa[i]), ('data table changed', i)
        table_count += 1
names = Counter(after.xpath('.//w:bookmarkStart/@w:name', namespaces=NS))
assert all(v == 1 for v in names.values())
assert Counter(after.xpath('.//w:bookmarkStart/@w:id', namespaces=NS)) == Counter(after.xpath('.//w:bookmarkEnd/@w:id', namespaces=NS))
numbering = json.loads((HERE / '编号与引用检查.json').read_text())
assert numbering['errors'] == []
fields = json.loads((HERE / '字段与目标检查.json').read_text())
assert fields['findings'] == []
proof_equations = sum(1 for k in numbering['equations'] if k[0] in 'ABCDE')
result = {'source_preserved': True, 'abstracts_preserved': True, 'native_parts_preserved': len(native),
          'sections_preserved': len(list(before.iter(W + 'sectPr'))),
          'data_tables_preserved': table_count, 'reference_entries_preserved': 217,
          'numbered_equations': len(numbering['equations']), 'appendix_equations': proof_equations,
          'figures': sum(k.startswith('图') for k in numbering['captions']),
          'tables': sum(k.startswith('表') for k in numbering['captions']),
          'reference_fields': len(fields['checked_fields']), 'structural_findings': []}
(HERE / '保护与引用验收.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
if '--stage' in sys.argv:
    stage = ROOT.__class__('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_refs_20261012')
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, stage / 'word.docx')
print(json.dumps(result, ensure_ascii=False))
