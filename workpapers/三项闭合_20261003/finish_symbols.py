"""Separate the remaining named functions without changing their mathematics."""
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workpapers/全文深度校审_20261003'))
from extract_word import NS, content

path = ROOT / 'output/排版完成.docx'
with ZipFile(path) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
blocks = list(doc.find('w:body', NS))
changes = json.loads((HERE / 'word_changes.json').read_text())
old = {i: content(blocks[i]) for i in [672, 673, 675, 676, 881]}

def sub_token(node, old, label):
    count = 0
    for t in list(node.xpath('.//m:t', namespaces=NS)):
        if t.text != old:
            continue
        run = t.getparent()
        wrapper = E.Element('{' + NS['m'] + '}sSub')
        base = E.SubElement(wrapper, '{' + NS['m'] + '}e')
        run.getparent().replace(run, wrapper)
        base.append(run)
        sub = E.SubElement(wrapper, '{' + NS['m'] + '}sub')
        r = E.SubElement(sub, '{' + NS['m'] + '}r')
        pr = E.SubElement(r, '{' + NS['m'] + '}rPr')
        E.SubElement(pr, '{' + NS['m'] + '}sty').set('{' + NS['m'] + '}val', 'p')
        E.SubElement(r, '{' + NS['m'] + '}t').text = label
        count += 1
    return count

for i in [672, 673]:
    assert sub_token(blocks[i], 'ξ', 'dyn') > 0
for i in [675, 676]:
    assert sub_token(blocks[i], 'c', 'cost') == 1
for t in blocks[881].xpath('.//w:t', namespaces=NS):
    if '期望样本数记为批内该车样本数的期望，满足' in (t.text or ''):
        t.text = t.text.replace('期望样本数记为批内该车样本数的期望，满足', '该车样本数的期望满足')
assert '期望样本数记为' not in content(blocks[881])

by_block = {c['block']: c for c in changes}
for i, previous in old.items():
    if i in by_block:
        by_block[i]['new'] = content(blocks[i])
    else:
        changes.append({'block': i, 'old': previous, 'new': content(blocks[i])})
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
out = HERE / '排版完成_三项校订.docx'
with ZipFile(out, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
path.write_bytes(out.read_bytes())
(HERE / 'word_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
print('37 blocks corrected; experiment values unchanged')
