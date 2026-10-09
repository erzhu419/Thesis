"""Synchronize the remaining body/appendix names without replacing objects."""
from copy import deepcopy
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
    parts = {name: z.read(name) for name in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
blocks = list(doc.find('w:body', NS))
changes = json.loads((HERE / 'word_changes.json').read_text())
by_block = {c['block']: c for c in changes}
targets = [414, 989, 990, 991, 1518, 1519, 2411, 2412, 2413, 2484]
previous = {i: content(blocks[i]) for i in targets}

def tag(name):
    prefix, local = name.split(':')
    return '{' + NS[prefix] + '}' + local

def penalty(node, symbol='c', suffix='pen'):
    count = 0
    for t in list(node.xpath('.//m:t', namespaces=NS)):
        if symbol == 'c' and t.text == 'γc':
            run = t.getparent()
            prefix = deepcopy(run)
            prefix.find('m:t', NS).text = 'γ'
            run.addprevious(prefix)
            t.text = 'c'
        if t.text != symbol:
            continue
        run = t.getparent()
        container = run.getparent()
        if container.tag == tag('m:sub') or (container.tag == tag('m:e')
                and container.getparent().tag in [tag('m:sSub'), tag('m:sSubSup')]):
            continue
        sub = E.Element(tag('m:sSub'))
        base = E.SubElement(sub, tag('m:e'))
        container.replace(run, sub)
        base.append(run)
        label = E.SubElement(sub, tag('m:sub'))
        r = E.SubElement(label, tag('m:r'))
        rpr = E.SubElement(r, tag('m:rPr'))
        E.SubElement(rpr, tag('m:sty')).set(tag('m:val'), 'p')
        fonts = E.SubElement(E.SubElement(r, tag('w:rPr')), tag('w:rFonts'))
        fonts.set(tag('w:ascii'), 'Cambria Math')
        fonts.set(tag('w:hAnsi'), 'Cambria Math')
        E.SubElement(r, tag('m:t')).text = suffix
        count += 1
    return count

for i in [989, 990, 991, 2411, 2412, 2413]:
    assert penalty(blocks[i]) >= 1, i
for i in [1518, 1519]:
    assert penalty(blocks[i], 'β', 'lcb') >= 1, i
for n in blocks[2413].xpath('.//m:sSub | .//m:sSubSup', namespaces=NS):
    if content(n.find('m:e', NS)) in ['c', 'κ'] and content(n.find('m:sub', NS)) == 'k':
        n.find('m:sub/m:r/m:t', NS).text = 'j'
assert 'c_{k}' not in content(blocks[2413])

row = next(r for r in blocks[414].findall('w:tr', NS)
           if '固定策略下第j个成员的有效惩罚' in content(r))
math = row.findall('w:tc', NS)[0].xpath('.//m:oMath', namespaces=NS)[0]
prefix = deepcopy(next(n for n in blocks[989].xpath('.//m:sSub', namespaces=NS)
                       if content(n) == 'c_{pen}'))
for run in prefix.xpath('.//m:r', namespaces=NS):
    pr = run.find('w:rPr', NS)
    E.SubElement(pr, tag('w:sz')).set(tag('w:val'), '21')
    E.SubElement(pr, tag('w:szCs')).set(tag('w:val'), '21')
comma = deepcopy(math.find('m:r', NS))
comma.find('m:t', NS).text = '，'
math.insert(0, comma)
math.insert(0, prefix)
texts = row.findall('w:tc', NS)[1].xpath('.//w:t', namespaces=NS)
assert '固定策略下第' in ''.join(t.text or '' for t in texts)
texts[0].text = '固定惩罚、' + (texts[0].text or '')
texts = blocks[2484].xpath('.//w:t', namespaces=NS)
old = '为轨迹的可测的非负重加权密度'
joined = ''.join(t.text or '' for t in texts)
start = joined.index(old)
end, offset = start + len(old), 0
for t in texts:
    value = t.text or ''
    lo, hi = offset, offset + len(value)
    if lo < end and hi > start:
        left, right = max(start - lo, 0), min(end - lo, len(value))
        t.text = value[:left] + ('为可测的非负轨迹重加权密度' if lo <= start < hi else '') + value[right:]
    offset = hi

for i, before in previous.items():
    if i in by_block:
        by_block[i]['new'] = content(blocks[i])
    else:
        changes.append({'block': i, 'old': before, 'new': content(blocks[i])})
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
out = HERE / '排版完成_三项校订.docx'
with ZipFile(out, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
path.write_bytes(out.read_bytes())
(HERE / 'word_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'changed_blocks': len(changes), 'synchronized': targets}, ensure_ascii=False))
