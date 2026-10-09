"""Finish the coefficient condition and the human-readable CVaR endpoint."""
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
beta = next(n for n in blocks[1518].xpath('.//m:sSub', namespaces=NS)
            if content(n) == 'β_{lcb}')
tail = next(t for t in blocks[1518].xpath('.//m:t', namespaces=NS)
            if t.text == '\u2001\u2001β')
tail.text = '\u2001\u2001'
tail.getparent().addnext(deepcopy(beta))
assert 'β<0' not in content(blocks[1518])
interval = blocks[2482].xpath('.//m:t', namespaces=NS)
end = next(t for t in interval if t.text == ')')
assert interval[interval.index(end) - 1].text == '1'
end.text = ']'
texts = blocks[2490].xpath('.//w:t', namespaces=NS)
assert texts[0].text.startswith('累计成本的分布函数')
texts[0].text = '尾部概率小于1时，' + texts[0].text
assert texts[-1].text.endswith('证毕。')
texts[-1].text = texts[-1].text[:-3] + (
    '若尾部概率等于1，所有可行密度均在几乎处处等于1；'
    '取不大于成本下界的阈值时，原优化目标也等于期望成本。'
    '因此该端点的风险包络表示同样成立。证毕。')
for i in [1518, 2482, 2490]:
    by_block[i]['new'] = content(blocks[i])
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True,
                                     encoding='UTF-8', standalone=True)
out = HERE / '排版完成_三项校订.docx'
with ZipFile(out, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
path.write_bytes(out.read_bytes())
(HERE / 'word_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
print('46 approved blocks; coefficient condition and CVaR endpoint synchronized')
