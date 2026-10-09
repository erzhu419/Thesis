"""Make the first CVaR formula use the same cost hypothesis as appendix A."""
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
block = list(doc.find('w:body', NS))[683]
before = content(block)
changed = 0
for text in block.xpath('.//w:t', namespaces=NS):
    if '令随机累计成本为' in (text.text or ''):
        text.text = text.text.replace('令随机累计成本为', '令有界可测的随机累计成本为')
        changed += 1
assert changed == 1
changes = json.loads((HERE / 'word_changes.json').read_text())
assert not any(item['block'] == 683 for item in changes)
changes.append({'block': 683, 'old': before, 'new': content(block)})
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True,
                                     encoding='UTF-8', standalone=True)
out = HERE / '排版完成_三项校订.docx'
with ZipFile(out, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
path.write_bytes(out.read_bytes())
(HERE / 'word_changes.json').write_text(
    json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
print('46 approved block changes; no experimental values changed')
