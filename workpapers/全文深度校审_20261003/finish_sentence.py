"""Place the ensemble-index explanation before its equation introduction."""
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from extract_word import NS

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'output/排版完成.docx'
with ZipFile(BASE) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
p = list(doc.find('w:body', NS))[900]
nodes = p.xpath('.//w:t', namespaces=NS)
assert sum('，集成均值为以下以' in (t.text or '') for t in nodes) == 1
for t in nodes:
    if '，集成均值为以下以' in (t.text or ''):
        t.text = t.text.replace('，集成均值为以下以', '。以')
assert nodes[-1].text == '区分。'
nodes[-1].text = '区分。集成均值为'
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
stage = HERE / '排版完成_句式收尾.docx'
with ZipFile(stage, 'w', ZIP_DEFLATED) as z:
    for name, value in parts.items():
        z.writestr(name, value)
shutil.copy2(stage, BASE)
print('Ensemble definition sentence corrected without changing equations.')
