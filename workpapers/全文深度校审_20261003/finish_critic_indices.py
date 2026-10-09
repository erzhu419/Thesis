"""Keep decision-event and ensemble-member indices distinct in RE-SAC."""
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from extract_word import NS, content
from revise_word import q, run

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'output/排版完成.docx'
with ZipFile(BASE) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
blocks = list(doc.find('w:body', NS))
assert '(3-23)' in content(blocks[904])
for t in blocks[904].xpath('.//m:t', namespaces=NS):
    if t.text == 'j':
        t.text = 'j′'
for block in blocks[899:935]:
    for t in block.xpath('.//m:t', namespaces=NS):
        if t.text == 'k':
            t.text = 'j'
    for t in block.xpath('.//w:t', namespaces=NS):
        if t.text == 'k':
            t.text = 'j'
        elif '第k个' in (t.text or ''):
            t.text = t.text.replace('第k个', '第j个')
blocks[900].append(run('以下以j及内层求和的j′表示集成成员索引，与决策事件序号k区分。'))
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
stage = HERE / '排版完成_评论家下标收尾.docx'
with ZipFile(stage, 'w', ZIP_DEFLATED) as z:
    for name, value in parts.items():
        z.writestr(name, value)
shutil.copy2(stage, BASE)
print('RE-SAC ensemble indices updated in definitions, targets and losses.')
