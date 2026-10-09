"""Keep the user's exact abstract XML after Word refreshes fields and pagination."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import json
HERE=Path(__file__).resolve().parent
TARGET=HERE.parents[1]/'output/排版完成_导师意见复核_20261005.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
PID='{http://schemas.microsoft.com/office/word/2010/wordml}paraId'
with ZipFile(HERE/'用户找回原稿_备份.docx') as z:
    old=E.fromstring(z.read('word/document.xml')).find('w:body',NS).findall('w:p',NS)[61:80]
with ZipFile(TARGET) as z:data={n:z.read(n) for n in z.namelist()}
doc=E.fromstring(data['word/document.xml']);body=doc.find('w:body',NS)
byid={p.get(PID):p for p in body.findall('w:p',NS)}
for source in old:
    current=byid[source.get(PID)]
    assert ''.join(current.xpath('.//w:t/text()',namespaces=NS))==''.join(source.xpath('.//w:t/text()',namespaces=NS))
    body.replace(current,deepcopy(source))
byid={p.get(PID):p for p in body.findall('w:p',NS)}
assert [E.tostring(byid[p.get(PID)]) for p in old]==[E.tostring(p) for p in old]
data['word/document.xml']=E.tostring(doc,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(TARGET,'w',ZIP_DEFLATED) as z:
    for n,v in data.items():z.writestr(n,v)
(HERE/'摘要锁定检查.json').write_text(json.dumps({'abstract_and_keywords_paragraphs':19,
    'text_unchanged':True,'paragraph_xml_unchanged':True,'source':'用户找回原稿_备份.docx'},ensure_ascii=False,indent=2))
print('ABSTRACT_AND_KEYWORDS_UNCHANGED')
