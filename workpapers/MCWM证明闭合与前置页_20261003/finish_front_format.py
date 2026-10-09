"""Preserve template paragraph-mark and East Asian face selection on the front pages."""
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from edit_word import BASE, TEMPLATE, NS, q, template_front, Doc, set_text, content
import re

with ZipFile(BASE) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
body = doc.find('w:body', NS)
styles = E.fromstring(parts['word/styles.xml'])
sid = next(n.get(q('w:styleId')) for n in styles.findall('w:style', NS)
           if n.xpath('./w:name/@w:val', namespaces=NS) == ['学位论文前置页'])
front = template_front(Doc(TEMPLATE), sid)
old = list(body)[:62]
for i in (6, 7, 8, 9, 19, 21, 22, 27, 28, 29, 30, 37, 42):
    set_text(front[i], content(old[i]))
for i in (13, 33):
    for new_row, old_row in zip(front[i].findall('w:tr', NS), old[i].findall('w:tr', NS)):
        set_text(new_row.findall('w:tc', NS)[1].find('w:p', NS),
                 content(old_row.findall('w:tc', NS)[1]))
for n in old:
    body.remove(n)
for i, n in enumerate(front):
    body.insert(i, n)
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(BASE, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
print('Restored template paragraph marks and font hints on 62 front blocks.')
