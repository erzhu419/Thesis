"""Shorten only two blank gaps so signature dates stay on their template page."""
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from edit_word import BASE, NS, content

with ZipFile(BASE) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
blocks = list(doc.find('w:body', NS))
assert content(blocks[48]).startswith('学位论文作者签名：')
assert content(blocks[60]).startswith('日期：')
signature = blocks[48].findall('w:r', NS)[1].find('w:t', NS)
assert signature.text == ' ' * 11
signature.text = ' ' * 8
date_gap = blocks[60].findall('w:r', NS)[7].find('w:t', NS)
assert date_gap.text == ' ' * 6
date_gap.text = ' '
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(BASE, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
print('Shortened signature blank by 3 spaces and date-column gap by 5 spaces.')
