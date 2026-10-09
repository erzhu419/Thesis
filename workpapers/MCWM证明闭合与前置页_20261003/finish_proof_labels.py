"""Keep the established bold proof label as a valid first run property."""
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from edit_word import BASE, NS, q, content

with ZipFile(BASE) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
changed = 0
for p in doc.findall('.//w:p', NS):
    text = content(p)
    if text.startswith(('证明。 价值误差非负', '证明。 以有限索引集')):
        r = next(n for n in p.findall('w:r', NS) if content(n) == '证明。')
        pr = r.find('w:rPr', NS)
        if pr is None:
            pr = E.Element(q('w:rPr'))
            r.insert(0, pr)
        bold = pr.find('w:b', NS)
        if bold is None:
            bold = E.SubElement(pr, q('w:b'))
        bold.set(q('w:val'), '1')
        changed += 1
assert changed == 2
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(BASE, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
print('Updated proof labels:', changed)
