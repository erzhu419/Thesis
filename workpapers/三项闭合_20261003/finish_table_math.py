"""Apply the existing symbols-table size to newly inserted native math runs."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
path = ROOT / 'output/排版完成.docx'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with ZipFile(path) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
table = list(doc.find('w:body', ns))[414]
count = 0
for run in table.xpath('.//m:r', namespaces=ns):
    pr = run.find('w:rPr', ns)
    if pr is None:
        pr = E.SubElement(run, '{' + ns['w'] + '}rPr')
        run.insert(1 if run.find('m:rPr', ns) is not None else 0, pr)
    size = pr.find('w:sz', ns)
    if size is None:
        E.SubElement(pr, '{' + ns['w'] + '}sz').set('{' + ns['w'] + '}val', '21')
        count += 1
assert count > 0
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
out = HERE / '排版完成_三项校订.docx'
with ZipFile(out, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
path.write_bytes(out.read_bytes())
print(f'{count} inserted math runs set to the existing 10.5 pt table size')
