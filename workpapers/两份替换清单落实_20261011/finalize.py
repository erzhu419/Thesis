"""Synchronize the remaining appendix header without rewriting other parts."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

HERE=Path(__file__).resolve().parent
OUTPUT=HERE.parents[1]/'output/排版完成_双清单替换_20261011.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
with ZipFile(OUTPUT) as z:
    name='word/header13.xml'
    root=E.fromstring(z.read(name))
    texts=list(root.iter(W+'t'))
    assert ''.join(t.text or '' for t in texts).endswith('附录C 早期BAPR置信加权算子的数学证明')
    for node in texts:
        if node.text=='早期':
            node.text=''
    updated=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    entries=[(i,updated if i.filename==name else z.read(i.filename)) for i in z.infolist()]
with ZipFile(OUTPUT,'w') as z:
    for info,data in entries:
        z.writestr(info,data)
print('Appendix C header synchronized; all other package parts retained.')
