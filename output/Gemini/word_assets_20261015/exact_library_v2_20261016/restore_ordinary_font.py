"""Restore only the three body-font runs in the supplied D.1 paragraph."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import sys
import native_helpers as h
W=h.W;M=h.M;path=Path(sys.argv[1])
with ZipFile(path)as z:parts={n:z.read(n)for n in z.namelist()}
r=E.fromstring(parts['word/document.xml']);p=next(p for p in r.find(W+'body')if h.text(p).startswith('一般地，设')and '实赋范空间'in h.text(p))
before=h.text(p);changed=0
for run in p.iter(W+'r'):
 fonts=run.find(W+'rPr/'+W+'rFonts')
 if run.find(W+'t')is not None and fonts is not None and fonts.get(W+'eastAsia')=='Times New Roman':
  fonts.attrib.pop(W+'eastAsia');changed+=1
assert changed==3,changed
assert h.text(p)==before
parts['word/document.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(path,'w',ZIP_DEFLATED)as z:
 for n,d in parts.items():z.writestr(n,d)
print(str(path),'3 ordinary runs restored to inherited body East Asian font; visible text unchanged')
