from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import json
A=Path(__file__).resolve().parent
D=Path('/mnt/d/Paper/Thesis/Gemini/language_v2_20261015')
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
with ZipFile(D/'current.docx') as z:parts={n:z.read(n) for n in z.namelist()}
r=E.fromstring(parts['word/document.xml'])
fix=json.loads((A/'plain_x_single_node_repair.json').read_text())
ps=[p for p in r.iter(W+'p') if ''.join(n.text or '' for n in p.iter(W+'t')).startswith(fix['paragraph_start'])]
p,=ps
maths=list(p.iter(M+'oMath'));assert len(maths)==12
node=maths[0]; node.getparent().replace(node,E.fromstring((A/'correct_plain_x.xml').read_bytes()))
old='robust value learning with two regularization terms'
new='robust value learning with weight-norm and ensemble-disagreement regularization'
ps=[p for p in r.iter(W+'p') if old in ''.join(n.text or '' for n in p.iter(W+'t'))]
p,=ps
nodes=list(p.iter(W+'t'));value=''.join(n.text or '' for n in nodes)
start=value.index(old);end=start+len(old);offset=0;first=True
for n in nodes:
 v=n.text or '';lo=max(start-offset,0);hi=min(end-offset,len(v))
 if lo<hi:n.text=v[:lo]+(new if first else '')+v[hi:];first=False
 offset+=len(v)
parts['word/document.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
out=D/'final.docx'
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for name,data in parts.items():z.writestr(name,data)
print(out)
