from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import re,json
A=Path(__file__).resolve().parent;D=Path('/mnt/d/Paper/Thesis/Gemini/language_v2_20261015')
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
def text(e):return ''.join(n.text or '' for n in e.iter() if n.tag in (W+'t',M+'t'))
with ZipFile(D/'final.docx') as z:parts={n:z.read(n) for n in z.namelist()}
r=E.fromstring(parts['word/document.xml']);body=r.find(W+'body');changes=[]
pattern=re.compile(r'([+-]?\d+(?:\.\d+)?)\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]')
for label in ('表6-3 ','表6-4 ','表6-6 '):
 caption,=[p for p in body if p.tag==W+'p' and text(p).startswith(label) and p.find(W+'pPr/'+W+'pStyle') is not None and p.find(W+'pPr/'+W+'pStyle').get(W+'val')=='tablecaption']
 table=caption.getnext();assert table.tag==W+'tbl'
 for row_i,row in enumerate(table.findall(W+'tr')[1:],1):
  for col_i,cell in enumerate(row.findall(W+'tc'),1):
   value=text(cell);match=pattern.fullmatch(value)
   if match:
    p=cell.find(W+'p');pr=deepcopy(p.find(W+'pPr'))
    for n in list(cell):
     if n.tag!=W+'tcPr':cell.remove(n)
    p=E.SubElement(cell,W+'p');p.append(pr)
    jc=pr.find(W+'jc')
    if jc is None:jc=E.SubElement(pr,W+'jc')
    jc.set(W+'val','center')
    lines=[line.replace('-', '−') for line in [match[1],'['+match[2]+',',match[3]+']']]
    for i,line in enumerate(lines):
     run=E.SubElement(p,W+'r');rp=E.SubElement(run,W+'rPr');font=E.SubElement(rp,W+'rFonts')
     for k,v in [('ascii','Times New Roman'),('hAnsi','Times New Roman'),('eastAsia','宋体')]:font.set(W+k,v)
     for k,v in [('sz','21'),('szCs','21'),('b','0'),('bCs','0')]:E.SubElement(rp,W+k).set(W+'val',v)
     if i:E.SubElement(run,W+'br')
     E.SubElement(run,W+'t').text=line
    assert re.sub(r'\s','',value)==text(cell).replace('−','-')
    changes.append(dict(table=label.strip(),row=row_i,column=col_i,old=value,new='\n'.join(lines),scope='Word only; numeric values unchanged'))
   if match or label!='表6-4 ':
    for p in cell.findall(W+'p'):
     pr=p.find(W+'pPr');spacing=pr.find(W+'spacing')
     if spacing is None:spacing=E.SubElement(pr,W+'spacing')
     spacing.set(W+'line','260');spacing.set(W+'lineRule','atLeast')
assert len(changes)==39,len(changes)
parts['word/document.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(D/'layout_final.docx','w',ZIP_DEFLATED) as z:
 for name,data in parts.items():z.writestr(name,data)
(A/'word_ci_layout_changes.json').write_text(json.dumps({'changes':changes,'reason':'Narrow columns clipped inline confidence intervals; print all numbers on three explicit lines.'},ensure_ascii=False,indent=2))
print('39 confidence-interval cells formatted; no numeric changes')
