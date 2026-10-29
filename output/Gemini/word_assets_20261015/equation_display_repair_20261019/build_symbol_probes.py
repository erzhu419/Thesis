from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import json

A=Path(__file__).resolve().parent
SRC=A.parents[1]/'调整完格式的.docx'
OLD=Path('/mnt/d/Paper/Thesis/Gemini/重构论文_导言小结同步_20261018.docx')
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'

def text(node):
    return ''.join(t.text or '' for t in node.iter() if t.tag in (W+'t',M+'t'))

def load(path):
    with ZipFile(path) as z:
        return E.fromstring(z.read('word/document.xml'))

root=load(SRC)
old=load(OLD)
def row(root,label):
    return next(tr for tr in root.iter(W+'tr') if '('+label+')' in text(tr))

def font_only(tr):
    for r in tr.iter(M+'r'):
        if '∈' not in text(r):
            continue
        pr=r.find(W+'rPr')
        if pr is None:
            pr=E.Element(W+'rPr');r.insert(1,pr)
        f=pr.find(W+'rFonts')
        if f is None:
            f=E.Element(W+'rFonts');pr.insert(0,f)
        f.attrib.clear()
        for k in ('ascii','hAnsi','eastAsia','cs'):
            f.set(W+k,'Cambria Math')

def restore_symbol(tr,ref):
    oldruns=[r for r in ref.iter(M+'r') if '∈' in text(r)]
    runs=[r for r in tr.iter(M+'r') if '∈' in text(r)]
    assert len(oldruns)==len(runs)
    for r,o in zip(runs,oldruns):
        for p in list(r):
            if p.tag in (M+'rPr',W+'rPr'):
                r.remove(p)
        for p in reversed([p for p in o if p.tag in (M+'rPr',W+'rPr')]):
            r.insert(0,deepcopy(p))

cases=[]
for label in ('4-6','2-35'):
    cur=row(root,label);ref=row(old,label)
    cases.append((label+' current',deepcopy(cur)))
    a=deepcopy(cur);font_only(a)
    cases.append((label+' only membership font set to Cambria Math',a))
    b=deepcopy(cur);restore_symbol(b,ref)
    cases.append((label+' only membership run restored to mathematical mode',b))
cases.append(('3-9 current',deepcopy(row(root,'3-9'))))

body=root.find(W+'body')
sect=deepcopy(body.find(W+'sectPr'))
for n in list(sect):
    if n.tag in (W+'headerReference',W+'footerReference'):
        sect.remove(n)
body.clear()
for i,(desc,tr) in enumerate(cases):
    p=E.SubElement(body,W+'p');pr=E.SubElement(p,W+'pPr')
    sp=E.SubElement(pr,W+'spacing');sp.set(W+'before','80');sp.set(W+'after','0');sp.set(W+'line','240');sp.set(W+'lineRule','auto')
    r=E.SubElement(p,W+'r');rp=E.SubElement(r,W+'rPr');fonts=E.SubElement(rp,W+'rFonts');fonts.set(W+'ascii','Arial');fonts.set(W+'hAnsi','Arial');sz=E.SubElement(rp,W+'sz');sz.set(W+'val','20')
    E.SubElement(r,W+'t').text=chr(65+i)+' - '+desc
    tbl=E.SubElement(body,W+'tbl')
    orig=row(root,'4-6') if False else tr.getparent()
    # The cloned row has no parent; use the unchanged source table properties.
    sr=row(load(SRC),'4-6' if i<3 else ('2-35' if i<6 else '3-9'))
    for tag in ('tblPr','tblGrid'):
        n=sr.getparent().find(W+tag)
        if n is not None:tbl.append(deepcopy(n))
    tbl.append(tr)
    for n in list(tbl.iter()):
        if n.tag in (W+'bookmarkStart',W+'bookmarkEnd',W+'lastRenderedPageBreak'):
            n.getparent().remove(n)
body.append(sect)
with ZipFile(SRC) as z,ZipFile(A/'membership_display_probes.docx','w',ZIP_DEFLATED) as out:
    for item in z.infolist():
        out.writestr(item,E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True) if item.filename=='word/document.xml' else z.read(item.filename))
(A/'membership_probe_cases.json').write_text(json.dumps([chr(65+i)+' - '+d for i,(d,_) in enumerate(cases)],indent=2)+'\n')
print('membership_display_probes.docx')
