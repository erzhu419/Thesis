"""Restore native math italic only where first-run text italic shears delimiters."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import json
A=Path(__file__).resolve().parent
P=A.parents[1]/'调整完格式的_括号正体与公式校对修复_20261019.docx'
BACK=A/'before_native_delimiter_shear_fix.docx'
assert not BACK.exists()
BACK.write_bytes(P.read_bytes())
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
with ZipFile(P) as z:old=E.fromstring(z.read('word/document.xml'))
r=deepcopy(old);changed=[]
def txt(n):return ''.join(t.text or '' for t in n.iter(M+'t'))
for ordinal, math in enumerate(r.iter(M+'oMath'),1):
    if not list(math.iter(M+'d')):continue
    run=next((x for x in math.iter(M+'r') if txt(x)),None)
    if run is None:continue
    mp=run.find(M+'rPr');wp=run.find(W+'rPr')
    if mp is None or wp is None:continue
    nor=mp.find(M+'nor');italic=wp.find(W+'i')
    if nor is None or italic is None or italic.get(W+'val','1') in ('0','false','off'):continue
    assert mp.find(M+'scr') is None and mp.find(M+'sty') is None
    assert not any('\u4e00'<=c<='\u9fff' for c in txt(run))
    mp.remove(nor)
    bold=wp.find(W+'b')
    is_bold=bold is not None and bold.get(W+'val','1') not in ('0','false','off')
    style=('bi' if is_bold else 'i') if any(c.isalpha() for c in txt(run)) else 'p'
    sty=E.SubElement(mp,M+'sty');sty.set(M+'val',style)
    if is_bold:wp.remove(bold)
    for name in ('i','iCs'):
        for n in list(wp.findall(W+name)):wp.remove(n)
    f=wp.find(W+'rFonts')
    if f is None:f=E.Element(W+'rFonts');wp.insert(0,f)
    f.attrib.clear()
    for slot in ('ascii','hAnsi','eastAsia','cs'):f.set(W+slot,'Cambria Math')
    changed.append({'math_ordinal':ordinal,'first_run_text':txt(run),'native_math_style':sty.get(M+'val')})
# Literal and semantic math content must remain exact; body and cached TOC remain unchanged.
assert len(changed)==106
assert [txt(x) for x in old.iter(M+'oMath')]==[txt(x) for x in r.iter(M+'oMath')]
def strip_properties(node):
    q=deepcopy(node)
    for n in list(q.iter()):
        if n.tag in (W+'rPr',M+'rPr'):n.getparent().remove(n)
    return E.tostring(q,method='c14n')
assert strip_properties(old)==strip_properties(r)
# Only the first runs on the explicit target list may differ.
a=deepcopy(old);b=deepcopy(r)
for row in changed:
    for root in (a,b):
        math=list(root.iter(M+'oMath'))[row['math_ordinal']-1]
        run=next(x for x in math.iter(M+'r') if txt(x))
        for prop in (M+'rPr',W+'rPr'):
            n=run.find(prop)
            if n is not None:run.remove(n)
assert E.tostring(a,method='c14n')==E.tostring(b,method='c14n')
with ZipFile(BACK) as zin,ZipFile(P,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        zout.writestr(item,E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True) if item.filename=='word/document.xml' else zin.read(item.filename))
(A/'native_delimiter_shear_fix.json').write_text(json.dumps({'changed_first_runs':changed,'count':len(changed),'characters_and_structure_unchanged':True,'all_other_formatting_unchanged':True,'backup':str(BACK)},ensure_ascii=False,indent=2)+'\n')
print('Restored native math style for 106 first runs; all formula characters and other formatting unchanged.')
