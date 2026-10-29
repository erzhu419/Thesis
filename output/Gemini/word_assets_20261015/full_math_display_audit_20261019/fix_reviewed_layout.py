"""Fix only the nine formula layouts confirmed in native Word screenshots."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import importlib.util,json
A=Path(__file__).resolve().parent
SRC=A.parents[1]/'调整完格式的_全公式显示校订_20261019.docx'
OUT=A.parents[1]/'调整完格式的_全公式显示终校_20261019.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
with ZipFile(SRC) as z:old=E.fromstring(z.read('word/document.xml'))
r=deepcopy(old);ms=list(r.iter(M+'oMath'))
def text(n):return ''.join(t.text or '' for t in n.iter(M+'t'))
def eqarray(rows,font_run):
    arr=E.Element(M+'eqArr');pr=E.SubElement(arr,M+'eqArrPr')
    n=E.SubElement(pr,M+'baseJc');n.set(M+'val','center')
    ctrl=E.SubElement(pr,M+'ctrlPr');wp=deepcopy(font_run.find(W+'rPr'))
    if wp is None:wp=E.Element(W+'rPr')
    for name,value in [('i','0'),('iCs','0'),('noProof','1')]:
        n=wp.find(W+name)
        if n is None:n=E.SubElement(wp,W+name)
        n.set(W+'val',value)
    ctrl.append(wp)
    for children in rows:
        e=E.SubElement(arr,M+'e')
        for n in children:e.append(n)
    return arr
ci_ids=[1357,1361,1365,1369,1373,1377]
for idx in ci_ids:
    math=ms[idx-1];nodes=list(math);comma=next(i for i,n in enumerate(nodes) if text(n)==',')
    arr=eqarray([nodes[:comma+1],nodes[comma+1:]],next(math.iter(M+'r')))
    for n in list(math):math.remove(n)
    math.append(arr)
# 4-21: keep each norm whole; three short lines preserve the existing formula sequence.
math=ms[1761];nodes=list(math);le=next(i for i,n in enumerate(nodes) if text(n)=='≤');comma=next(i for i,n in enumerate(nodes) if i>le and text(n)==',')
arr=eqarray([nodes[:le],nodes[le:comma+1],nodes[comma+1:]],next(math.iter(M+'r')))
for n in list(math):math.remove(n)
math.append(arr)
# B-53: split each long bound at its outer addition and remove invisible alignment markers.
math=ms[3991];arr=math.find(M+'eqArr');rows=list(arr.findall(M+'e'));newrows=[]
for number,row in enumerate(rows):
    for run in list(row.iter(M+'r')):
        if text(run)=='&':run.getparent().remove(run)
    nodes=list(row)
    if number<3:
        plus=next(i for i,n in enumerate(nodes) if n.tag==M+'r' and '+' in text(n))
        value=text(nodes[plus]);at=value.index('+')
        if at:
            left=deepcopy(nodes[plus]);left.find(M+'t').text=value[:at]
            nodes[plus].find(M+'t').text=value[at:]
            newrows.extend([nodes[:plus]+[left],nodes[plus:]])
        else:newrows.extend([nodes[:plus],nodes[plus:]])
    else:newrows.append(nodes)
replacement=eqarray(newrows,next(math.iter(M+'r')))
math.remove(arr);math.append(replacement)
# B-57: break the expectation's summand before its square root, inside the original delimiter.
math=ms[3998];d=next(math.iter(M+'d'));arg=d.find(M+'e');nodes=list(arg);rad=next(i for i,n in enumerate(nodes) if n.tag==M+'rad')
arr=eqarray([nodes[:rad],nodes[rad:]],next(math.iter(M+'r')))
for n in list(arg):arg.remove(n)
arg.append(arr)
# User explicitly authorized only these two REF code fixes; leave result caches for native Word.
changed_fields=[]
for p in r.iter(W+'p'):
    plain=''.join(n.text or '' for n in p.iter(W+'t'))
    if '错误' not in plain:continue
    for n in p.iter(W+'instrText'):
        code=n.text or ''
        if any('REF '+name in code for name in ('eqA_005','eqA_004')):
            changed_fields.append({'bookmark_code':code,'native_refresh_pending':True})
assert len(changed_fields)==2,changed_fields
spec=importlib.util.spec_from_file_location('canon',A.parent/'exact_library_v2_20261016/prepare_math_verification.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
assert len(list(r.iter(M+'oMath')))==4738
assert all(h.content([a])==h.content([b]) for a,b in zip(old.iter(M+'oMath'),r.iter(M+'oMath')))
# All visible plain Word text, including the not-yet-refreshed result caches, is exact.
assert [(n.tag,n.text) for n in old.iter() if n.tag==W+'t']==[(n.tag,n.text) for n in r.iter() if n.tag==W+'t']
assert not OUT.exists()
with ZipFile(SRC) as zin,ZipFile(OUT,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():zout.writestr(item,E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True) if item.filename=='word/document.xml' else zin.read(item.filename))
(A/'reviewed_layout_fixes.json').write_text(json.dumps({'output':str(OUT),'math_ids':ci_ids+[1762,3992,3999],'math_semantics_unchanged':True,'all_plain_Word_text_unchanged':True,'REF_refresh_targets':changed_fields,'native_recheck_pending':True},ensure_ascii=False,indent=2)+'\n')
print('Created final-review candidate; 9 formula layouts and exactly 2 authorized REF codes corrected.')
