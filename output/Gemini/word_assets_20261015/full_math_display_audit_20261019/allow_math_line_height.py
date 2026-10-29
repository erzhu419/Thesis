"""Let formula lines grow beyond their existing fixed minimum; no text changes."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import json
A=Path(__file__).resolve().parent
P=A.parents[1]/'调整完格式的_全公式显示校订_20261019.docx'
B=A/'before_math_line_height_fix.docx'
assert not B.exists();B.write_bytes(P.read_bytes())
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
with ZipFile(B) as z:old=E.fromstring(z.read('word/document.xml'));styles=E.fromstring(z.read('word/styles.xml'))
r=deepcopy(old);by={s.get(W+'styleId'):s for s in styles.findall(W+'style')}
def effective_spacing(p):
    ppr=p.find(W+'pPr');props=[]
    if ppr is not None:props.append(ppr)
    sn=ppr.find(W+'pStyle') if ppr is not None else None
    sid=sn.get(W+'val') if sn is not None else 'Normal'
    while sid in by:
        s=by[sid];pr=s.find(W+'pPr')
        if pr is not None:props.append(pr)
        base=s.find(W+'basedOn');sid=base.get(W+'val') if base is not None else ''
    dd=styles.find(W+'docDefaults/'+W+'pPrDefault/'+W+'pPr')
    if dd is not None:props.append(dd)
    vals={}
    for pr in props:
        sp=pr.find(W+'spacing')
        if sp is not None:
            for k,v in sp.attrib.items():vals.setdefault(k,v)
    return vals
changes=[]
for p in r.iter(W+'p'):
    if not list(p.iter(M+'oMath')):continue
    effective=effective_spacing(p)
    if effective.get(W+'lineRule')!='exact':continue
    ppr=p.find(W+'pPr')
    if ppr is None:ppr=E.Element(W+'pPr');p.insert(0,ppr)
    sp=ppr.find(W+'spacing')
    if sp is None:
        sp=E.Element(W+'spacing')
        anchor=next((n for n in ppr if n.tag in (W+'ind',W+'jc',W+'textAlignment',W+'rPr')),None)
        ppr.insert(ppr.index(anchor) if anchor is not None else len(ppr),sp)
    sp.set(W+'line',effective[W+'line']);sp.set(W+'lineRule','atLeast')
    changes.append({'paragraph_xpath':r.getroottree().getpath(p),'minimum_line_twips':effective[W+'line']})
assert len(list(old.iter(M+'oMath')))==len(list(r.iter(M+'oMath')))==4738
assert [E.tostring(x,method='c14n') for x in old.iter(M+'oMath')]==[E.tostring(x,method='c14n') for x in r.iter(M+'oMath')]
assert [(x.tag,x.text) for x in old.iter() if x.tag in (W+'t',M+'t')]==[(x.tag,x.text) for x in r.iter() if x.tag in (W+'t',M+'t')]
with ZipFile(B) as zin,ZipFile(P,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():zout.writestr(item,E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True) if item.filename=='word/document.xml' else zin.read(item.filename))
(A/'math_line_height_fix.json').write_text(json.dumps({'paragraphs_changed':len(changes),'changes':changes,'all_text_unchanged':True,'all_OMML_unchanged':True,'backup':str(B)},ensure_ascii=False,indent=2)+'\n')
print('Formula paragraphs changed from fixed to existing minimum line height:',len(changes))
