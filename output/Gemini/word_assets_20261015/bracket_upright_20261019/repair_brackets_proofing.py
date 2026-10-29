"""Unify equation brackets and formula-only proofing in a separate DOCX."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import importlib.util
import json

A=Path(__file__).resolve().parent
OUT=A.parents[1]
SRC=A/'source_snapshot.docx'
LIVE=OUT/'调整完格式的_公式显示修复_20261019.docx'
DEST=OUT/'调整完格式的_括号正体与公式校对修复_20261019.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS={'w':W[1:-1],'m':M[1:-1]}
SPACE='{http://www.w3.org/XML/1998/namespace}space'
FULLWIDTH=set('（）［］｛｝')

def text(n):
    return ''.join(t.text or '' for t in n.iter() if t.tag in (W+'t',M+'t'))

def wpr(r):
    pr=r.find(W+'rPr')
    if pr is None:
        pr=E.Element(W+'rPr')
        r.insert(1 if r.find(M+'rPr') is not None else 0,pr)
    return pr

def onoff(pr,name,value):
    n=pr.find(W+name)
    if n is None:n=E.SubElement(pr,W+name)
    n.set(W+'val','1' if value else '0')

def roman_word_properties(pr,math_font):
    if math_font:
        f=pr.find(W+'rFonts')
        if f is None:f=E.Element(W+'rFonts');pr.insert(0,f)
        f.attrib.clear()
        for slot in ('ascii','hAnsi','eastAsia','cs'):
            f.set(W+slot,'Cambria Math')
    onoff(pr,'i',False);onoff(pr,'iCs',False)

def upright_bracket(r,is_fullwidth):
    pr=r.find(M+'rPr')
    if pr is None:pr=E.Element(M+'rPr');r.insert(0,pr)
    if not is_fullwidth:
        # This segment contains brackets only; retain surrounding variable styles.
        for n in list(pr):
            if n.tag in (M+'nor',M+'scr',M+'sty'):pr.remove(n)
        n=E.SubElement(pr,M+'scr');n.set(M+'val','roman')
        n=E.SubElement(pr,M+'sty');n.set(M+'val','p')
    roman_word_properties(wpr(r),not is_fullwidth)

with ZipFile(SRC) as z:original=E.fromstring(z.read('word/document.xml'))
root=deepcopy(original)
inventory=json.loads((A/'literal_bracket_runs.json').read_text())
native=json.loads((A/'native_delimiter_brackets.json').read_text())
resolved=[]
for row in inventory:
    nodes=root.xpath(row['runXPath'],namespaces=NS)
    assert len(nodes)==1 and text(nodes[0])==row['text']
    positions={p['charposition_zero_based'] for p in row['positions']}
    resolved.append((nodes[0],row,positions))
native_nodes=[]
for row in native:
    nodes=root.xpath(row['delimiterXPath'],namespaces=NS)
    assert len(nodes)==1;native_nodes.append(nodes[0])

split_count=0
for r,row,positions in resolved:
    value=row['text'];segments=[]
    for i,c in enumerate(value):
        kind=('fullwidth' if c in FULLWIDTH else 'math') if i in positions else 'other'
        if segments and segments[-1][0]==kind:segments[-1][1]+=c
        else:segments.append([kind,c])
    if len(segments)==1:
        assert segments[0][0]!='other'
        upright_bracket(r,segments[0][0]=='fullwidth')
        continue
    parent=r.getparent();index=parent.index(r);parent.remove(r);split_count+=1
    assert len(r.findall(M+'t'))==1
    for offset,(kind,value) in enumerate(segments):
        new=deepcopy(r);t=new.find(M+'t');t.text=value
        if value[:1].isspace() or value[-1:].isspace():t.set(SPACE,'preserve')
        if kind!='other':upright_bracket(new,kind=='fullwidth')
        parent.insert(index+offset,new)

for delimiter in native_nodes:
    pr=delimiter.find(M+'dPr')
    if pr is None:pr=E.Element(M+'dPr');delimiter.insert(0,pr)
    ctrl=pr.find(M+'ctrlPr')
    if ctrl is None:ctrl=E.SubElement(pr,M+'ctrlPr')
    prw=ctrl.find(W+'rPr')
    if prw is None:prw=E.SubElement(ctrl,W+'rPr')
    roman_word_properties(prw,True)

math_runs=list(root.iter(M+'r'))
for r in math_runs:onoff(wpr(r),'noProof',True)
controls=list(root.iter(M+'ctrlPr'))
for ctrl in controls:
    pr=ctrl.find(W+'rPr')
    if pr is None:pr=E.SubElement(ctrl,W+'rPr')
    onoff(pr,'noProof',True)

# Detect lost math, edits outside equations, or unintended changes to variables.
spec=importlib.util.spec_from_file_location('canon',A.parent/'exact_library_v2_20261016/prepare_math_verification.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
before_math=list(original.iter(M+'oMath'));after_math=list(root.iter(M+'oMath'))
assert len(before_math)==len(after_math)==4738
assert [text(n) for n in before_math]==[text(n) for n in after_math]
assert all(h.content([a])==h.content([b]) for a,b in zip(before_math,after_math))
brackets={p['char'] for r in inventory for p in r['positions']}
def nonbracket_formats(node):
    out=[]
    for r in node.iter(M+'r'):
        mr=r.find(M+'rPr');wr=deepcopy(r.find(W+'rPr'))
        if wr is not None:
            for n in list(wr.findall(W+'noProof')):wr.remove(n)
            if len(wr)==0 and not wr.attrib:wr=None
        props=(E.tostring(mr,method='c14n') if mr is not None else b'',E.tostring(wr,method='c14n') if wr is not None else b'')
        for c in text(r):
            if c not in brackets:out.append((c,props))
    return out
assert nonbracket_formats(original)==nonbracket_formats(root)
def outside_math(node):
    n=deepcopy(node)
    for math in n.iter(M+'oMath'):math.clear()
    return E.tostring(n,method='c14n')
assert outside_math(original)==outside_math(root)
assert all(r.find(W+'rPr/'+W+'noProof').get(W+'val')=='1' for r in math_runs)
assert LIVE.read_bytes()==SRC.read_bytes(),'Latest source changed; refresh snapshot before delivery.'
assert not DEST.exists(),f'Preserve existing output: {DEST}'
with ZipFile(SRC) as zin,ZipFile(DEST,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        zout.writestr(item,E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True) if item.filename=='word/document.xml' else zin.read(item.filename))
with ZipFile(SRC) as a,ZipFile(DEST) as b:
    assert a.namelist()==b.namelist()
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert changed==['word/document.xml'],changed
report={'source':str(LIVE),'source_snapshot':str(SRC),'output':str(DEST),
        'math_objects':4738,'literal_brackets_upright':4242,'math_font_brackets':4238,
        'fullwidth_Chinese_brackets_preserved':4,'native_delimiter_boundaries':576,
        'native_delimiter_objects':296,'mixed_runs_split':split_count,
        'math_runs_with_noProof':len(math_runs),'math_controls_with_noProof':len(controls),
        'math_text_and_structures_unchanged':True,'nonbracket_math_formats_unchanged':True,
        'all_outside_math_XML_unchanged':True,'original_preserved':True,
        'modified_package_parts':changed,'native_Word_verified':False}
(A/'repair_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
