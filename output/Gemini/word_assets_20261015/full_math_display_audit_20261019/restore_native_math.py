"""Restore native equation typography from the corresponding pre-formatting equations.
Keep every current character and mathematical/layout element; only change math typography.
"""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import json,difflib,importlib.util
A=Path(__file__).resolve().parent
OUT=A.parents[1]
SRC=A/'source_snapshot.docx'
DONOR=Path('/mnt/d/Paper/Thesis/Gemini/重构论文_导言小结同步_20261018.docx')
DEST=OUT/'调整完格式的_全公式显示校订_20261019.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
SPACE='{http://www.w3.org/XML/1998/namespace}space'
STYLE={M+'nor',M+'scr',M+'sty'}
TYPO={W+'rFonts',W+'b',W+'bCs',W+'i',W+'iCs'}
BRACKETS=set('()[]{}⟨⟩⌊⌋⌈⌉')
FULLWIDTH=set('（）［］｛｝')
def text(n):return ''.join(t.text or '' for t in n.iter(M+'t'))
def prop(n):return (n.tag,tuple(sorted(n.attrib.items())))
def make(p):
    n=E.Element(p[0]);n.attrib.update(dict(p[1]));return n

def donor_characters(math):
    out=[]
    for run in math.iter(M+'r'):
        mp=run.find(M+'rPr');wp=run.find(W+'rPr')
        style=(tuple(prop(n) for n in mp if n.tag in STYLE) if mp is not None else (),
               tuple(prop(n) for n in wp if n.tag in TYPO and n.tag!=W+'rFonts') if wp is not None else ())
        out.extend((c,style) for c in text(run))
    return out
with ZipFile(SRC) as z:original=E.fromstring(z.read('word/document.xml'))
with ZipFile(DONOR) as z:donor=E.fromstring(z.read('word/document.xml'))
root=deepcopy(original)
current=list(root.iter(M+'oMath'));old=list(donor.iter(M+'oMath'))
assert len(current)==4738 and len(old)==4728
mapping=json.loads((A/'native_D_format_mapping.json').read_text())
char_styles={}
for i,j in mapping['mapping']:
    seq=donor_characters(old[j-1]);assert text(current[i-1])==''.join(c for c,_ in seq)
    char_styles[i]=[s for c,s in seq]
for block in mapping['non_equal_blocks']:
    ids=block['current_indices']
    if not ids:continue
    chars=[item for j in block['donor_indices'] for item in donor_characters(old[j-1])]
    a=''.join(text(current[i-1]) for i in ids);b=''.join(c for c,_ in chars)
    styles=[]
    for kind,i1,i2,j1,j2 in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        # Current differences are only preserved line splits or already-authorized deletions.
        assert kind in ('equal','insert'),(ids,kind,a[i1:i2],b[j1:j2])
        if kind=='equal':styles.extend(s for c,s in chars[j1:j2])
    assert len(styles)==len(a)
    pos=0
    for i in ids:
        n=len(text(current[i-1]));char_styles[i]=styles[pos:pos+n];pos+=n
assert set(char_styles)==set(range(1,4739))

def upright(s):
    return (((M+'scr',((M+'val','roman'),)),(M+'sty',((M+'val','p'),))),())
def typography(run,style,segment):
    mp=run.find(M+'rPr')
    if mp is None:mp=E.Element(M+'rPr');run.insert(0,mp)
    for n in list(mp):
        if n.tag in STYLE:mp.remove(n)
    for p in style[0]:mp.append(make(p))
    if len(mp)==0:run.remove(mp)
    wp=run.find(W+'rPr')
    if wp is None:wp=E.Element(W+'rPr');run.insert(1 if run.find(M+'rPr') is not None else 0,wp)
    for n in list(wp):
        if n.tag in TYPO:wp.remove(n)
    f=E.Element(W+'rFonts')
    cjk=any('\u3000'<=c<='\u9fff' or '\uff00'<=c<='\uffef' for c in segment)
    for slot in ('ascii','hAnsi','cs'):f.set(W+slot,'Cambria Math')
    f.set(W+'eastAsia','宋体' if cjk else 'Cambria Math');wp.insert(0,f)
    for p in style[1]:wp.append(make(p))
    np=wp.find(W+'noProof')
    if np is None:np=E.SubElement(wp,W+'noProof')
    np.set(W+'val','1')

splits=0;modified=0;brackets=0;normal_units=0
for ordinal,math in enumerate(current,1):
    styles=char_styles[ordinal];pos=0
    for run in list(math.iter(M+'r')):
        value=text(run);these=styles[pos:pos+len(value)];pos+=len(value)
        assert len(run.findall(M+'t'))<=1
        if not value:
            typography(run,(((M+'sty',((M+'val','p'),)),),()),'');continue
        segments=[]
        for char,s in zip(value,these):
            if char in BRACKETS:s=upright(s);brackets+=1
            if char in FULLWIDTH:
                s=(((M+'nor',()),),());brackets+=1
            if segments and segments[-1][1]==s:segments[-1][0]+=char
            else:segments.append([char,s])
        if len(segments)==1:
            typography(run,segments[0][1],value);modified+=1;continue
        parent=run.getparent();index=parent.index(run);parent.remove(run);splits+=1
        for offset,(segment,style) in enumerate(segments):
            new=deepcopy(run);t=new.find(M+'t');t.text=segment
            if segment[:1].isspace() or segment[-1:].isspace():t.set(SPACE,'preserve')
            # A cached rendered-page boundary belongs only before the original first character.
            if offset:
                for cache in list(new.findall(W+'lastRenderedPageBreak')):new.remove(cache)
            typography(new,style,segment);parent.insert(index+offset,new);modified+=1
    assert pos==len(styles)
# All math control glyphs, including the delimiters, use the actual mathematics font.
for ctrl in root.iter(M+'ctrlPr'):
    wp=ctrl.find(W+'rPr')
    if wp is None:wp=E.SubElement(ctrl,W+'rPr')
    f=wp.find(W+'rFonts')
    if f is None:f=E.Element(W+'rFonts');wp.insert(0,f)
    f.attrib.clear()
    for slot in ('ascii','hAnsi','eastAsia','cs'):f.set(W+slot,'Cambria Math')
    for name,val in [('i','0'),('iCs','0'),('noProof','1')]:
        n=wp.find(W+name)
        if n is None:n=E.SubElement(wp,W+name)
        n.set(W+'val',val)
# Concrete preservation gates: text, actual mathematical structure, all non-math XML.
spec=importlib.util.spec_from_file_location('canon',A.parent/'exact_library_v2_20261016/prepare_math_verification.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
before=list(original.iter(M+'oMath'));after=list(root.iter(M+'oMath'))
assert len(after)==4738
assert [text(m) for m in before]==[text(m) for m in after]
assert all(h.content([a])==h.content([b]) for a,b in zip(before,after))
def outside(r):
    r=deepcopy(r)
    for m in r.iter(M+'oMath'):m.clear()
    return E.tostring(r,method='c14n')
assert outside(original)==outside(root)
assert not DEST.exists()
with ZipFile(SRC) as zin,ZipFile(DEST,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():zout.writestr(item,E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True) if item.filename=='word/document.xml' else zin.read(item.filename))
report={'source':str(SRC),'donor_typography_only':str(DONOR),'output':str(DEST),'math_objects':4738,'math_characters_matched_to_native_donor':sum(len(s) for s in char_styles.values()),'same_text_objects':4692,'line_split_or_prior_deletion_objects':46,'current_math_text_and_structure_unchanged':True,'all_non_math_XML_identical':True,'run_segments_formatted':modified,'runs_split':splits,'literal_bracket_characters_upright':brackets,'native_math_runs':sum(r.find(M+'rPr/'+M+'nor') is None for r in root.iter(M+'r')),'normal_text_math_runs':sum(r.find(M+'rPr/'+M+'nor') is not None for r in root.iter(M+'r')),'actual_rendering_review_pending':True}
(A/'native_math_restore_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
