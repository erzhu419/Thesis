"""Apply the revised source to matching prose/table blocks in the preserved Word package."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from collections import defaultdict, deque
from difflib import SequenceMatcher
from functools import lru_cache
from lxml import etree as E
import json, re, pypandoc

A = Path(__file__).resolve().parent; ROOT = A.parents[1]
BASE = A/'修改前_重构论文_通篇校核_20261015.docx'
OUT = ROOT/'重构论文_语言修订_v2_20261015.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
NS = {'w': W[1:-1]}
nums = json.loads((A.parent/'reference_numbers.json').read_text())

def text(e): return ''.join(n.text or '' for n in e.iter() if n.tag in (W+'t', M+'t'))
def style(e):
    p = e.find('./'+W+'pPr/'+W+'pStyle')
    return p.get(W+'val') if p is not None else ''
def canon(e): return re.sub(r'\s|\u200b', '', text(e)).replace('−','-').replace('′',"'")
def math_key(e):
    if e.tag in (W+'rPr',M+'ctrlPr',M+'argPr'):return None
    return (e.tag,tuple(sorted(e.attrib.items())),(e.text or '').strip(),tuple(k for n in e if (k:=math_key(n)) is not None))
def read(path):
    with ZipFile(path) as z: return E.fromstring(z.read('word/document.xml'))
def core(root):
    return [e for e in root.find(W+'body') if canon(e) and not text(e).startswith('DISPLAY_EQUATION_')]

source = (ROOT/'重构论文_语言修订_v2_20261015.md').read_text()
pattern = re.compile(r'\[([A-Za-z]+\d{4}[\w-]*(?:\s*,\s*[A-Za-z]+\d{4}[\w-]*)*)\]')
def cite(m):
    return '^\\['+','.join('['+str(nums[k.strip()])+'](#latest_ref_'+str(nums[k.strip()])+')' for k in m[1].split(','))+'\\]^'
source = pattern.sub(cite, source)
source = re.sub(r'^!\[图[^\n]*$','',source,flags=re.M)
source = re.sub(r'\$\$(.*?)\$\$', lambda m: '\n\nDISPLAY_EQUATION_'+re.search(r'\\tag\{([^}]+)\}',m[1])[1]+'\n\n',source,flags=re.S)
source = re.sub(r'^---\s*$','',source,flags=re.M)
converted = A/'source_after_native.docx'
pypandoc.convert_text(source, 'docx', format='markdown+tex_math_dollars-smart', outputfile=str(converted))
oldsource, newsource = core(read(A/'source_before_native.docx')), core(read(converted))
mapping = {int(k):v for k,v in json.loads((A/'source_word_block_mapping.json').read_text())['mapping'].items()}
with ZipFile(BASE) as z: package = {n:z.read(n) for n in z.namelist()}
root = E.fromstring(package['word/document.xml']); body = root.find(W+'body'); oldblocks = list(body)
oldmath = defaultdict(deque)
for index, e in enumerate(oldsource):
    srcmath, wordmath = list(e.iter(M+'oMath')), list(oldblocks[mapping[index]].iter(M+'oMath'))
    if len(srcmath) == len(wordmath):
        for a,b in zip(srcmath,wordmath): oldmath[math_key(a)].append(deepcopy(b))

def clean(e):
    for n in list(e.iter()):
        if n.tag in (W+'bookmarkStart', W+'bookmarkEnd'): n.getparent().remove(n)
    for n in list(e.iter(M+'oMath')):
        if oldmath[math_key(n)]: n.getparent().replace(n, oldmath[math_key(n)].popleft())
    for link in e.iter(W+'hyperlink'):
        assert link.get(W+'anchor','').startswith('latest_ref_')
        for run in link.iter(W+'r'):
            pr = run.find(W+'rPr')
            if pr is None: pr=E.Element(W+'rPr');run.insert(0,pr)
            for tag, val in [('color','000000'),('u','none')]:
                child=pr.find(W+tag)
                if child is None: child=E.SubElement(pr,W+tag)
                child.set(W+'val',val)
    return e

def paragraph(new, old):
    p = deepcopy(old); props=deepcopy(old.find(W+'pPr'))
    markers = [deepcopy(n) for n in old if n.tag in (W+'bookmarkStart',W+'bookmarkEnd')]
    for n in list(p):p.remove(n)
    if props is not None:p.append(props)
    if new.find('./'+W+'pPr/'+W+'numPr') is None and props is not None:
        for n in list(props.findall(W+'numPr')):props.remove(n)
    for mark in markers:
        if mark.tag == W+'bookmarkStart':p.append(mark)
    for n in new:
        if n.tag != W+'pPr':p.append(deepcopy(n))
    for mark in markers:
        if mark.tag == W+'bookmarkEnd':p.append(mark)
    if p.find('.//'+M+'oMath') is not None:
        if props is None:props=E.Element(W+'pPr');p.insert(0,props)
        spacing=props.find(W+'spacing')
        if spacing is None:spacing=E.SubElement(props,W+'spacing')
        spacing.set(W+'lineRule','atLeast');spacing.set(W+'line','400')
    return p

def table(new, old):
    t=deepcopy(old);rows=t.findall(W+'tr');wanted=new.findall(W+'tr')
    assert len(rows[0].findall(W+'tc')) == len(wanted[0].findall(W+'tc'))
    while len(rows)<len(wanted):
        row=deepcopy(rows[-1]);t.append(row);rows.append(row)
    for row in rows[len(wanted):]:t.remove(row)
    for i,(row,wrow) in enumerate(zip(rows,wanted)):
        for cell,wcell in zip(row.findall(W+'tc'),wrow.findall(W+'tc')):
            template=cell.find(W+'p')
            for child in list(cell):
                if child.tag!=W+'tcPr':cell.remove(child)
            for wp in wcell.findall(W+'p'):
                p=paragraph(wp,template);cell.append(p)
                pr=p.find(W+'pPr')
                markpr=pr.find(W+'rPr')
                if markpr is None:markpr=E.SubElement(pr,W+'rPr')
                sz=markpr.find(W+'sz')
                if sz is None:sz=E.SubElement(markpr,W+'sz')
                sz.set(W+'val','21')
                for run in p.iter(W+'r'):
                    rp=run.find(W+'rPr')
                    if rp is None:rp=E.Element(W+'rPr');run.insert(0,rp)
                    sz=rp.find(W+'sz')
                    if sz is None:sz=E.SubElement(rp,W+'sz')
                    sz.set(W+'val','21')
    return t

changes=[]
opcodes=SequenceMatcher(None,[canon(e) for e in oldsource],[canon(e) for e in newsource],autojunk=False).get_opcodes()
for op,a,b,c,d in reversed(opcodes):
    if op=='equal':continue
    assert op!='insert','Unexpected insertion without an editable source anchor'
    originals=[oldblocks[mapping[i]] for i in range(a,b)]
    # Split revised prose at its existing Word position; leave intervening figures
    # and display equations at their original anchors.
    @lru_cache(None)
    def partition(i,j):
        if i==b:return (0,()) if j==d else (-1e9,())
        best=(-1e9,())
        for end in range(j+1,d-(b-i-1)+1):
            group=newsource[j:end]
            if any(n.tag!=oldsource[i].tag for n in group):break
            if oldsource[i].tag==W+'tbl' and len(group)>1:break
            score=SequenceMatcher(None,canon(oldsource[i]),''.join(canon(n) for n in group),autojunk=False).ratio()
            rest,groups=partition(i+1,end)
            if score+rest>best[0]:best=(score+rest,((i,j,end),)+groups)
        return best
    score,groups=partition(a,c)
    assert score>-1e8,(op,a,b,c,d)
    for i,j,end in groups:
        anchor=oldblocks[mapping[i]]
        for k,new in enumerate(newsource[j:end]):
            new=clean(deepcopy(new));template=anchor
            replacement=table(new,template) if new.tag==W+'tbl' else paragraph(new,template)
            if k:
                for marker in list(replacement.iter()):
                    if marker.tag in (W+'bookmarkStart',W+'bookmarkEnd'):marker.getparent().remove(marker)
            anchor.addprevious(replacement)
        body.remove(anchor)
    changes.append({'old_blocks':b-a,'new_blocks':d-c,'old_source_index':a,'new_source_index':c,'placements':groups})

# Remove ordinary mid-sentence/data emphasis, retain short paragraph headings and table headers.
def clear_bold(p, keep_prefix=''):
    offset=0
    for run in p:
        if run.tag!=W+'r':continue
        rt=''.join(run.xpath('.//w:t/text()',namespaces=NS));pr=run.find(W+'rPr')
        if pr is None:pr=E.Element(W+'rPr');run.insert(0,pr)
        n=pr.find(W+'b')
        if n is None:n=E.SubElement(pr,W+'b')
        n.set(W+'val','1' if keep_prefix and offset<len(keep_prefix) else '0')
        offset+=len(rt)
main=False;appendix_g=False
for block in body:
    if style(block)=='chapter' and text(block).startswith('第1章'):main=True
    if style(block)=='bibliographytitle':main=False
    if style(block)=='chapter' and text(block).startswith('附录G'):appendix_g=True
    editable=main or appendix_g or style(block)=='abstractcnbody'
    if not editable:continue
    if block.tag==W+'p' and style(block) in ('body','abstractcnbody'):
        value=text(block);match=re.match(r'^(?:（[一二三四五六七八九十0-9]+）|[A-F]\.\d+ ).{0,90}?[。．：]',value)
        clear_bold(block,match[0] if match else '')
    if block.tag==W+'tbl' and not re.search(r'\((?:[2-6]-\d+|附G-\d+)\)$',text(block)):
        for row in block.findall(W+'tr')[1:]:
            for p in row.iter(W+'p'):clear_bold(p)

def replace_plain(p,old,new):
    nodes=list(p.iter(W+'t'));value=''.join(n.text or '' for n in nodes)
    for match in reversed(list(re.finditer(re.escape(old),value))):
        start=match.start();end=match.end();offset=0;first=True
        for n in nodes:
            v=n.text or '';lo=max(start-offset,0);hi=min(end-offset,len(v))
            if lo<hi:n.text=v[:lo]+(new if first else '')+v[hi:];first=False
            offset+=len(v)
    return p
for p in root.iter(W+'p'):
    replace_plain(p,'robust value learning with two regularization terms','robust value learning with weight-norm and ensemble-disagreement regularization')
    replace_plain(p,'已出发及已完成行程的乘客数','已进入系统及已完成行程的乘客数')
    replace_plain(p,'已出发乘客','已进入系统的乘客')
    plain=''.join(n.text or '' for n in p.iter(W+'t'))
    if '跨保真' in plain:
        replace_plain(p,'跨保真度','CROSS_FIDELITY_DEGREE')
        replace_plain(p,'跨保真','跨保真度')
        replace_plain(p,'CROSS_FIDELITY_DEGREE','跨保真度')

# Keep the existing picture geometry and replace only the three edited line-diagram images.
rels=E.fromstring(package['word/_rels/document.xml.rels']);targets={n.get('Id'):n.get('Target') for n in rels}
for update in json.loads((A/'figure_language_updates.json').read_text()):
    caption,=[p for p in body if style(p)=='figurecaption' and text(p).startswith(update['figure']+' ')]
    pic=caption.getprevious();blip,=pic.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
    path='word/'+targets[blip.get(R+'embed')];package[path]=Path(update['png']).read_bytes()
for name in list(package):
    if not re.match(r'word/header\d+\.xml$',name):continue
    header=E.fromstring(package[name])
    for p in header.iter(W+'p'):
        replace_plain(p,'跨保真运行环境下基于迁移强化学习','基于跨保真度迁移强化学习')
        replace_plain(p,'跨保真度','CROSS_FIDELITY_DEGREE');replace_plain(p,'跨保真','跨保真度');replace_plain(p,'CROSS_FIDELITY_DEGREE','跨保真度')
    package[name]=E.tostring(header,encoding='UTF-8',xml_declaration=True,standalone=True)

names=[m.get(W+'name') for m in root.iter(W+'bookmarkStart')]
assert len(names)==len(set(names))
assert all(n.get(W+'anchor') in names for n in root.iter(W+'hyperlink') if n.get(W+'anchor'))
package['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for name,data in package.items():z.writestr(name,data)
(A/'word_revision_blocks.json').write_text(json.dumps(list(reversed(changes)),ensure_ascii=False,indent=2))
print(json.dumps({'output':str(OUT),'replacement_groups':len(changes),'source_before_blocks':len(oldsource),'source_after_blocks':len(newsource),'bibliography':226,'figure_text_updates':3},ensure_ascii=False))
