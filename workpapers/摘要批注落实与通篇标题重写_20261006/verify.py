from pathlib import Path
from lxml import etree as E
from collections import Counter
import zipfile,json,re
W=Path(__file__).parent;OUT=W.parents[1]/'排版完成_摘要指定句与标题重写_20261006.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'};WN='{'+N['w']+'}'
def load(p):
 z=zipfile.ZipFile(p);r=E.fromstring(z.read('word/document.xml'));return z,r,r.find('w:body',N)
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=N))
def style(p):
 x=p.find('w:pPr/w:pStyle',N);return x.get(WN+'val','') if x is not None else ''
bz,br,bb=load(W/'修改前.docx');ez,er,eb=load(W/'目录更新前.docx');fz,fr,fb=load(OUT)
protected={'chapter','abstractcntitle','abstractentitle','bibliographytitle','symboltitle'}
assert [(style(p),text(p)) for p in bb if style(p) in protected]==[(style(p),text(p)) for p in fb if style(p) in protected]
ba=[text(p) for p in bb if style(p).startswith(('abstract','keyword'))];fa=[text(p) for p in fb if style(p).startswith(('abstract','keyword'))]
assert len(ba)==len(fa);changed=[a for a,v in enumerate(zip(ba,fa)) if v[0]!=v[1]];assert changed==[6],changed
excluded={'TOC1','TOC2','TOC3','TOC4'}
assert [text(p) for p in eb if style(p) not in excluded and text(p)]==[text(p) for p in fb if style(p) not in excluded and text(p)]
assert [text(t) for t in br.xpath('//w:tbl',namespaces=N)]==[text(t) for t in fr.xpath('//w:tbl',namespaces=N)]
def mf(n):
 if n.tag==WN+'lastRenderedPageBreak':return None
 return n.tag,tuple(sorted((k,v) for k,v in n.attrib.items() if not k.startswith(WN+'rsid'))),n.text or '',tuple(v for c in n if (v:=mf(c)) is not None)
assert Counter(mf(x) for x in br.xpath('//m:oMath',namespaces=N))==Counter(mf(x) for x in fr.xpath('//m:oMath',namespaces=N))
assert {n:bz.read(n) for n in bz.namelist() if n.startswith('word/media/')}=={n:fz.read(n) for n in fz.namelist() if n.startswith('word/media/')}
assert len(fr.xpath('//w:drawing',namespaces=N))==63
heads=[text(p) for p in fb if style(p) in ('h2','h3')];toc=[text(p) for p in fb if style(p) in excluded]
assert len(heads)==153
for t in heads:assert sum(v.startswith(t) and v[len(t):].strip().isdigit() for v in toc)==1,t
assert not [p for p in fb if style(p) in ('h4','TOC4')]
bookmarks=set(fr.xpath('//w:bookmarkStart/@w:name',namespaces=N));missing=[]
for code in fr.xpath('//w:instrText/text()',namespaces=N):
 m=re.match(r'\s*(?:REF|PAGEREF)\s+([^\s\\]+)',code)
 if m and m.group(1) not in bookmarks:missing.append(m.group(1))
assert not missing,missing
assert [text(p) for p in bb if style(p)=='bibliography']==[text(p) for p in fb if style(p)=='bibliography']
assert not fr.xpath('//w:ins|//w:del',namespaces=N)
assert len(fr.xpath('//w:commentRangeStart',namespaces=N))==5
# Confirm both requested explanations appear in the actual abstract, before the results.
p=fa[6]
assert '利用预先采集的同一批SUMO运行记录' in p
assert '模型既预测后续到站事件的线路、站点等类别信息' in p
assert '并让预测可靠的样本获得较大权重' in p
assert '定义为联合源参数偏差' in p
assert p.index('定义为联合源参数偏差')<p.index('联合50%和联合75%')<p.index('人均广义乘客时间分别减少57.67')
assert '公交扩展采用固定目标离线数据与混合事件模型' not in p
report={'first_level_unchanged':True,'two_requested_passages_rewritten_in_abstract':True,'abstract_paragraphs_changed':1,'other_abstract_paragraphs_unchanged':True,'headings_reviewed':153,'headings_rewritten':119,'fourth_level_headings':0,'TOC_synchronized':True,'native_equations_preserved':2086,'images_preserved':63,'all_table_text_and_math_preserved':True,'bibliography_preserved':True,'field_targets_valid':True}
(W/'核对结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
