from pathlib import Path
from lxml import etree as E
from collections import Counter
import zipfile,json,re,sys
W=Path(__file__).parent;ROOT=W.parents[1];OUT=ROOT/'排版完成_概括性行文与仿真章节整合_20261006.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'};WN='{'+N['w']+'}'
def load(p):
 z=zipfile.ZipFile(p);r=E.fromstring(z.read('word/document.xml'));return z,r,r.find('w:body',N)
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=N))
def style(p):
 s=p.find('w:pPr/w:pStyle',N);return s.get(WN+'val','') if s is not None else ''
def mf(n):
 if n.tag==WN+'lastRenderedPageBreak':return None
 return n.tag,tuple(sorted((k,v) for k,v in n.attrib.items() if not k.startswith(WN+'rsid'))),n.text or '',tuple(v for c in n if (v:=mf(c)) is not None)
def fields(body):
 result=[]
 for p in body:
  if style(p).startswith('TOC'):continue
  stack=[]
  for n in p.iter():
   if n.tag==WN+'fldChar':
    kind=n.get(WN+'fldCharType')
    if kind=='begin':stack.append({'code':'','result':'','sep':False})
    elif stack and kind=='separate':stack[-1]['sep']=True
    elif stack and kind=='end':
     f=stack.pop()
     if re.match(r'\s*(REF|PAGEREF|SEQ)\b',f['code']):result.append((f['code'],f['result']))
   elif stack and n.tag==WN+'instrText':stack[-1]['code']+=n.text or ''
   elif stack and stack[-1]['sep'] and n.tag==WN+'t':stack[-1]['result']+=n.text or ''
 return Counter(result)

bz,br,bb=load(W/'修改前.docx');ez,er,eb=load(W/'目录更新前.docx');fz,fr,fb=load(OUT)
assert [text(p) for p in bb if style(p).startswith(('abstract','keyword'))]==[text(p) for p in fb if style(p).startswith(('abstract','keyword'))]
protected={'chapter','bibliographytitle','symboltitle'}
assert [(style(p),text(p)) for p in bb if style(p) in protected]==[(style(p),text(p)) for p in fb if style(p) in protected]
assert [text(p) for p in bb if style(p)=='bibliography']==[text(p) for p in fb if style(p)=='bibliography']
assert [(style(p),text(p)) for p in bb if style(p) in ('h2','h3') and not text(p).startswith('2.')]==[(style(p),text(p)) for p in fb if style(p) in ('h2','h3') and not text(p).startswith('2.')]
assert not [p for p in fb if style(p)=='h4']
# Moving formula tables may change their order and displayed equation numbers.
def table_content(r):
 def normalize(t):
  t=re.sub(r'\(2-\d+\)','(chapter2-equation)',text(t))
  return t.replace('不同离线学习骨干','不同离线学习框架').replace('参照骨干','参照策略').replace('相对骨干/秒','相对基线/秒')
 return Counter(normalize(t) for t in r.xpath('//w:tbl',namespaces=N))
assert table_content(br)==table_content(fr)
assert Counter(mf(n) for n in br.xpath('//m:oMath',namespaces=N))==Counter(mf(n) for n in fr.xpath('//m:oMath',namespaces=N))
assert {n:bz.read(n) for n in bz.namelist() if n.startswith('word/media/')}=={n:fz.read(n) for n in fz.namelist() if n.startswith('word/media/')}
assert [mf(n) for n in br.xpath('//w:sectPr',namespaces=N)]==[mf(n) for n in fr.xpath('//w:sectPr',namespaces=N)]
assert Counter(re.findall(r'\[\d+(?:[,，]\d+)*\]',''.join(text(p) for p in bb)))==Counter(re.findall(r'\[\d+(?:[,，]\d+)*\]',''.join(text(p) for p in fb)))
assert not fr.xpath('//w:ins|//w:del',namespaces=N)
def comment_anchors(z,r):
 comments=E.fromstring(z.read('word/comments.xml'))
 labels={c.get(WN+'id'):(c.get(WN+'author'),text(c)) for c in comments}
 active={};result=[]
 for n in r.iter():
  if n.tag==WN+'commentRangeStart':active[n.get(WN+'id')]=[]
  elif n.tag in (WN+'t','{'+N['m']+'}t'):
   for parts in active.values():parts.append(n.text or '')
  elif n.tag==WN+'commentRangeEnd':
   ident=n.get(WN+'id');result.append((labels[ident],''.join(active.pop(ident))))
 assert not active
 return Counter(result)
# Word renumbers a comment ID when removing headings; author, text and anchor
# content, rather than that internal number, must stay intact.
assert comment_anchors(bz,br)==comment_anchors(fz,fr)
bookmarks=set(fr.xpath('//w:bookmarkStart/@w:name',namespaces=N))
for code in fr.xpath('//w:instrText/text()',namespaces=N):
 m=re.match(r'\s*(?:REF|PAGEREF)\s+([^\s\\]+)',code)
 if m:assert m.group(1) in bookmarks,m.group(1)
starts={p.get(WN+'id'):j for j,p in enumerate(fr.iter()) if p.tag==WN+'bookmarkStart'}
endids=set()
for j,p in enumerate(fr.iter()):
 if p.tag==WN+'bookmarkEnd':
  ident=p.get(WN+'id');endids.add(ident)
  assert ident in starts and starts[ident]<=j,ident
assert set(starts)==endids
assert fields(eb)==fields(fb),(fields(eb)-fields(fb),fields(fb)-fields(eb))
assert [text(p) for p in eb if not style(p).startswith('TOC') and text(p)]==[text(p) for p in fb if not style(p).startswith('TOC') and text(p)]
# Equation labels must increase with their new physical order.
eq2=[]
for p in fb:
 for (code,value),count in fields([p]).items():
  if re.match(r'\s*SEQ\s+Eq2\b',code):eq2.extend([int(value)]*count)
assert eq2==list(range(1,34)),eq2
heads=[text(p) for p in fb if style(p) in ('h2','h3')]
toc=[text(p) for p in fb if style(p).startswith('TOC')]
for h in heads:assert sum(v.startswith(h) and v[len(h):].strip().isdigit() for v in toc)==1,h
assert not any(re.match(r'2\.(?:5\.|6\b)',h) for h in heads)
assert not any('Error! Reference source not found' in text(p) or '错误!未找到引用源' in text(p) for p in fb)
report={'正文与附录行文已通读':True,'正文改写段落':95,'新增导读':5,'表内术语统一':3,
 '第二章结构':[h for h in heads if h.startswith('2.')],
 '摘要与一级标题不变':True,'第二章以外标题不变':True,'实验表格数值不变':True,
 '原生公式保留':len(fr.xpath('//m:oMath',namespaces=N)),
 '插图保留':len(fr.xpath('//w:drawing',namespaces=N)),
 '参考文献条数':len([p for p in fb if style(p)=='bibliography']),
 '正文引文保留':True,'第二章公式顺序正确':True,'交叉引用有效':True,
 '分节设置不变':True,'批注保留':len(fr.xpath('//w:commentRangeStart',namespaces=N)),
 '目录已更新':True,'本轮未重跑实验':True}
(W/'核对结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
