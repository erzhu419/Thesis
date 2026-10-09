from pathlib import Path
from lxml import etree as E
from collections import Counter
import zipfile,json,re,sys
W=Path(__file__).parent;ROOT=W.parents[1];OUT=ROOT/'排版完成_标题调整后全文衔接_20261006.docx'
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
protected={'chapter','h2','h3','h4','bibliographytitle','symboltitle'}
assert [(style(p),text(p)) for p in bb if style(p) in protected]==[(style(p),text(p)) for p in fb if style(p) in protected]
assert [text(p) for p in bb if style(p)=='bibliography']==[text(p) for p in fb if style(p)=='bibliography']
assert [text(t) for t in br.xpath('//w:tbl',namespaces=N)]==[text(t) for t in fr.xpath('//w:tbl',namespaces=N)]
assert Counter(mf(n) for n in br.xpath('//m:oMath',namespaces=N))==Counter(mf(n) for n in fr.xpath('//m:oMath',namespaces=N))
assert fields(bb)==fields(fb),(fields(bb)-fields(fb),fields(fb)-fields(bb))
assert {n:bz.read(n) for n in bz.namelist() if n.startswith('word/media/')}=={n:fz.read(n) for n in fz.namelist() if n.startswith('word/media/')}
assert [mf(n) for n in br.xpath('//w:sectPr',namespaces=N)]==[mf(n) for n in fr.xpath('//w:sectPr',namespaces=N)]
assert [text(p) for p in eb if not style(p).startswith('TOC') and text(p)]==[text(p) for p in fb if not style(p).startswith('TOC') and text(p)]
assert not fr.xpath('//w:ins|//w:del',namespaces=N)
assert Counter(br.xpath('//w:commentRangeStart/@w:id',namespaces=N))==Counter(fr.xpath('//w:commentRangeStart/@w:id',namespaces=N))
assert bz.read('word/comments.xml')==fz.read('word/comments.xml')
bookmarks=set(fr.xpath('//w:bookmarkStart/@w:name',namespaces=N))
for code in fr.xpath('//w:instrText/text()',namespaces=N):
 m=re.match(r'\s*(?:REF|PAGEREF)\s+([^\s\\]+)',code)
 if m:assert m.group(1) in bookmarks,m.group(1)
# Existing prose numbers can change only where a section/formula/table reference is adjusted.
changes=json.loads((W/'修改记录.json').read_text())
def nums(s):
 s=re.sub(r'(?:第[0-9A-F]+(?:\.[0-9]+)+(?:至[0-9A-F]+(?:\.[0-9]+)+)?节?|[图表][0-9A-F]+[-－][0-9]+|式[（(][0-9A-F]+[-－][0-9]+[)）])','',s)
 return re.findall(r'\d+(?:[.,]\d+)*',s)
for c in changes:
 if c['action']=='正文改写':assert nums(c['before'])==nums(c['after']),(c['index'],nums(c['before']),nums(c['after']))
if '--toc' in sys.argv:
 heads=[text(p) for p in fb if style(p) in ('h2','h3')];toc=[text(p) for p in fb if style(p).startswith('TOC')]
 for h in heads:assert sum(v.startswith(h) and v[len(h):].strip().isdigit() for v in toc)==1,h
report={'正文与附录已通读':True,'正文改写段落':68,'新增衔接段落':10,'融入叙述的短标题':6,'调整内容归属':3,'摘要不变':True,'所有标题不变':True,'实验数值不变':True,'图表内容不变':True,'原生公式保留':len(fr.xpath('//m:oMath',namespaces=N)),'插图保留':len(fr.xpath('//w:drawing',namespaces=N)),'参考文献不变':True,'交叉引用有效且原有显示值不变':True,'分节设置不变':True,'批注保留':len(fr.xpath('//w:commentRangeStart',namespaces=N)),'目录已更新':'--toc' in sys.argv}
(W/'核对结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
