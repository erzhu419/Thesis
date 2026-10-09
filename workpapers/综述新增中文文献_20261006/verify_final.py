from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
import json,re
P=Path(__file__).parent
OUT=P.parent.parent/'排版完成_综述补充50篇中文文献_20261006.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W='{'+NS['w']+'}'
def read(path):
 z=ZipFile(path);r=E.fromstring(z.read('word/document.xml'));return z,r,r.find('w:body',NS)
def txt(p):return ''.join(p.xpath('.//w:t/text()',namespaces=NS))
def style(p):
 s=p.find('w:pPr/w:pStyle',NS);return s.get(W+'val','') if s is not None else ''
def signature(n):
 if n.tag==W+'lastRenderedPageBreak':return None
 return n.tag,tuple(sorted((k,v) for k,v in n.attrib.items() if not k.startswith(W+'rsid'))),n.text or '',tuple(v for c in n if (v:=signature(c)) is not None)
before,br,bb=read(P/'修改前.docx');after,ar,ab=read(OUT)
refs=json.loads((P/'references.json').read_text());groups=json.loads((P/'insertions.json').read_text())
assert len(ab)==len(bb)+50
new_start=next(i for i,p in enumerate(ab) if txt(p).startswith('[160] '))
removed={g['paragraph']:g['insert_text'] for g in groups}
for i,p in enumerate(bb):
 n=ab[i if i<new_start else i+50]
 assert style(p)==style(n),(i,'style')
 if style(p).startswith('TOC'):continue
 assert txt(n).replace(removed[i],'',1)==txt(p) if i in removed else txt(n)==txt(p),(i,'text')
assert [signature(n) for n in br.xpath('//m:oMath',namespaces=NS)]==[signature(n) for n in ar.xpath('//m:oMath',namespaces=NS)]
assert [signature(n) for n in br.xpath('//w:sectPr',namespaces=NS)]==[signature(n) for n in ar.xpath('//w:sectPr',namespaces=NS)]
assert {n:before.read(n) for n in before.namelist() if n.startswith('word/media/')}=={n:after.read(n) for n in after.namelist() if n.startswith('word/media/')}
assert Counter(txt(t) for t in br.xpath('//w:tbl',namespaces=NS))==Counter(txt(t) for t in ar.xpath('//w:tbl',namespaces=NS))
assert not ar.xpath('//w:ins|//w:del',namespaces=NS)
def comments(z,r):
 if 'word/comments.xml' not in z.namelist():
  assert not r.findall('.//w:commentRangeStart',NS)
  return Counter()
 c=E.fromstring(z.read('word/comments.xml'))
 names={n.get(W+'id'):(n.get(W+'author'),txt(n)) for n in c}
 active={};result=[]
 for n in r.iter():
  if n.tag==W+'commentRangeStart':active[n.get(W+'id')]=[]
  elif n.tag in (W+'t','{'+NS['m']+'}t'):
   for v in active.values():v.append(n.text or '')
  elif n.tag==W+'commentRangeEnd':
   i=n.get(W+'id');result.append((names[i],''.join(active.pop(i))))
 assert not active
 return Counter(result)
assert comments(before,br)==comments(after,ar)
bookmarks=set(ar.xpath('//w:bookmarkStart/@w:name',namespaces=NS))
for d in refs:
 nodes=ar.xpath('//w:hyperlink[@w:anchor=$a]',namespaces=NS,a=d['bookmark'])
 assert len(nodes)==1 and txt(nodes[0])==str(d['number']) and d['bookmark'] in bookmarks
 assert all(n.get(W+'val')=='superscript' for n in nodes[0].xpath('.//w:vertAlign',namespaces=NS))
bibliography=[txt(p) for p in ab if style(p)=='bibliography']
assert len(bibliography)==209
assert [int(re.match(r'\[(\d+)\]',s).group(1)) for s in bibliography]==list(range(1,210))
assert sum(bool(re.search('[\u4e00-\u9fff]',s)) for s in bibliography)==54
assert [txt(p) for p in bb if style(p)=='bibliography']==bibliography[:159]
errors=[txt(p) for p in ab if 'Error! Reference source not found' in txt(p) or '错误!未找到引用源' in txt(p)]
assert not errors,errors
info=json.loads((P/'word_info.json').read_text())
assert info['Status']=='pagination_not_completed'
# The interrupted native-render attempt only opened a temporary Windows copy.
# Check that no unrelated package part was changed in the actual deliverable.
assert set(before.namelist())==set(after.namelist())
for name in before.namelist():
 if name not in {'word/document.xml','word/_rels/document.xml.rels'}:assert before.read(name)==after.read(name),name
report={'新增中文文献':50,'发表年份':'2022—2025','新增引用位置':len(groups),'文献总数':209,'中文文献总数':54,'1.2原文逐字保留':True,'摘要及其他正文保留':True,'原有159条参考文献保留':True,'新增引文链接全部有效':True,'公式保留':len(ar.xpath('//m:oMath',namespaces=NS)),'图像及表格内容保留':True,'批注保留':len(comments(after,ar)),'原样式分节及其他文件组件保留':True,'目录页码已更新':False,'页面显示核对已完成':False,'未完成项':'Word自动重分页和样张导出未返回，目录页码及页面显示尚未核对。'}
(P/'最终核对结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
