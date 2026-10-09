from pathlib import Path
from lxml import etree as E
from collections import Counter
import zipfile,re,json
W=Path(__file__).parent;OUT=W.parents[1]/'排版完成_标题整合与摘要澄清_20261006.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'};WN='{'+N['w']+'}'
def load(path):
 z=zipfile.ZipFile(path);r=E.fromstring(z.read('word/document.xml'));return z,r,r.find('w:body',N)
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=N))
def style(p):
 n=p.find('w:pPr/w:pStyle',N);return n.get(WN+'val','') if n is not None else ''
bz,br,bb=load(W/'修改前.docx');ez,er,eb=load(W/'目录更新前.docx');fz,fr,fb=load(OUT)
# Detect accidental edits to protected titles/abstracts.
protected={'chapter','bibliographytitle','abstractcntitle','abstractentitle','symboltitle'}
assert [(style(p),text(p)) for p in bb if style(p) in protected]==[(style(p),text(p)) for p in fb if style(p) in protected]
babs=[text(p) for p in bb if style(p).startswith('abstract') or style(p).startswith('keyword')]
fabs=[text(p) for p in fb if style(p).startswith('abstract') or style(p).startswith('keyword')]
assert len(babs)==len(fabs)
abstract_changed=[i for i,(a,b) in enumerate(zip(babs,fabs)) if a!=b]
assert len(abstract_changed)==1 and babs[abstract_changed[0]].startswith('（4）')
# Word's TOC refresh must not drop or rewrite manuscript content.
exclude={'TOC1','TOC2','TOC3','TOC4'}
a=[text(p) for p in eb if style(p) not in exclude and text(p)]
b=[text(p) for p in fb if style(p) not in exclude and text(p)]
assert a==b,'Word changed non-TOC manuscript text'
# Native mathematical content is compared independent of pagination cache/rsid.
def math_form(n):
 def walk(e):
  if e.tag==WN+'lastRenderedPageBreak':return None
  attrs=tuple(sorted((k,v) for k,v in e.attrib.items() if not k.startswith(WN+'rsid')))
  children=tuple(c for x in e if (c:=walk(x)) is not None)
  return e.tag,attrs,e.text or '',children
 return walk(n)
assert Counter(math_form(x) for x in br.xpath('//m:oMath',namespaces=N))==Counter(math_form(x) for x in fr.xpath('//m:oMath',namespaces=N))
bm={n:bz.read(n) for n in bz.namelist() if n.startswith('word/media/')};fm={n:fz.read(n) for n in fz.namelist() if n.startswith('word/media/')}
assert bm==fm,'Image media changed'
assert len(fr.xpath('//w:drawing',namespaces=N))==len(br.xpath('//w:drawing',namespaces=N))==63
# Section references in tables may change; experimental cell values must not.
def tab_content(root):
 tables=[]
 for t in root.xpath('//w:tbl',namespaces=N):
  if t.xpath('.//m:oMath',namespaces=N):continue
  cells=[]
  for c in t.xpath('./w:tr/w:tc',namespaces=N):
   v=text(c)
   v=re.sub(r'第[1-7]\.\d[\d.第至到、和及与]*节','SECTION',v)
   v=re.sub(r'(?<![\w.])[A-F]\.\d+(?:\.\d+)?','APPENDIX',v)
   cells.append(v)
  tables.append(cells)
 return tables
assert tab_content(br)==tab_content(fr),'Table data changed'
# Check that every field/hyperlink target still exists after heading mergers.
bookmarks=set(fr.xpath('//w:bookmarkStart/@w:name',namespaces=N))
missing=[]
for code in fr.xpath('//w:instrText/text()',namespaces=N):
 m=re.match(r'\s*(?:REF|PAGEREF)\s+([^\s\\]+)',code)
 if m and m.group(1) not in bookmarks:missing.append(m.group(1))
for anchor in fr.xpath('//w:hyperlink/@w:anchor',namespaces=N):
 if anchor not in bookmarks:missing.append(anchor)
assert not missing,missing
assert not fr.xpath('//w:ins|//w:del',namespaces=N)
assert len(fr.xpath('//w:commentRangeStart',namespaces=N))==5
# Ensure all numbered section references resolve and no fourth-level heading survives.
heading_text=[text(p) for p in fb if style(p) in ('h2','h3')]
valid={t.split()[0] for t in heading_text}
assert len(valid)==len(heading_text)
assert not [p for p in fb if style(p)=='h4']
invalid=[]
for p in fb:
 if style(p) in exclude:continue
 for m in re.finditer(r'第[1-7]\.\d[\d.第至到、和及与]*节',text(p)):
  nums=re.findall(r'[1-7]\.\d+(?:\.\d+)*',m.group())
  if any(v not in valid for v in nums):invalid.append(m.group())
assert not invalid,invalid
# TOC entries must display each retained heading with a page field.
toc=[text(p) for p in fb if style(p) in exclude]
for t in heading_text:
 assert sum(v.startswith(t+'\t') or v.startswith(t) and v[len(t):].strip().isdigit() for v in toc)==1,('TOC heading',t)
assert not [p for p in fb if style(p)=='TOC4']
bib=[text(p) for p in fb if style(p)=='bibliography'];assert len(bib)==159
assert [text(p) for p in bb if style(p)=='bibliography']==bib[:157]
assert '2610.05143v1' in bib[-2] and 'github.com/erzhu419/CS-BAPR' in bib[-1]
report={'first_level_titles_unchanged':True,'changed_abstract_paragraphs':1,'remaining_fourth_level_headings':0,'level2_and_3_headings':len(heading_text),'native_equations_unchanged':len(fr.xpath('//m:oMath',namespaces=N)),'images_unchanged':len(fm),'non_equation_tables_unchanged':len(tab_content(fr)),'author_comments_retained':5,'bibliography_entries':len(bib),'all_references_resolve':True,'non_TOC_text_preserved_by_Word':True}
(W/'核对结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
