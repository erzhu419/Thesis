from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
import json,re
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OUT=ROOT/'排版完成_接受作者修订_全文统一_20261006.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W='{'+NS['w']+'}'
def load(path):
 z=ZipFile(path);return z,E.fromstring(z.read('word/document.xml'))
def txt(n):return ''.join(n.xpath('.//w:t/text()|.//m:t/text()',namespaces=NS))
def style(n):
 p=n.find('w:pPr/w:pStyle',NS);return p.get(W+'val') if p is not None else ''
def math_sig(n):
 return (n.tag,tuple(sorted((k,v) for k,v in n.attrib.items() if not k.startswith(W+'rsid'))),n.text or '',tuple(math_sig(c) for c in n if c.tag != W+"lastRenderedPageBreak"))
a,old=load(WORK/'仅接受作者修订.docx');b,new=load(OUT)
oldbody=old.find('w:body',NS);body=new.find('w:body',NS)
oldabstract=[txt(n) for n in oldbody if style(n).startswith('abstract') or style(n).startswith('keyword')]
newabstract=[txt(n) for n in body if style(n).startswith('abstract') or style(n).startswith('keyword')]
assert oldabstract==newabstract,'Abstract/keyword changed'
# Compare every level-1 heading, using both style and effective outline definition.
styles=E.fromstring(a.read('word/styles.xml'));styledefs={s.get(W+'styleId'):s for s in styles}
def level1(n):
 sid=style(n);ppr=n.find('w:pPr',NS)
 if ppr is not None and ppr.find('w:outlineLvl',NS) is not None:return ppr.find('w:outlineLvl',NS).get(W+'val')=='0'
 for _ in range(8):
  ss=styledefs.get(sid)
  if ss is None:return False
  lev=ss.find('w:pPr/w:outlineLvl',NS)
  if lev is not None:return lev.get(W+'val')=='0'
  base=ss.find('w:basedOn',NS)
  if base is None:return False
  sid=base.get(W+'val')
 return False
h1_old=[txt(n) for n in oldbody if level1(n)];h1_new=[txt(n) for n in body if level1(n)]
assert h1_old==h1_new,(h1_old,h1_new)
mo=Counter(math_sig(n) for n in old.xpath('//m:oMath',namespaces=NS));mn=Counter(math_sig(n) for n in new.xpath('//m:oMath',namespaces=NS))
assert mo==mn,('Math changed',len(mo-mn),len(mn-mo))
assert Counter(a.read(n) for n in a.namelist() if n.startswith('word/media/'))==Counter(b.read(n) for n in b.namelist() if n.startswith('word/media/')),'Images changed'
assert not new.xpath('//w:ins|//w:del|//w:moveFrom|//w:moveTo',namespaces=NS),'Revisions remain'
assert txt(E.fromstring(a.read('word/comments.xml')))==txt(E.fromstring(b.read('word/comments.xml'))),'Comments changed'
for label in ['commentRangeStart','commentRangeEnd','commentReference']:
 assert len(old.xpath('//w:'+label,namespaces=NS))==len(new.xpath('//w:'+label,namespaces=NS))==5,label
 assert set(n.get(W+'id') for n in new.xpath('//w:'+label,namespaces=NS))==set(n.get(W+'id') for n in E.fromstring(b.read('word/comments.xml'))),label
# All measured table cell contents and math remain unchanged, including appendix tables.
def measured_tables(root):
 out=[]
 for n in root.find('w:body',NS):
  if n.tag!=W+'tbl':continue
  prev=n.getprevious();caption=txt(prev) if prev is not None else ''
  if re.match(r'表[1-7A-F]-',caption):out.append((caption,[txt(c) for c in n.xpath('./w:tr/w:tc',namespaces=NS)]))
 return out
assert measured_tables(old)==measured_tables(new),'Experimental table values changed'
# Bibliography text and all figure captions must remain intact.
def captions(root):return [txt(n) for n in root.find('w:body',NS) if style(n)=='figurecaption']
assert captions(old)==captions(new),'Figure caption changed'
def bibliography(root):
 nodes=list(root.find('w:body',NS));start=next(i for i,n in enumerate(nodes) if txt(n)=='参考文献' and not style(n).startswith('TOC'))
 end=next((i for i in range(start+1,len(nodes)) if txt(nodes[i]).startswith('附录A') and level1(nodes[i])),len(nodes))
 return [txt(n) for n in nodes[start:end]]
assert bibliography(old)==bibliography(new),'Bibliography changed'
# Sequence labels must be consecutive after relocating Eq. (2-8) to Chapter 3.
labels={2:[],3:[]}
for n in new.xpath('//w:tbl',namespaces=NS):
 texts=n.xpath('.//w:instrText/text()',namespaces=NS)
 for chap in (2,3):
  if any(re.search(r'SEQ\s+Eq'+str(chap)+r'\b',t) for t in texts):
   tail=txt(n.xpath('./w:tr/w:tc',namespaces=NS)[-1]);m=re.search(r'\('+str(chap)+r'-(\d+)\)',tail);assert m,(chap,tail);labels[chap].append(int(m.group(1)))
assert labels[2]==list(range(1,34)),labels[2]
assert labels[3]==list(range(1,43)),labels[3]
text='\n'.join(txt(n) for n in body)
assert not re.search(r'错误[!！]|Error!|未定义书签|未找到引用源',text),'Broken field found'
assert '2.2.2 公交环境的信息化与算法接口统一' in text
assert '3.2.3 类别特征、连续状态与共享策略' in text
# User-facing abstract exclusion remains part of the persistent TOC field.
toc=[n.text for n in new.xpath('//w:instrText',namespaces=NS) if 'TOC ' in (n.text or '')]
assert len(toc)==1 and '\\b ThesisMainText' in toc[0],toc
result={'accepted_insertions':17,'accepted_deletions':14,'remaining_revisions':0,'protected_abstract_paragraphs':len(oldabstract),'unchanged_first_level_headings':len(h1_old),'unchanged_native_math':sum(mo.values()),'unchanged_media_files':len([n for n in a.namelist() if n.startswith('word/media/')]),'unchanged_data_tables':len(measured_tables(old)),'retained_comments':5,'chapter2_equation_numbers':'1–33','chapter3_equation_numbers':'1–42','persistent_toc_scope':toc[0]}
(WORK/'内容核对.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
