"""Synchronize only the supplied Markdown changes into the native Word copy."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from difflib import SequenceMatcher
from functools import lru_cache
from collections import defaultdict,deque,Counter
import json,re,copy,unicodedata
A=Path(__file__).resolve().parent;R=A.parents[1]
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}';P='{http://schemas.microsoft.com/office/word/2010/wordml}'
def text(e):return ''.join(n.text or ''for n in e.iter()if n.tag in(W+'t',M+'t'))
def norm(s):return ''.join(c for c in unicodedata.normalize('NFKC',s).replace('−','-').replace('′',"'").replace('ℓ','l').replace('ϵ','ε').replace('ς','σ')if not c.isspace()and c!='\u200b')
def read(p):
 with ZipFile(p)as z:return E.fromstring(z.read('word/document.xml'))
def core(r):return[e for e in r.find(W+'body')if norm(text(e))and not text(e).startswith('DISPLAY_EQUATION_')]
old=core(read(A/'word_sync_before_native.docx'));new=core(read(A/'word_sync_after_native.docx'))
mp={int(k):v for k,v in json.loads((A/'word_sync_mapping.json').read_text())['mapping'].items()}
with ZipFile(A/'修改前.docx')as z:parts={n:z.read(n)for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find(W+'body');blocks=list(body)
def mathkey(e):
 if e.tag in(W+'rPr',M+'ctrlPr',M+'argPr'):return None
 content=(e.text or '').replace('−','-').replace('′',"'").replace('ϵ','ε').replace('ς','σ')
 children=tuple(k for ch in e if(k:=mathkey(ch))is not None)
 return(e.tag,tuple(sorted(e.attrib.items())),content,children)
def signature(e):
 if e.tag==W+'pPr' or e.tag in(W+'tcPr',W+'tblPr',W+'tblGrid'):return None
 attrs=tuple(sorted((k,v)for k,v in e.attrib.items()if k not in(P+'paraId',P+'textId')))
 return(e.tag,attrs,e.text or '',tuple(k for ch in e if(k:=signature(ch))is not None))
def category(e):
 if e.tag==W+'tbl':return'table'
 st=e.find(W+'pPr/'+W+'pStyle');v=st.get(W+'val','')if st is not None else''
 if v.lower().startswith('heading'):return v
 m=re.match(r'([图表])\d+-\d+\s',text(e))
 return m[1]if m else'p'
pairs=[];deleted=[];inserted=[]
for mode,a,b,c,d in SequenceMatcher(None,[norm(text(e))for e in old],[norm(text(e))for e in new],autojunk=False).get_opcodes():
 if mode=='equal':pairs.extend(zip(range(a,b),range(c,d)));continue
 @lru_cache(None)
 def align(i,j):
  if i==b:return(0,())
  if j==d:return(0,())
  best=max(align(i+1,j),align(i,j+1),key=lambda x:x[0])
  if category(old[i])==category(new[j]):
   rest,tail=align(i+1,j+1);ratio=SequenceMatcher(None,norm(text(old[i])),norm(text(new[j])),autojunk=False).ratio()
   if ratio+rest>best[0]:best=(ratio+rest,((i,j),)+tail)
  return best
 _,matched=align(a,c);pairs.extend(matched)
 deleted.extend(i for i in range(a,b)if i not in {x for x,_ in matched})
 inserted.extend(j for j in range(c,d)if j not in {y for _,y in matched})
jobs=[(i,j)for i,j in pairs if signature(old[i])!=signature(new[j])]
FONT={W+x for x in('rFonts','sz','szCs','lang','kern')}
EMPH={W+x for x in('b','bCs','i','iCs')}
def base_rpr(p):
 candidates=[r.find(W+'rPr')for r in p.iter(W+'r')if r.find(W+'t')is not None and (r.find(W+'t').text or '').strip()]
 source=next((r for r in candidates if r is not None),None)
 result=E.Element(W+'rPr')
 if source is not None:
  for x in source:
   if x.tag in FONT:result.append(copy.deepcopy(x))
 for tag in (W+'sz',W+'szCs'):
  if result.find(tag)is None:
   x=p.find(W+'pPr/'+W+'rPr/'+tag)
   if x is not None:result.append(copy.deepcopy(x))
 return result
def merge_font(run,base):
 rp=run.find(W+'rPr')
 if rp is None:rp=E.Element(W+'rPr');run.insert(0,rp)
 for x in base:
  previous=rp.find(x.tag)
  if previous is not None:rp.remove(previous)
  rp.append(copy.deepcopy(x))
def reuse_emphasis(original,donor):
 oo=list(original.iter(M+'r'));dd=list(donor.iter(M+'r'));assert len(oo)==len(dd)
 for o,d in zip(oo,dd):
  rp=o.find(W+'rPr');dp=d.find(W+'rPr')
  if rp is None:rp=E.Element(W+'rPr');o.insert(0,rp)
  for x in list(rp):
   if x.tag in EMPH:rp.remove(x)
  if dp is not None:
   for x in dp:
    if x.tag in EMPH:rp.append(copy.deepcopy(x))
def locate_offsets(p):
 offsets=[];position=0
 for n in p.iter():
  if n.tag in(W+'bookmarkStart',W+'bookmarkEnd'):offsets.append((position,copy.deepcopy(n)))
  if n.tag in(W+'t',M+'t'):position+=len(n.text or '')
 return offsets
def map_offset(pos,oldtext,newtext):
 if pos==0:return 0
 if pos==len(oldtext):return len(newtext)
 for mode,a,b,c,d in SequenceMatcher(None,oldtext,newtext,autojunk=False).get_opcodes():
  if a<=pos<=b:
   if mode=='equal':return c+pos-a
   if pos==a:return c
   if pos==b:return d
   return c+round((pos-a)*(d-c)/(b-a))
 raise AssertionError(pos)
def insert_marker(p,pos,marker):
 offset=0
 for k,ch in enumerate(p):
  if ch.tag==W+'pPr':continue
  size=len(text(ch))
  if pos==offset:p.insert(k,marker);return
  if offset<pos<offset+size:
   assert ch.tag==W+'r' and len(ch.findall(W+'t'))==1,('Bookmark within mathematical/link object',pos,text(p))
   t=ch.find(W+'t');left=pos-offset;right=copy.deepcopy(ch);t.text=(t.text or '')[:left];right.find(W+'t').text=(right.find(W+'t').text or '')[left:]
   p.insert(k+1,marker);p.insert(k+2,right);return
  offset+=size
 if pos==offset:p.append(marker);return
 raise AssertionError((pos,offset,text(p)))
def sync_p(p,donor):
 assert not list(p.iter(W+'instrText')),('Unexpected field in changed paragraph',text(p))
 before=text(p);marks=locate_offsets(p);base=base_rpr(p);pool=defaultdict(deque)
 for math in p.iter(M+'oMath'):pool[mathkey(math)].append(copy.deepcopy(math))
 q=copy.deepcopy(donor)
 for math in list(q.iter(M+'oMath')):
  queue=pool[mathkey(math)]
  if queue:
   replacement=queue.popleft();reuse_emphasis(replacement,math);math.getparent().replace(math,replacement)
  else:
   for run in math.iter(M+'r'):
    font=copy.deepcopy(base);rf=font.find(W+'rFonts')
    if rf is not None:rf.set(W+'ascii','Cambria Math');rf.set(W+'hAnsi','Cambria Math')
    merge_font(run,font)
 for run in q.iter(W+'r'):merge_font(run,base)
 for child in list(p):
  if child.tag!=W+'pPr':p.remove(child)
 for child in q:
  if child.tag!=W+'pPr':p.append(copy.deepcopy(child))
 after=text(p)
 for pos,mark in reversed(marks):insert_marker(p,map_offset(pos,before,after),mark)
 assert norm(text(p))==norm(text(donor))
def patch_plain(p,donor):
 oldtext=text(p);newtext=text(donor);assert not list(donor.iter(M+'oMath'))
 # Static figure labels and their bookmark ranges are untouched.
 label=re.match(r'(?:图|表)\d+-\d+',oldtext)
 for mode,a,b,c,d in reversed(SequenceMatcher(None,oldtext,newtext,autojunk=False).get_opcodes()):
  if mode=='equal':continue
  if label:assert a>=label.end()
  nodes=[n for n in p.iter(W+'t')];off=0;placed=False
  for n in nodes:
   value=n.text or '';lo=max(0,a-off);hi=min(len(value),b-off)
   if lo<hi:n.text=value[:lo]+(newtext[c:d]if not placed else'')+value[hi:];placed=True
   elif a==b and not placed and off<=a<=off+len(value):k=a-off;n.text=value[:k]+newtext[c:d]+value[k:];placed=True
   off+=len(value)
  assert placed
 assert norm(text(p))==norm(newtext)
def resize_66(table):
 rows=table.findall(W+'tr');assert len(rows)==5 and all(len(r.findall(W+'tc'))==7 for r in rows)
 table.remove(rows[3]);grid=table.find(W+'tblGrid');cols=grid.findall(W+'gridCol');total=sum(int(c.get(W+'w'))for c in cols)
 first=int(cols[0].get(W+'w'))
 for k in (2,1):grid.remove(cols[k])
 numeric=round((total-first)/4);widths=[first,numeric,numeric,numeric,total-first-3*numeric]
 for col,width in zip(grid,widths):col.set(W+'w',str(width))
 for row in table.findall(W+'tr'):
  cells=row.findall(W+'tc')
  for k in (2,1):row.remove(cells[k])
  for cell,width in zip(row.findall(W+'tc'),widths):cell.find(W+'tcPr/'+W+'tcW').set(W+'w',str(width))
targets={j:blocks[mp[i]]for i,j in pairs}
journal=[]
for i,j in jobs:
 original=blocks[mp[i]];donor=new[j]
 if original.tag==W+'p':
  if re.match(r'^图\d+-\d+\s',text(original))and not list(donor.iter(M+'oMath')):patch_plain(original,donor)
  else:sync_p(original,donor)
 else:
  if i==754:resize_66(original)
  if i==735:
   rows=original.findall(W+'tr');assert len(rows)==5
   original.remove(rows[4])
  rr=original.findall(W+'tr');dr=donor.findall(W+'tr');assert len(rr)==len(dr),(i,j,mp[i],len(rr),len(dr),text(original)[:120])
  sr=old[i].findall(W+'tr')
  if i==754:sr=[sr[k]for k in(0,1,2,4)]
  if i==735:sr=sr[:4]
  for row,drow,srow in zip(rr,dr,sr):
   cc=row.findall(W+'tc');dc=drow.findall(W+'tc');assert len(cc)==len(dc)
   sc=srow.findall(W+'tc')
   if i==754:sc=[sc[k]for k in(0,3,4,5,6)]
   for cell,dcell,scell in zip(cc,dc,sc):
    pp=cell.findall(W+'p');dd=dcell.findall(W+'p');assert len(pp)==len(dd)==1
    if signature(scell.find(W+'p'))!=signature(dd[0]):sync_p(pp[0],dd[0])
 assert norm(text(original))==norm(text(donor))
 journal.append({'old_source_block':i,'new_source_block':j,'word_block':mp[i],'kind':'table'if original.tag==W+'tbl'else'paragraph'})
for i in deleted:
 e=blocks[mp[i]];assert not list(e.iter(W+'bookmarkStart'));body.remove(e)
for j in inserted:
 previous=targets[j-1]
 notes=[(abs(k-mp[735]),e)for k,e in enumerate(blocks)if e.tag==W+'p'and text(e).lstrip().startswith('注：')]
 template=min(notes,key=lambda x:x[0])[1]if notes else blocks[mp[736]]
 p=copy.deepcopy(template)
 for attr in(P+'paraId',P+'textId'):p.attrib.pop(attr,None)
 for marker in list(p.iter(W+'bookmarkStart'))+list(p.iter(W+'bookmarkEnd')):marker.getparent().remove(marker)
 sync_p(p,new[j]);previous.addnext(p);targets[j]=p
# Exactly the two authorized policy-parameter substitutions in native equation 2-27.
equation=next(e for e in body if e.tag==W+'tbl'and text(e).rstrip().endswith('(2-27)'))
changed=0
for t in equation.iter(M+'t'):
 if t.text in('φ','ϕ'):t.text='θ';changed+=1
assert changed==2,changed
# The chapter title is also the running-header title, with identical supplied wording.
oldtitle='跨保真运行环境下基于迁移强化学习的公交车队驻站控制'
chapter_title=re.findall(r'^#\s+第5章\s+([^\n]+?)\s*$',(R/'重构论文_替换库修订_20261015.md').read_text(),re.M)
assert len(chapter_title)==1,chapter_title
newtitle=chapter_title[0]
headers=0
for name,data in list(parts.items()):
 if not re.fullmatch(r'word/header\d+\.xml',name):continue
 h=E.fromstring(data)
 for t in h.iter(W+'t'):
  if oldtitle in(t.text or''):t.text=t.text.replace(oldtitle,newtitle);headers+=1
 parts[name]=E.tostring(h,encoding='UTF-8',xml_declaration=True,standalone=True)
assert all(norm(text(targets[j]))==norm(text(new[j]))for j in range(len(new)))
parts['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
out=R/'重构论文_替换库修订_20261015.docx'
with ZipFile(out,'w',ZIP_DEFLATED)as z:
 for name,data in parts.items():z.writestr(name,data)
report={'output':str(out),'old_source_blocks':len(old),'new_source_blocks':len(new),'changed_blocks':len(jobs),'deleted_source_blocks':deleted,'inserted_source_blocks':inserted,'equation_2_27_phi_to_theta':changed,'chapter5_headers_changed':headers,'source_match':True,'journal':journal}
(A/'word_sync_build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(A/'word_sync_final_core_positions.json').write_text(json.dumps({j:list(body).index(e)for j,e in targets.items()},indent=2))
print(json.dumps({k:v for k,v in report.items()if k!='journal'},ensure_ascii=False))
