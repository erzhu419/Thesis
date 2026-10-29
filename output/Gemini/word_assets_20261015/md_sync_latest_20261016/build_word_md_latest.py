"""Apply the supplied complete Markdown only to matched native Word blocks."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from difflib import SequenceMatcher
from functools import lru_cache
from collections import defaultdict,deque,Counter
import json,re,copy,unicodedata
import native_helpers as h
A=Path(__file__).resolve().parent;R=A.parents[1]
W=h.W;M=h.M;P=h.P
REL='{http://schemas.openxmlformats.org/package/2006/relationships}'
C='{http://schemas.openxmlformats.org/package/2006/content-types}'
text=h.text

def norm(s):return h.norm(s).replace('ϕ','φ')
def read(p):return h.read(p)
def flow(r):return[e for e in r.find(W+'body')if norm(text(e))]
def eq_label(e):
 m=re.fullmatch(r'DISPLAY_EQUATION_(.+)',text(e));return m[1]if m else None

def category(e):
 if eq_label(e):return'eq:'+eq_label(e)
 if e.tag==W+'tbl':return'table'
 st=e.find(W+'pPr/'+W+'pStyle');v=st.get(W+'val','')if st is not None else''
 if v.lower().startswith('heading'):return v
 # Formal captions use a native caption style; prose beginning with a figure
 # reference is ordinary paragraph content and must remain available to merge.
 return'p'
old=flow(read(A/'word_sync_before_native.docx'));new=flow(read(A/'word_sync_after_native.docx'))
mp={int(k):v for k,v in json.loads((A/'word_sync_mapping.json').read_text())['mapping'].items()}
with ZipFile(A/'修改前.docx')as z:parts={n:z.read(n)for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find(W+'body');blocks=list(body)
original_xml={id(e):E.tostring(e)for e in blocks}
old_core_ord={};i=0
for k,e in enumerate(old):
 if not eq_label(e):old_core_ord[k]=i;i+=1
assert i==len(mp)
# Locate every numbered row, retaining the native table, widths and number cell.
eq_rows={}
for bi,block in enumerate(blocks):
 if block.tag!=W+'tbl':continue
 for row in block.findall(W+'tr'):
  cells=row.findall(W+'tc')
  if len(cells)<2:continue
  m=re.fullmatch(r'\s*[（(]([1-7A-F]-\d+[a-z]*)[）)]\s*',text(cells[-1]))
  if m:
   assert m[1]not in eq_rows
   eq_rows[m[1]]=(block,row,cells)
assert len(eq_rows)==376,len(eq_rows)
native={k:eq_rows[eq_label(e)][0]if eq_label(e)else blocks[mp[old_core_ord[k]]]for k,e in enumerate(old)}
# Display markers anchor independent alignment intervals, so changed prose
# cannot shift a native equation/image past another source block.
pairs=[];deleted=[];inserted=[];segments=[]
for mode,a,b,c,d in SequenceMatcher(None,[norm(text(e))for e in old],[norm(text(e))for e in new],autojunk=False).get_opcodes():
 if mode=='equal':pairs.extend(zip(range(a,b),range(c,d)));continue
 segments.append((a,b,c,d))
 @lru_cache(None)
 def align(i,j):
  if i==b or j==d:return(0,())
  best=max(align(i+1,j),align(i,j+1),key=lambda x:x[0])
  if category(old[i])==category(new[j])and not eq_label(old[i]):
   rest,tail=align(i+1,j+1)
   ratio=SequenceMatcher(None,norm(text(old[i])),norm(text(new[j])),autojunk=False).ratio()
   if ratio+rest>best[0]:best=(ratio+rest,((i,j),)+tail)
  return best
 _,matched=align(a,c);pairs.extend(matched)
 deleted.extend(i for i in range(a,b)if i not in{x for x,_ in matched})
 inserted.extend(j for j in range(c,d)if j not in{y for _,y in matched})
assert not any(eq_label(old[i])for i in deleted)
assert not any(eq_label(new[j])for j in inserted)
pairs.sort();targets={j:native[i]for i,j in pairs}
old_to_new=dict(pairs)
# New list identifiers must refer to numbering definitions in this package.
numbering=E.fromstring(parts['word/numbering.xml'])
with ZipFile(A/'word_sync_after_native.docx')as z:donor_numbering=E.fromstring(z.read('word/numbering.xml'))
num_map={};abs_map={}
def translate_num(p):
 np=p.find(W+'pPr/'+W+'numPr')
 if np is None:return None
 result=copy.deepcopy(np);nid=result.find(W+'numId');oldid=nid.get(W+'val')
 if oldid not in num_map:
  dn=next(n for n in donor_numbering.findall(W+'num')if n.get(W+'numId')==oldid)
  aid=dn.find(W+'abstractNumId').get(W+'val')
  if aid not in abs_map:
   newaid=str(max(int(n.get(W+'abstractNumId'))for n in numbering.findall(W+'abstractNum'))+1)
   da=next(n for n in donor_numbering.findall(W+'abstractNum')if n.get(W+'abstractNumId')==aid)
   da=copy.deepcopy(da);da.set(W+'abstractNumId',newaid);numbering.append(da);abs_map[aid]=newaid
  newnid=str(max(int(n.get(W+'numId'))for n in numbering.findall(W+'num'))+1)
  dn=copy.deepcopy(dn);dn.set(W+'numId',newnid);dn.find(W+'abstractNumId').set(W+'val',abs_map[aid]);numbering.append(dn);num_map[oldid]=newnid
 nid.set(W+'val',num_map[oldid]);return result

def fields(p):
 out=[];seq=[];depth=0
 for child in p:
  starts=sum(x.get(W+'fldCharType')=='begin'for x in child.iter(W+'fldChar'))
  ends=sum(x.get(W+'fldCharType')=='end'for x in child.iter(W+'fldChar'))
  if starts and depth==0:seq=[]
  if depth or starts:seq.append(copy.deepcopy(child))
  depth+=starts-ends
  if seq and depth==0:
   instr=''.join(t.text or''for c in seq for t in c.iter(W+'instrText'))
   assert instr.strip().startswith('REF '),('Unexpected changed field',instr,text(p))
   label=''.join(text(c)for c in seq).strip()
   if label:out.append((label,seq,instr))
   seq=[]
 assert depth==0
 return out

def split_run(ch,offset):
 assert ch.tag==W+'r'and len(ch.findall(W+'t'))==1,('Non-text boundary',text(ch),offset)
 t=ch.find(W+'t');value=t.text or'';left=copy.deepcopy(ch);right=copy.deepcopy(ch)
 left.find(W+'t').text=value[:offset];right.find(W+'t').text=value[offset:]
 for r in(left,right):r.find(W+'t').set('{http://www.w3.org/XML/1998/namespace}space','preserve')
 return left,right

def splice(p,a,b,replacement):
 # Split only ordinary donor text; math and hyperlink objects remain atomic.
 def boundary(pos):
  off=0
  for k,ch in enumerate(p):
   if ch.tag==W+'pPr':continue
   size=len(text(ch))
   if pos==off:return k
   if off<pos<off+size:
    left,right=split_run(ch,pos-off);p.replace(ch,left);p.insert(k+1,right);return k+1
   off+=size
  assert pos==off,(pos,off,text(p));return len(p)
 boundary(b);start=boundary(a);end=boundary(b)
 for ch in list(p)[start:end]:p.remove(ch)
 for k,ch in enumerate(replacement):p.insert(start+k,copy.deepcopy(ch))

field_log=[]
def keep_fields(q,original):
 matches=[];occupied=[];value=text(q)
 offset=0;begin=None
 for ch in q:
  if any(x.get(W+'fldCharType')=='begin'for x in ch.iter(W+'fldChar')):begin=offset
  offset+=len(text(ch))
  if begin is not None and any(x.get(W+'fldCharType')=='end'for x in ch.iter(W+'fldChar')):
   occupied.append((begin,offset));begin=None
 for label,seq,instr in fields(original):
  # Cached labels include incidental trailing spaces in the old Word. The
  # source controls those spaces, while the original field keeps its target.
  pattern=r'\s*'.join(map(re.escape,label.split()))
  found=[m for m in re.finditer(pattern,value)if not any(m.start()<b and a<m.end()for a,b in occupied)]
  if not found:
   field_log.append({'instruction':instr.strip(),'action':'removed_by_supplied_text','cached':label});continue
  m=found[0];occupied.append((m.start(),m.end()))
  desired=value[m.start():m.end()];ts=[n for c in seq for n in c.iter(W+'t')]
  assert ts
  ts[0].text=desired
  for t in ts[1:]:t.text=''
  matches.append((m.start(),m.end(),seq,instr))
 for a,b,seq,instr in reversed(sorted(matches,key=lambda x:x[0])):
  splice(q,a,b,seq);field_log.append({'instruction':instr.strip(),'action':'preserved'})

def sync_p(p,donor,*,inserted_p=False):
 before=text(p);marks=h.locate_offsets(p);base=h.base_rpr(p);pool=defaultdict(deque)
 for math in p.iter(M+'oMath'):pool[h.mathkey(math)].append(copy.deepcopy(math))
 q=copy.deepcopy(donor)
 # Remove donor bookmarks, since the authoritative targets are native Word's.
 for mark in list(q.iter(W+'bookmarkStart'))+list(q.iter(W+'bookmarkEnd')):mark.getparent().remove(mark)
 for math in list(q.iter(M+'oMath')):
  queue=pool[h.mathkey(math)]
  if queue:
   replacement=queue.popleft();h.reuse_emphasis(replacement,math);math.getparent().replace(math,replacement)
  else:
   for run in math.iter(M+'r'):
    font=copy.deepcopy(base);rf=font.find(W+'rFonts')
    if rf is not None:rf.set(W+'ascii','Cambria Math');rf.set(W+'hAnsi','Cambria Math')
    h.merge_font(run,font)
 for run in q.iter(W+'r'):h.merge_font(run,base)
 keep_fields(q,p)
 for child in list(p):
  if child.tag!=W+'pPr':p.remove(child)
 for child in q:
  if child.tag!=W+'pPr':p.append(copy.deepcopy(child))
 original_num=p.find(W+'pPr/'+W+'numPr')
 np=translate_num(donor)if original_num is None or inserted_p else None
 if inserted_p and donor.find(W+'pPr/'+W+'numPr')is None and original_num is not None:
  original_num.getparent().remove(original_num)
 if np is not None:
  pp=p.find(W+'pPr')
  if pp is None:pp=E.Element(W+'pPr');p.insert(0,pp)
  previous=pp.find(W+'numPr')
  if previous is not None:pp.remove(previous)
  pp.append(np)
 after=text(p)
 mark_targets={}
 starts={m.get(W+'id'):pos for pos,m in marks if m.tag==W+'bookmarkStart'}
 ends={m.get(W+'id'):pos for pos,m in marks if m.tag==W+'bookmarkEnd'}
 for mid,a in starts.items():
  if mid not in ends:continue
  b=ends[mid];label=before[a:b]
  if label and len(label)<=50 and label in after:
   at=after.index(label);mark_targets[(mid,'bookmarkStart')]=at;mark_targets[(mid,'bookmarkEnd')]=at+len(label)
 mapped=[]
 for pos,mark in marks:
  at=mark_targets.get((mark.get(W+'id'),E.QName(mark).localname),h.map_offset(pos,before,after))
  mapped.append((at,mark))
 for at,mark in reversed(mapped):h.insert_marker(p,at,mark)
 assert norm(text(p))==norm(text(donor)),('Paragraph source mismatch',before,text(p),text(donor))

journal=[];changed_native=set();ppr_checks=[]
for i,j in pairs:
 if eq_label(old[i])or h.signature(old[i])==h.signature(new[j]):continue
 original=native[i];donor=new[j];before_xml=E.tostring(original)
 if original.tag==W+'p':
  oldpr=E.tostring(original.find(W+'pPr'))if original.find(W+'pPr')is not None else None
  st=original.find(W+'pPr/'+W+'pStyle')
  formal=st is not None and st.get(W+'val')in('figurecaption','tablecaption')
  if formal and not list(donor.iter(M+'oMath')):
   marks=h.locate_offsets(original);before=h.text(original)
   h.patch_plain(original,donor);after=h.text(original)
   ranges={m.get(W+'id'):pos for pos,m in marks if m.tag==W+'bookmarkStart'}
   ends={m.get(W+'id'):pos for pos,m in marks if m.tag==W+'bookmarkEnd'}
   positions={}
   for mid,start in ranges.items():
    if mid not in ends:continue
    label=before[start:ends[mid]]
    if label and len(label)<=50 and label in after:
     at=after.index(label);positions[(mid,W+'bookmarkStart')]=at;positions[(mid,W+'bookmarkEnd')]=at+len(label)
   for m in list(original.iter(W+'bookmarkStart'))+list(original.iter(W+'bookmarkEnd')):m.getparent().remove(m)
   for pos,m in reversed(marks):h.insert_marker(original,positions.get((m.get(W+'id'),m.tag),h.map_offset(pos,before,after)),m)
  else:sync_p(original,donor)
  newpr=E.tostring(original.find(W+'pPr'))if original.find(W+'pPr')is not None else None
  ppr_checks.append({'old_flow':i,'new_flow':j,'unchanged':oldpr==newpr,'new_numbering':donor.find(W+'pPr/'+W+'numPr')is not None})
 else:
  rr=original.findall(W+'tr');dr=donor.findall(W+'tr');sr=old[i].findall(W+'tr')
  assert len(rr)==len(dr)==len(sr),('Unexpected table row count',i,j,len(rr),len(dr),len(sr))
  for row,drow,srow in zip(rr,dr,sr):
   cc=row.findall(W+'tc');dc=drow.findall(W+'tc');sc=srow.findall(W+'tc');assert len(cc)==len(dc)==len(sc)
   for cell,dcell,scell in zip(cc,dc,sc):
    pp=cell.findall(W+'p');dd=dcell.findall(W+'p');ss=scell.findall(W+'p');assert len(pp)==len(dd)==len(ss)==1
    if h.signature(ss[0])!=h.signature(dd[0]):sync_p(pp[0],dd[0])
 assert norm(text(original))==norm(text(donor)),('Source block mismatch',i,j)
 changed_native.add(id(original));journal.append({'old_flow':i,'new_flow':j,'word_index':blocks.index(original),'kind':'table'if original.tag==W+'tbl'else'paragraph'})
# Insert new paragraphs at their exact source anchors, never after an intervening
# equation table. All inserted material is supplied text; formatting is native.
for j in sorted(inserted):
 assert new[j].tag==W+'p'
 candidates=[(abs(k-j),k,e)for k,e in targets.items()if e.tag==W+'p'and not eq_label(new[k])and (e.find(W+'pPr/'+W+'pStyle')is None or e.find(W+'pPr/'+W+'pStyle').get(W+'val')=='body')]
 template=min(candidates,key=lambda x:x[:2])[2]
 p=copy.deepcopy(template)
 for attr in(P+'paraId',P+'textId'):p.attrib.pop(attr,None)
 for mark in list(p.iter(W+'bookmarkStart'))+list(p.iter(W+'bookmarkEnd')):mark.getparent().remove(mark)
 # Template fields and formulas are not content donors for the new paragraph.
 for child in list(p):
  if child.tag!=W+'pPr':p.remove(child)
 sync_p(p,new[j],inserted_p=True)
 next_key=min((k for k in targets if k>j),default=None)
 if next_key is not None:targets[next_key].addprevious(p)
 else:
  prev_key=max(k for k in targets if k<j);targets[prev_key].addnext(p)
 targets[j]=p;journal.append({'new_flow':j,'kind':'inserted_paragraph','template_word_index':blocks.index(template)if template in blocks else None})
# Preserve bookmarks spanning removed prose at the same section boundary.
# G's own bookmarks are intentionally removed with the deleted appendix.
# A supplied merge can move a reference out of its original paragraph. Carry
# those original fields across the same changed source interval as well.
for a,b,c,d in segments:
 source_fields=[]
 for i in range(a,b):
  if eq_label(old[i])or native[i].tag!=W+'p':continue
  source_fields.extend(fields(E.fromstring(original_xml[id(native[i])])))
 used=Counter(instr for j in range(c,d)if j in targets and targets[j].tag==W+'p'for label,seq,instr in fields(targets[j]))
 consumed=Counter()
 for label,seq,instr in source_fields:
  consumed[instr]+=1
  if used[instr]>=consumed[instr]:continue
  for j in range(c,d):
   if j not in targets or eq_label(new[j])or targets[j].tag!=W+'p':continue
   p=targets[j];before_count=len(fields(p))
   fake=E.Element(W+'p')
   for ch in seq:fake.append(copy.deepcopy(ch))
   keep_fields(p,fake)
   if len(fields(p))>before_count:
    used[instr]+=1;changed_native.add(id(p));break
relocated=[]
for i in sorted(deleted):
 e=native[i];assert e.tag==W+'p'
 g=False
 marks=[copy.deepcopy(n)for n in e if n.tag in(W+'bookmarkStart',W+'bookmarkEnd')]
 if marks and not g:
  later=[(k,native[k])for k in old_to_new if k>i and native[k].tag==W+'p']
  earlier=[(k,native[k])for k in old_to_new if k<i and native[k].tag==W+'p']
  if later:
   dest=min(later,key=lambda x:x[0])[1]
   for mark in reversed(marks):h.insert_marker(dest,0,mark)
  else:
   dest=max(earlier,key=lambda x:x[0])[1]
   for mark in marks:dest.append(mark)
  relocated.extend({'id':m.get(W+'id'),'name':m.get(W+'name'),'from_old_flow':i}for m in marks)
 if e.getparent()is body:body.remove(e)
# Changed display OMML donors were independently prepared from the exact new MD.
def md_eqs(path):
 return[(re.search(r'\\tag\{([^}]+)\}',m[1])[1],m[1].strip())for m in re.finditer(r'\$\$(.*?)\$\$',path.read_text(),re.S)]
old_eq=md_eqs(A/'上次MD.md');new_eq=md_eqs(A/'最新MD.md')
assert len(old_eq)==len(new_eq)==376
assert [a for a,b in old_eq]==[a for a,b in new_eq]
math_donors={}
for path in(A/'math_changed_donor.docx',):
 current=None
 for p in read(path).find(W+'body'):
  m=re.fullmatch(r'EXPECT_SLOT_(\d+)',text(p))
  if m:current=int(m[1]);math_donors[current]=[]
  elif current is not None:math_donors[current].extend(copy.deepcopy(m)for m in p.iter(M+'oMath'))
changed_equations=[]
def native_alignment(math):
 for t in list(math.iter(M+'t')):
  if t.text!='&':continue
  assert any(a.tag==M+'eqArr'for a in t.iterancestors()),('Unexpected literal ampersand',text(math))
  run=t.getparent();pr=run.find(M+'rPr')
  if pr is None:pr=E.Element(M+'rPr');run.insert(0,pr)
  E.SubElement(pr,M+'aln');t.text='\u200b'
 return math
for ordinal,((label,prior),(label2,supply))in enumerate(zip(old_eq,new_eq),1):
 if re.sub(r'\s+','',prior)==re.sub(r'\s+','',supply):continue
 assert label==label2
 block,row,cells=eq_rows[label];oldmaths=[m for c in cells[:-1]for m in c.iter(M+'oMath')]
 assert oldmaths and math_donors.get(ordinal),(ordinal,label)
 # Every label row uses one paragraph; A3 and C9 contain two original oMaths.
 paragraph=oldmaths[0].getparent()
 if paragraph.tag==M+'oMathPara':mathpara=paragraph;p=mathpara.getparent()
 else:mathpara=None;p=paragraph
 assert p.tag==W+'p'
 donors=copy.deepcopy(math_donors[ordinal])
 # A-3 and C-9 have two native paragraphs under one number. The supplied
 # aligned expression has the same two rows; keep those native paragraphs.
 if len(oldmaths)==2:
  assert label in('A-3','C-9')and len(donors)==1
  arr=donors[0].find(M+'eqArr');assert arr is not None
  rows=arr.findall(M+'e');assert len(rows)==2
  donors=[]
  for erow in rows:
   m=E.Element(M+'oMath')
   for ch in erow:
    if ch.tag==M+'r'and text(ch)=='&':continue
    m.append(copy.deepcopy(ch))
   donors.append(m)
  for original,donor in zip(oldmaths,donors):
   for run in donor.iter(M+'r'):h.merge_font(run,h.base_rpr(original.getparent().getparent()))
   original.getparent().replace(original,donor)
  changed_native.add(id(block));changed_equations.append(label);continue
 assert all(m.getparent()is paragraph for m in oldmaths),(label,'multiple formula containers')
 base=h.base_rpr(p)
 if not len(base):
  source=next((r.find(W+'rPr')for m in oldmaths for r in m.iter(M+'r')if r.find(W+'rPr')is not None),None)
  if source is not None:
   for child in source:
    if child.tag in h.FONT:base.append(copy.deepcopy(child))
 for m in oldmaths:m.getparent().remove(m)
 if mathpara is None:
  mathpara=E.Element(M+'oMathPara');p.append(mathpara)
 for donor in donors:
  donor=native_alignment(donor)
  for run in donor.iter(M+'r'):h.merge_font(run,base)
  mathpara.append(donor)
 changed_native.add(id(block));changed_equations.append(label)
rels=E.fromstring(parts['word/_rels/document.xml.rels']);ct=E.fromstring(parts['[Content_Types].xml']);removed_parts=[];removed_toc=[]
f_break=None
# Explicit exact-source completeness check (lists have native numbering).
source_mismatches=[]
for j,e in enumerate(new):
 if eq_label(e):continue
 actual=targets[j]
 if actual.getparent()is not body or norm(text(actual))!=norm(text(e)):
  source_mismatches.append({'flow':j,'expected':text(e),'actual':text(actual)})
assert not source_mismatches,source_mismatches[:3]
positions=[list(body).index(targets[j])for j in range(len(new))]
assert positions==sorted(positions),('Source flow order mismatch',[(j,positions[j-1],positions[j])for j in range(1,len(positions))if positions[j]<positions[j-1]])
# Untouched body blocks (including images, table6-3 and bibliography) stay native.
removed_ids={id(native[i])for i in deleted}|set()
untouched_diffs=[]
for e in blocks:
 if id(e)in changed_native or id(e)in removed_ids or e.tag==W+'sectPr' or e.getparent()is not body:continue
 if E.tostring(e)!=original_xml[id(e)]:
  # Bookmark boundary relocation is the only permitted non-content operation.
  q=copy.deepcopy(e);o=E.fromstring(original_xml[id(e)])
  for x in(q,o):
   for m in list(x.iter(W+'bookmarkStart'))+list(x.iter(W+'bookmarkEnd')):m.getparent().remove(m)
  assert E.tostring(q)==E.tostring(o),('Unexpected unchanged-native difference',blocks.index(e),text(e)[:100])
# Native target inventory catches missing/duplicated live references before COM.
starts=list(root.iter(W+'bookmarkStart'));names=Counter(x.get(W+'name')for x in starts)
assert not[k for k,v in names.items()if v>1],[(k,v)for k,v in names.items()if v>1]
missing=[]
for el in root.iter():
 if el.tag==W+'hyperlink'and el.get(W+'anchor')and el.get(W+'anchor')not in names:missing.append(el.get(W+'anchor'))
 if el.tag==W+'instrText':
  m=re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)',el.text or'')
  if m and m[1]not in names:missing.append(m[1])
assert not missing,Counter(missing)
assert all(x['unchanged']or x['new_numbering']for x in ppr_checks),[x for x in ppr_checks if not x['unchanged']and not x['new_numbering']]
parts['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
parts['word/numbering.xml']=E.tostring(numbering,encoding='UTF-8',xml_declaration=True,standalone=True)
parts['word/_rels/document.xml.rels']=E.tostring(rels,encoding='UTF-8',xml_declaration=True,standalone=True)
parts['[Content_Types].xml']=E.tostring(ct,encoding='UTF-8',xml_declaration=True,standalone=True)
out=R/'重构论文_MD同步更新_20261016.docx'
with ZipFile(out,'w',ZIP_DEFLATED)as z:
 for name,data in parts.items():z.writestr(name,data)
report={'output':str(out),'old_core':sum(not eq_label(e)for e in old),'new_core':sum(not eq_label(e)for e in new),'source_match':True,'changed_native_blocks':len(changed_native),'changed_equations':changed_equations,'changed_equation_count':len(changed_equations),'inserted_paragraphs':inserted,'deleted_paragraphs':deleted,'numbered_list_definitions_added':num_map,'native_pPr_preserved':True,'relocated_bookmarks':relocated,'fields':field_log,'removed_G_parts':removed_parts,'removed_G_TOC_cache':removed_toc,'journal':journal}
(A/'word_sync_build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(A/'word_sync_final_flow_positions.json').write_text(json.dumps({j:list(body).index(e)for j,e in targets.items()},indent=2))
print(json.dumps({k:v for k,v in report.items()if k not in('journal','fields','relocated_bookmarks','changed_equations')},ensure_ascii=False))
