"""Keep native Word paragraphs and math; apply only the approved deletions and short diffs."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from difflib import SequenceMatcher
from functools import lru_cache
from lxml import etree as E
import json,re,unicodedata,pypandoc
from PIL import Image
A=Path(__file__).resolve().parent;R=A.parents[1];PRE=A.parent/'language_review_v2'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
def text(e):return ''.join(n.text or '' for n in e.iter() if n.tag in (W+'t',M+'t'))
def normalized(s):
 chars=[];positions=[]
 for i,char in enumerate(s):
  for c in unicodedata.normalize('NFKC',char).replace('−','-').replace('′',"'").replace('ℓ','l').replace('ϵ','ε').replace('ς','σ'):
   if not c.isspace() and c!='\u200b':chars.append(c);positions.append(i)
 return ''.join(chars),positions
def norm(e):return normalized(text(e))[0]
def parse_doc(p):
 with ZipFile(p) as z:return E.fromstring(z.read('word/document.xml'))
def source_core(root):return [e for e in root.find(W+'body') if norm(e) and not text(e).startswith('DISPLAY_EQUATION_')]
# Cites are kept linked to the existing bibliography, preserving its source identities.
nums=json.loads((A.parent/'reference_numbers.json').read_text())
s=(A/'source_before_renumber.md').read_text()
cites=re.compile(r'\[([A-Za-z]+\d{4}[\w-]*(?:\s*,\s*[A-Za-z]+\d{4}[\w-]*)*)\]')
s=cites.sub(lambda m:'^\\['+','.join('['+str(nums[k.strip()])+'](#latest_ref_'+str(nums[k.strip()])+')' for k in m[1].split(','))+'\\]^',s)
s=re.sub(r'^【图[^\n]*】[ \t]*$','',s,flags=re.M)
s=re.sub(r'\$\$(.*?)\$\$',lambda m:'\n\nDISPLAY_EQUATION_'+re.search(r'\\tag\{([^}]+)\}',m[1])[1]+'\n\n',s,flags=re.S)
s=re.sub(r'^---\s*$','',s,flags=re.M)
pypandoc.convert_text(s,'docx',format='markdown+tex_math_dollars-smart',outputfile=str(A/'source_cleanup_native.docx'))
oldsrc=source_core(parse_doc(PRE/'source_before_native.docx'));newsrc=source_core(parse_doc(A/'source_cleanup_native.docx'))
mapping={int(k):v for k,v in json.loads((PRE/'source_word_block_mapping.json').read_text())['mapping'].items()}
with ZipFile(A/'修改前_通篇校核版.docx') as z:parts={n:z.read(n) for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find(W+'body');oldblocks=list(body);journal=[]
def patch_range(e,start,end,replacement):
 nodes=[n for n in e.iter() if n.tag in (W+'t',M+'t')];offset=0;placed=False
 math_candidates=[n for n in e.iter() if n.tag.startswith(M) and n.tag!=M+'t' and text(n)]
 for n in nodes:
  v=n.text or '';lo=max(start-offset,0);hi=min(end-offset,len(v))
  if lo<hi:
   n.text=v[:lo]+(replacement if not placed else '')+v[hi:];placed=True
  elif start==end and not placed and offset<=start<=offset+len(v):
   p=start-offset;n.text=v[:p]+replacement+v[p:];placed=True
  offset+=len(v)
 assert placed,(start,end,replacement)
 # Remove only emptied semantic math structures; preserve every retained object.
 for n in reversed(math_candidates):
  if not text(n):
   parent=n.getparent()
   if parent is not None:parent.remove(n)
def literal(e,old,new,required=True):
 value=text(e);matches=list(re.finditer(re.escape(old),value))
 if required:assert matches,(old,value[:150])
 for m in reversed(matches):patch_range(e,m.start(),m.end(),new)
 return len(matches)
def style(e):
 p=e.find(W+'pPr/'+W+'pStyle');return p.get(W+'val') if p is not None else ''
# Each retained source paragraph keeps its original Word paragraph, formatting and boundaries.
for op,a,b,c,d in reversed(SequenceMatcher(None,[norm(e) for e in oldsrc],[norm(e) for e in newsrc],autojunk=False).get_opcodes()):
 if op=='equal':continue
 @lru_cache(None)
 def align(i,j):
  if j==d:return (0,())
  if i==b:return (-1e9,())
  best=align(i+1,j)
  if oldsrc[i].tag==newsrc[j].tag:
   ratio=SequenceMatcher(None,norm(oldsrc[i]),norm(newsrc[j]),autojunk=False).ratio()
   rest,pairs=align(i+1,j+1)
   if ratio+rest>best[0]:best=(ratio+rest,((i,j),)+pairs)
  return best
 score,pairs=align(a,c);assert score>-1e8,(op,a,b,c,d)
 paired={i:j for i,j in pairs}
 for i in range(a,b):
  original=oldblocks[mapping[i]]
  if i not in paired:
   if original.getparent() is body:body.remove(original)
   journal.append(dict(kind='deleted_block',index=mapping[i],old=text(original)))
   continue
  new=newsrc[paired[i]];old_text=text(oldsrc[i]);new_text=text(new)
  on,om=normalized(old_text);nn,nm=normalized(new_text);wn,wm=normalized(text(original))
  assert on==wn,('Word-source anchor mismatch',i,old_text[:120],text(original)[:120])
  # A changed table must keep its rows/cells; column deletion is handled explicitly below.
  if new.tag==W+'tbl':assert len(new.findall(W+'tr'))==len(original.findall(W+'tr'))
  diffs=SequenceMatcher(None,on,nn,autojunk=False).get_opcodes()
  for kind,lo,hi,ns,ne in reversed(diffs):
   if kind=='equal':continue
   begin=wm[lo] if lo<len(wm) else len(text(original));end=wm[hi-1]+1 if hi>lo else begin
   replacement=new_text[nm[ns]:nm[ne-1]+1] if ne>ns else ''
   old_fragment=text(original)[begin:end]
   patch_range(original,begin,end,replacement)
   journal.append(dict(kind='short_edit',index=mapping[i],old=old_fragment,new=replacement))
# Delete original images/equations between the removed experimental headings.
start=next(i for i,e in enumerate(oldblocks) if style(e)=='h3' and text(e).startswith('6.4.1 受控动力学基准对照实验'))
end=next(i for i,e in enumerate(oldblocks) if style(e)=='h3' and text(e).startswith('6.4.3 公交跨保真任务'))
for e in oldblocks[start:end]:
 if e.getparent() is body:body.remove(e)
deleted_eq=set(json.loads((A/'controlled_deletion_boundaries.json').read_text())['delete_equations'])
for e in list(body):
 if e.tag==W+'tbl' and any(text(e).rstrip().endswith('('+label+')') for label in deleted_eq):body.remove(e)
# Word-only appendix/symbol changes are short literal edits, never paragraph replacements.
extra=json.loads((A/'word_only_edits.json').read_text())
for edit in extra['edits']:
 scope=root
 if edit.get('index') is not None:scope=oldblocks[edit['index']]
 n=literal(scope,edit['old'],edit['new']);journal.append(dict(kind='word_only',old=edit['old'],new=edit['new'],count=n))
def remove_columns(table,columns):
 grid=table.find(W+'tblGrid');cols=grid.findall(W+'gridCol');total=sum(int(n.get(W+'w')) for n in cols)
 for i in reversed(columns):grid.remove(cols[i])
 remaining=grid.findall(W+'gridCol');weights=[int(n.get(W+'w')) for n in remaining]
 widths=[round(total*w/sum(weights)) for w in weights];widths[-1]+=total-sum(widths)
 for n,w in zip(remaining,widths):n.set(W+'w',str(w))
 for row in table.findall(W+'tr'):
  cells=row.findall(W+'tc')
  for i in reversed(columns):row.remove(cells[i])
  for cell,w in zip(row.findall(W+'tc'),widths):cell.find(W+'tcPr/'+W+'tcW').set(W+'w',str(w))
remove_columns(oldblocks[2141],[2]);remove_columns(oldblocks[2163],[3,4])
# Nviol is specific to the removed chapter-6 experiments.
symbols=oldblocks[42]
for row in list(symbols.findall(W+'tr')):
 if any(phrase in text(row) for phrase in ('受控动力学任务的平均违例次数','稀疏系数矩阵、经尺度变换的候选特征向量及所选特征掩码')):
  journal.append(dict(kind='deleted_symbol_row',old=text(row)));symbols.remove(row)
# All retained figures are updated in place, with their correct aspect ratio.
rels=E.fromstring(parts['word/_rels/document.xml.rels']);targets={n.get('Id'):n.get('Target') for n in rels}
updates=[]
for filename in ('manifest_ch1_4.json','figures/manifest_56F.json'):
 manifest=json.loads((A/filename).read_text());updates += [f for f in manifest['figures'] if f.get('status','modified')=='modified']
for f in updates:
 cap,=[p for p in body if style(p)=='figurecaption' and text(p).startswith(f['figure']+' ')]
 pic=cap.getprevious();blip,=pic.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
 rid=blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed');parts['word/'+targets[rid]]=Path(f['png']).read_bytes()
 with Image.open(f['png']) as im:iw,ih=im.size
 extent,=pic.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent')
 cx=int(extent.get('cx'));cy=round(cx*ih/iw)
 extent.set('cy',str(cy))
 for size in pic.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm/{http://schemas.openxmlformats.org/drawingml/2006/main}ext'):size.set('cy',str(cy))
# Remove the seven deleted experimental images from the package as well.
used={n.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed') for n in root.iter()}
for rel in list(rels):
 if rel.get('Type','').endswith('/image') and rel.get('Id') not in used:
  parts.pop('word/'+rel.get('Target'),None);rels.remove(rel)
parts['word/_rels/document.xml.rels']=E.tostring(rels,encoding='UTF-8',xml_declaration=True,standalone=True)
# Renumber retained chapter-6 material and all textual references without cascading replacements.
ren=json.loads((A/'renumbering.json').read_text())
def renumber(e):
 value=text(e);repls=[]
 for kind,key in [('图','figures'),('表','tables')]:
  for m in re.finditer(kind+r'6-(\d+)(?!\d)',value):
   old='6-'+m[1];assert old in ren[key],('Reference to removed item',kind,old,value[:200])
   repls.append((m.start()+1,m.end(),ren[key][old]))
 for m in re.finditer(r'(?<=[（(])6-(\d+)(?=[）)])',value):
  old='6-'+m[1];assert old in ren['equations'],('Reference to removed equation',old,value[:200])
  repls.append((m.start(),m.end(),ren['equations'][old]))
 for m in re.finditer(r'6\.4\.([1-6])(?!\d)',value):
  assert m[1] in ren['sections'],('Reference to removed section',m[0],value[:200])
  repls.append((m.start(),m.end(),'6.4.'+ren['sections'][m[1]]))
 for start,end,new in sorted(repls,reverse=True):patch_range(e,start,end,new)
# Old TOC is regenerated by Word, so its references to removed headings are excluded here.
for block in body:
 if style(block).startswith('TOC') or style(block).lower().startswith('toc'):continue
 renumber(block)
# Remove bibliography entries used only in the deleted experiments; reindex the retained citations.
removed_keys=json.loads((A/'citation_usage_after_controlled_deletion.json').read_text())['only_deleted_keys']
removed_numbers={nums[key] for key in removed_keys}
refmap={old:new for new,old in enumerate((n for n in range(1,227) if n not in removed_numbers),1)}
for p in list(body):
 bookmarks=[n.get(W+'name','') for n in p.iter(W+'bookmarkStart')]
 number=next((int(n.removeprefix('latest_ref_')) for n in bookmarks if re.fullmatch(r'latest_ref_\d+',n)),None)
 if number in removed_numbers:body.remove(p);continue
 if number is not None:
  m=re.match(r'\[(\d+)\]',text(p));assert m and int(m[1])==number
  patch_range(p,m.start(1),m.end(1),str(refmap[number]))
for link in list(root.iter(W+'hyperlink')):
 if not text(link):
  link.getparent().remove(link);continue
 anchor=link.get(W+'anchor','');match=re.fullmatch(r'latest_ref_(\d+)',anchor)
 if not match:continue
 old=int(match[1]);assert old in refmap,('Citation to removed reference',old)
 link.set(W+'anchor','latest_ref_'+str(refmap[old]))
 m=re.search(r'\d+',text(link));assert m and int(m[0])==old,(anchor,text(link))
 patch_range(link,m.start(),m.end(),str(refmap[old]))
for mark in root.iter(W+'bookmarkStart'):
 match=re.fullmatch(r'latest_ref_(\d+)',mark.get(W+'name',''))
 if match:mark.set(W+'name','latest_ref_'+str(refmap[int(match[1])]))
(A/'reference_numbers_final.json').write_text(json.dumps({key:refmap[number] for key,number in nums.items() if number in refmap},ensure_ascii=False,indent=2))
for name in list(parts):
 if re.match(r'word/header\d+\.xml$',name):
  h=E.fromstring(parts[name]);literal(h,'受控动力学任务模型与跨保真补充实验设置','公交跨保真补充实验设置',required=False);parts[name]=E.tostring(h,encoding='UTF-8',xml_declaration=True,standalone=True)
parts['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
out=R/'重构论文_终审局部修订_20261015.docx'
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for name,data in parts.items():z.writestr(name,data)
(A/'word_edit_journal.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2))
print(json.dumps(dict(output=str(out),source_blocks_before=len(oldsrc),source_blocks_after=len(newsrc),figures_replaced=len(updates),inserted_paragraphs=0,split_paragraphs=0),ensure_ascii=False))
