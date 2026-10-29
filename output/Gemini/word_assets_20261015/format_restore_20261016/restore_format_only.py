"""Remove prose emphasis only; preserve text, native math and structural labels."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import copy,re,json
A=Path(__file__).resolve().parent;R=A.parents[1]
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
text=lambda e:''.join(n.text or''for n in e.iter()if n.tag in(W+'t',M+'t'))
with ZipFile(A/'修改前.docx')as z:parts={n:z.read(n)for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find(W+'body')
original_paragraphs=[text(p)for p in root.iter(W+'p')]
original_ppr=[E.tostring(p.find(W+'pPr'))if p.find(W+'pPr')is not None else None for p in root.iter(W+'p')]
original_math=[E.tostring(m)for m in root.iter(M+'oMath')]
original_fields=[n.text or''for n in root.iter(W+'instrText')]
original_bookmarks=[(n.tag,dict(n.attrib))for n in root.iter()if n.tag in(W+'bookmarkStart',W+'bookmarkEnd')]
# These are the actual 50 section prompts in the approved complete source,
# including D.2, whose short prompt occurs after prose in the same paragraph.
md=(A.parent/'md_sync_latest_20261016/最新MD.md').read_text()
prompts=re.findall(r'\*\*([A-F]\.\d+\s+[^*\n]+?。)\*\*',md)
assert len(prompts)==50,len(prompts)
patterns=[re.compile(r'\s*'.join(re.escape(ch)for ch in re.sub(r'\s+','',label)))for label in prompts]
proof=re.compile(r'^\s*证明(?:要点)?[。:：]')
theorem=re.compile(r'(?:定理|命题|引理|推论)[A-F]\.\d+[a-z]?(?:[（(][^）)\n]{0,80}[）)])?[。:：]')
# Numbered formulas, including their label cells, are excluded in full.
eq_tables=set()
for table in body.findall(W+'tbl'):
 for row in table.findall(W+'tr'):
  cells=row.findall(W+'tc')
  if cells and re.fullmatch(r'\s*[（(][1-7A-F]-\d+[a-z]*[）)]\s*',text(cells[-1])):eq_tables.add(table);break

def val_true(n):return n is not None and n.get(W+'val','1')not in('0','false','off')
def positive(run):
 rp=run.find(W+'rPr')
 return rp is not None and(any(val_true(rp.find(W+tag))for tag in('b','bCs'))or (rp.find(W+'rStyle')is not None and rp.find(W+'rStyle').get(W+'val')=='Strong'))
def clear(run):
 rp=run.find(W+'rPr');count=0
 if rp is None:return count
 for n in list(rp):
  if (n.tag in(W+'b',W+'bCs')and val_true(n))or(n.tag==W+'rStyle'and n.get(W+'val')=='Strong'):
   rp.remove(n);count+=1
 return count

def split_text_run(run,cuts):
 # Only a boundary run needs splitting. All field codes/bookmarks remain at
 # their existing sibling positions and the visible text stays in order.
 assert len(run.findall(W+'t'))==1 and not any(c.tag not in(W+'rPr',W+'t')for c in run),('Unsupported guide boundary',text(run))
 value=run.find(W+'t').text or'';edges=[0]+sorted(cuts)+[len(value)];new=[]
 for a,b in zip(edges,edges[1:]):
  r=copy.deepcopy(run);t=r.find(W+'t');t.text=value[a:b];t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');new.append((a,b,r))
 return new

changes=[];preserved=[];appendix=False;in_table=0
for block_index,block in enumerate(body):
 if block.tag==W+'p':
  style=block.find(W+'pPr/'+W+'pStyle');sid=style.get(W+'val')if style is not None else''
  if sid=='chapter'and text(block).startswith('附录A'):appendix=True
 if block in eq_tables:continue
 for p in block.iter(W+'p')if block.tag!=W+'p'else[block]:
  style=p.find(W+'pPr/'+W+'pStyle');sid=style.get(W+'val')if style is not None else''
  in_table=block.tag==W+'tbl'
  # Table prose is handled only after the reference comparison; no font or
  # table geometry is reset by this operation.
  eligible=sid=='body'or in_table or sid in('abstractcnbody','abstractcnfirst','abstractenbody','abstractenfirst')
  if not eligible:continue
  value=text(p);ranges=[]
  if appendix and sid=='body'and not in_table:
   for pattern in patterns:
    ranges.extend((m.start(),m.end())for m in pattern.finditer(value))
   section_ranges=list(ranges)
   ranges.extend((m.start(),m.end())for m in theorem.finditer(value)if not value[:m.start()].strip()or any(not value[b:m.start()].strip()for a,b in section_ranges if b<=m.start()))
   m=proof.match(value)
   if m:ranges.append((m.start(),m.end()))
  for a,b in ranges:preserved.append({'block':block_index,'text':value[a:b]})
  offset=0;removed=0;split_count=0
  # Iterate the original paragraph tree so splitting runs cannot disturb
  # current offsets; mathematical text contributes length but remains intact.
  run_offsets={}
  for n in p.iter():
   if n.tag==W+'r':run_offsets[n]=offset
   if n.tag in(W+'t',M+'t'):offset+=len(n.text or'')
  for run in list(p.iter(W+'r')):
   if not positive(run):continue
   if list(run.iter(W+'instrText')):
    removed+=clear(run);continue
   size=len(text(run));a=run_offsets[run];b=a+size
   if not size:removed+=clear(run);continue
   covers=[(max(a,x),min(b,y))for x,y in ranges if max(a,x)<min(b,y)]
   if not covers:removed+=clear(run);continue
   if any(x<=a and b<=y for x,y in ranges):continue
   cuts={k-a for x,y in covers for k in(x,y)if a<k<b}
   parent=run.getparent();at=parent.index(run)
   for lo,hi,replacement in split_text_run(run,cuts):
    if not any(x<=a+lo and a+hi<=y for x,y in ranges):removed+=clear(replacement)
    parent.insert(at,replacement);at+=1
   parent.remove(run);split_count+=1
  if removed:changes.append({'block':block_index,'style':sid,'table':in_table,'removed_emphasis_properties':removed,'split_boundary_runs':split_count,'text':value[:90]})
# Match the old manuscript's two short keyword labels, including the colon.
keywords=[]
for p in body.findall(W+'p'):
 value=text(p);label=next((v for v in('关键词：','Keywords:')if value.startswith(v)),None)
 if label is None:continue
 boundary=len(label);offset=0
 for run in list(p.iter(W+'r')):
  size=len(text(run));a=offset;b=a+size;offset=b
  if a>=boundary or not size:continue
  targets=[run]
  if a<boundary<b:
   parent=run.getparent();at=parent.index(run);fragments=split_text_run(run,{boundary-a});targets=[]
   for lo,hi,newrun in fragments:
    parent.insert(at,newrun);at+=1
    if a+hi<=boundary:targets.append(newrun)
   parent.remove(run)
  for target in targets:
   rp=target.find(W+'rPr')
   if rp is None:rp=E.Element(W+'rPr');target.insert(0,rp)
   bold=rp.find(W+'b')
   if bold is None:bold=E.SubElement(rp,W+'b')
   bold.attrib.pop(W+'val',None)
 keywords.append(label)
assert keywords==['关键词：','Keywords:'],keywords
# Formatting-only invariants: retained native math/fields/bookmarks and every
# paragraph's visible text, order, paragraph properties are unchanged.
assert original_paragraphs==[text(p)for p in root.iter(W+'p')]
assert original_ppr==[E.tostring(p.find(W+'pPr'))if p.find(W+'pPr')is not None else None for p in root.iter(W+'p')]
assert original_math==[E.tostring(m)for m in root.iter(M+'oMath')]
assert original_fields==[n.text or''for n in root.iter(W+'instrText')]
assert original_bookmarks==[(n.tag,dict(n.attrib))for n in root.iter()if n.tag in(W+'bookmarkStart',W+'bookmarkEnd')]
parts['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
out=R/'重构论文_格式恢复_20261016.docx'
with ZipFile(out,'w',ZIP_DEFLATED)as z:
 for name,data in parts.items():z.writestr(name,data)
report={'output':str(out),'body_abstract_paragraphs_cleaned':sum(not x['table']for x in changes),'table_paragraphs_cleaned':sum(x['table']for x in changes),'emphasis_properties_removed':sum(x['removed_emphasis_properties']for x in changes),'boundary_runs_split':sum(x['split_boundary_runs']for x in changes),'actual_section_prompts':len(prompts),'preserved_label_occurrences':len(preserved),'keyword_labels_restored':keywords,'text_and_paragraph_properties_unchanged':True,'native_math_full_XML_unchanged':True,'fields_and_bookmarks_unchanged':True,'styles_fonts_sizes_spacing_unchanged':True}
(A/'format_restore_applied.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in('changed_paragraphs','preserved_labels')},ensure_ascii=False))
