"""Read-only scope, math and link checks for the authorized terminal cleanup."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
from difflib import SequenceMatcher
import json, re, sys

A=Path(__file__).resolve().parent
before=A/'修改前_通篇校核版.docx'
after=Path(sys.argv[1]) if len(sys.argv)>1 else A.parents[1]/'重构论文_终审局部修订_20261015.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
P='{http://schemas.microsoft.com/office/word/2010/wordml}'
D='{http://schemas.openxmlformats.org/drawingml/2006/main}'
def read(path):
    with ZipFile(path) as z:return E.fromstring(z.read('word/document.xml'))
def text(e):return ''.join(n.text or '' for n in e.iter() if n.tag in (W+'t',M+'t'))
def style(e):
    n=e.find(W+'pPr/'+W+'pStyle')
    return n.get(W+'val','') if n is not None else ''
def toc(e):return style(e).lower().startswith('toc')
def mathkey(e):return E.tostring(e)
def xml(e):return E.tostring(e) if e is not None else b''
old=read(before);new=read(after)
ob=list(old.find(W+'body'));nb=list(new.find(W+'body'))
op={e.get(P+'paraId'):(i,e) for i,e in enumerate(ob) if e.tag==W+'p' and not toc(e)}
np={e.get(P+'paraId'):(i,e) for i,e in enumerate(nb) if e.tag==W+'p' and not toc(e)}
new_ids=set(np)-set(op)
order_old=[k for k in op if k in np];order_new=list(np)
format_changes=[];changes=[]
for key,(ni,n) in np.items():
    if key not in op:continue
    oi,o=op[key]
    if xml(o.find(W+'pPr'))!=xml(n.find(W+'pPr')):format_changes.append(oi)
    ot,nt=text(o),text(n)
    if ot!=nt and not toc(n):
        diffs=[{'old':ot[a:b],'new':nt[c:d]} for mode,a,b,c,d in SequenceMatcher(None,ot,nt,autojunk=False).get_opcodes() if mode!='equal']
        changes.append({'old_block':oi,'new_block':ni,'paragraph_id':key,'changes':diffs})
removed=[{'old_block':i,'text':text(e)[:180]} for key,(i,e) in op.items() if key not in np]
all_old_math=Counter(mathkey(e) for e in old.iter(M+'oMath'))
all_new_math=Counter(mathkey(e) for e in new.iter(M+'oMath'))
new_math=all_new_math-all_old_math
math_differences=[{'count':count,'text':text(E.fromstring(xml))[:250],'xml':xml.decode()} for xml,count in new_math.items()]
bookmarks={e.get(W+'name') for e in new.iter(W+'bookmarkStart')}
broken=[];toc_pending=[]
for i,block in enumerate(nb):
    dest=toc_pending if toc(block) else broken
    for e in block.iter(W+'hyperlink'):
        anchor=e.get(W+'anchor')
        if anchor and anchor not in bookmarks:dest.append({'block':i,'type':'hyperlink','target':anchor,'text':text(e)})
    for e in block.iter(W+'instrText'):
        m=re.search(r'\b(?:REF|PAGEREF)\s+(\S+)',e.text or '')
        if m and m[1] not in bookmarks:dest.append({'block':i,'type':'field','target':m[1]})
figures=[text(e) for e in nb if style(e)=='figurecaption']
tables=[text(e) for e in nb if style(e)=='tablecaption']
equations=[]
for block in nb:
    if block.tag!=W+'tbl':continue
    for row in block.findall(W+'tr'):
        cells=row.findall(W+'tc')
        if not cells:continue
        value=re.sub(r'\s+','',text(cells[-1]))
        if re.fullmatch(r'[（(](?:[2-6]|附G|[A-F])-\d+[a-z]*[）)]',value):equations.append(value[1:-1])
ch6eq=[e for e in equations if e.startswith('6-')]
fig6=[re.match(r'图(6-\d+)\s',s)[1] for s in figures if re.match(r'图6-\d+\s',s)]
tab6=[re.match(r'表(6-\d+)\s',s)[1] for s in tables if re.match(r'表6-\d+\s',s)]
refs_missing=[]
for i,e in enumerate(nb):
    if toc(e):continue
    for m in re.finditer(r'式[（(](6-\d+)[）)]',text(e)):
        if m[1] not in ch6eq:refs_missing.append({'block':i,'reference':m[0]})
reference_numbers=[];citation_label_mismatches=[]
for i,e in enumerate(nb):
    if any((b.get(W+'name') or '').startswith('latest_ref_') for b in e.iter(W+'bookmarkStart')):
        m=re.match(r'\s*\[(\d+)\]',text(e))
        if m:reference_numbers.append(int(m[1]))
    for h in e.iter(W+'hyperlink'):
        m=re.fullmatch(r'latest_ref_(\d+)',h.get(W+'anchor',''))
        ht=text(h).strip()
        if m and ht and ht.strip('[]')!=m[1]:citation_label_mismatches.append({'block':i,'anchor':h.get(W+'anchor'),'label':ht})
report={'before':str(before),'after':str(after),
 'new_top_level_paragraph_ids':sorted(new_ids),'retained_top_level_paragraph_order_unchanged':order_old==order_new,
 'retained_paragraph_properties_changed':format_changes,'removed_top_level_paragraphs':removed,
 'changed_retained_paragraphs':changes,'original_math_objects':sum(all_old_math.values()),
 'current_math_objects':sum(all_new_math.values()),'math_objects_not_identical_to_any_original':math_differences,
 'figure_caption_count':len(figures),'drawing_count':len(list(new.iter(D+'blip'))),'table_caption_count':len(tables),
 'chapter6_figures':fig6,'chapter6_tables':tab6,'chapter6_equations':ch6eq,
 'reference_count':len(reference_numbers),'references_continuous':reference_numbers==list(range(1,len(reference_numbers)+1)),
 'citation_label_target_mismatches':citation_label_mismatches,
 'missing_chapter6_equation_references':refs_missing,'broken_links_excluding_old_TOC':broken,
 'old_TOC_links_pending_native_refresh':toc_pending}
(A/'independent_word_cleanup_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('removed_top_level_paragraphs','changed_retained_paragraphs','math_objects_not_identical_to_any_original','old_TOC_links_pending_native_refresh')},ensure_ascii=False))
print(json.dumps({'removed_paragraph_count':len(removed),'changed_paragraph_count':len(changes),'new_math_tree_count':sum(new_math.values()),'old_TOC_pending_links':len(toc_pending)},ensure_ascii=False))
