"""Check losses reachable during this actual chapter reconstruction."""
from collections import Counter
from zipfile import ZipFile
from pathlib import Path
import json
import re
from lxml import etree as E
from inventory import SOURCE, HERE, W, M, NS, content, load
from restructure import OUTPUT, style


def structure(node):
    return (node.tag,tuple(sorted(node.attrib.items())),node.text,node.tail,tuple(structure(c) for c in node))

_,old,oldblocks=load(SOURCE)
_,new,newblocks=load(OUTPUT)
assert SOURCE.read_bytes()==(HERE/'修改前.docx').read_bytes(),'source changed'
with ZipFile(SOURCE) as a,ZipFile(OUTPUT) as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)<={'word/document.xml','word/header4.xml'},changed
    assert a.namelist()==b.namelist(),'package entries changed'
    preserved=[n for n in a.namelist() if n.startswith(('word/embeddings/','word/media/'))]
    assert all(a.read(n)==b.read(n) for n in preserved)
assert [E.tostring(x) for x in old.iter(W+'sectPr')]==[E.tostring(x) for x in new.iter(W+'sectPr')],'sections changed'
names=Counter(new.xpath('.//w:bookmarkStart/@w:name',namespaces=NS))
assert all(v==1 for v in names.values()),'duplicate bookmark names'
start=Counter(new.xpath('.//w:bookmarkStart/@w:id',namespaces=NS))
end=Counter(new.xpath('.//w:bookmarkEnd/@w:id',namespaces=NS))
assert start==end,'unpaired bookmarks'
targets=set(new.xpath('.//w:hyperlink/@w:anchor',namespaces=NS))
for code in new.xpath('.//w:instrText/text()',namespaces=NS):
    hit=re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)',code)
    if hit:targets.add(hit[1])
assert not targets-set(names),targets-set(names)
old_bib=[content(p) for p in oldblocks if style(p)=='bibliography']
new_bib=[content(p) for p in newblocks if style(p)=='bibliography']
assert new_bib[:len(old_bib)]==old_bib,'existing bibliography changed'
assert len(new_bib)==len(old_bib)+3


def equation_math(blocks):
    result={}
    for p in blocks:
        if p.tag!=W+'tbl':continue
        cells=p.findall(W+'tr/'+W+'tc')
        if not cells or not re.fullmatch(r'\([2-6]-\d+\)',content(cells[-1])):continue
        for bm in cells[-1].iter(W+'bookmarkStart'):
            if bm.get(W+'name').startswith('eq'):
                result[bm.get(W+'name')]=[structure(x) for x in cells[-2].iter(M+'oMath')]
    return result


before=equation_math(oldblocks);after=equation_math(newblocks)
replaced_g=set(oldblocks[1211].xpath('.//w:bookmarkStart/@w:name',namespaces=NS))
retained=0
for name,math in before.items():
    if name.startswith('eq2_') or name in replaced_g:continue
    assert name in after and math==after[name],('research equation changed',name)
    retained+=1
old_results=[structure(oldblocks[i+1]) for i,p in enumerate(oldblocks)
             if 633<=i<1774 and style(p)=='tablecaption' and oldblocks[i+1].tag==W+'tbl']
new_tables={structure(p) for p in newblocks if p.tag==W+'tbl'}
assert all(p in new_tables for p in old_results),'result table changed'
outlook=lambda blocks:[content(p) for p in blocks[next(i for i,p in enumerate(blocks) if content(p)=='7.3 研究展望'):next(i for i,p in enumerate(blocks) if style(p)=='bibliographytitle')]]
assert outlook(oldblocks)==outlook(newblocks),'7.3 changed'
titles=[content(p) for p in newblocks if style(p) in {'chapter','h2','h3'}]
second=[s for s in titles if re.match(r'^2\.',s)]
assert second==['2.1 传统公交在途控制','2.2 强化学习的基本原理','2.3 鲁棒强化学习与环境适应','2.4 迁移强化学习','2.5 世界模型与模型辅助学习','2.6 本章小结']
assert not any(re.search(r'第2\.[0-9]|图2-|表2-',content(p)) for p in newblocks)
report={'source_preserved':True,'media_and_embeddings_preserved':len(preserved),'sections_preserved':True,
        'research_equations_preserved':retained,'original_research_result_tables_preserved':len(old_results),
        'existing_references_preserved':len(old_bib),'verified_references_added':3,'internal_links_valid':True,
        'chapter_2_sections':second,'section_7_3_preserved':True}
(HERE/'结构核对.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
