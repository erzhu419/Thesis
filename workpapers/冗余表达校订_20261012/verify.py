"""Check the objects and scientific records that wording edits must not alter."""
from collections import Counter
from zipfile import ZipFile
import json
import re

from lxml import etree as E

from edit import BASE, OUT, HERE, ROOT, W, M, text


def main():
    with ZipFile(BASE) as a, ZipFile(OUT) as b:
        assert a.namelist()==b.namelist(), 'package parts changed'
        unchanged=[n for n in a.namelist() if n!='word/document.xml']
        assert all(a.read(n)==b.read(n) for n in unchanged), 'native object or formatting part changed'
        src=E.fromstring(a.read('word/document.xml'))
        dst=E.fromstring(b.read('word/document.xml'))
    sb,db=list(src.find(W+'body')),list(dst.find(W+'body'))
    original=json.loads((ROOT/'workpapers/全文引用符号复核_20261012/全文块.json').read_text())
    labels=json.loads((ROOT/'workpapers/全文引用符号复核_20261012/编号与引用检查.json').read_text())
    section=lambda r:[E.tostring(n) for n in r.iter(W+'sectPr')]
    assert section(src)==section(dst), 'section configuration changed'
    for i in range(64,70):
        assert any(E.tostring(sb[i])==E.tostring(n) for n in db[:80]), ('Chinese abstract changed',i)
    # Scientific result tables stay identical; only explanatory/setting tables were edited.
    changed_tables={657,788,837,1511,1710,1712}
    eq_indices={r['i'] for r in labels['equations'].values()}
    original_tables=[n for i,n in enumerate(sb) if n.tag==W+'tbl' and i not in changed_tables|eq_indices]
    final_tables=[E.tostring(n) for n in db if n.tag==W+'tbl']
    for n in original_tables:
        # Equation-reference prose may change its number, but all cell values must remain.
        val=text(n)
        assert any(text(E.fromstring(s))==val for s in final_tables), ('result or fixed table changed',val[:120])
    for tag in ('drawing','object','pict'):
        a=Counter(E.tostring(n) for n in src.iter(W+tag))
        b=Counter(E.tostring(n) for n in dst.iter(W+tag))
        assert a==b, ('native picture/object changed',tag)
    # Moving/renumbering equations must preserve each retained formula's native OMML.
    removed={'3-31','5-24','5-25','5-26'}
    math_final=Counter(E.tostring(m) for m in dst.iter(M+'oMath'))
    for label,row in labels['equations'].items():
        if label in removed: continue
        for m in sb[row['i']].iter(M+'oMath'):
            assert math_final[E.tostring(m)], ('retained equation changed',label)
    src_refs=[E.tostring(n) for n in sb if any(E.QName(c).localname=='pStyle' and c.get(W+'val')=='bibliography' for c in n.iter())]
    dst_refs=[E.tostring(n) for n in db if any(E.QName(c).localname=='pStyle' and c.get(W+'val')=='bibliography' for c in n.iter())]
    assert src_refs==dst_refs, 'bibliography changed'
    anchors=[n.get(W+'name') for n in dst.iter(W+'bookmarkStart')]
    assert len(anchors)==len(set(anchors)), 'duplicate bookmark'
    starts=Counter(n.get(W+'id') for n in dst.iter(W+'bookmarkStart'))
    ends=Counter(n.get(W+'id') for n in dst.iter(W+'bookmarkEnd'))
    assert starts==ends, 'unbalanced bookmarks'
    findings=json.loads((HERE/'字段与目标检查.json').read_text())
    assert not findings['findings'], findings['findings']
    result={'unchanged_package_parts':len(unchanged),'native_media_embeddings':sum(n.startswith(('word/media/','word/embeddings/')) for n in unchanged),
        'unchanged_sections':len(section(src)), 'Chinese_abstract':'unchanged','unchanged_scientific_tables':len(original_tables),
        'unchanged_bibliography_entries':len(src_refs),'retained_numbered_equations':len(labels['equations'])-len(removed),
        'reference_fields_checked':len(findings['checked_fields']),'field_findings':0}
    (HERE/'内容与对象保护核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
