"""Check preservation of equations, table data, links, and native objects."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from io import BytesIO
import json
import re
import sys

from lxml import etree as E

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'workpapers/全文编号符号简称复核_20261011'))
from scan import W,M,NS,content
SOURCE=ROOT/'output/排版完成_残留问题替换_20261011.docx'
OUTPUT=ROOT/'output/排版完成_双清单替换_20261011.docx'


def numbered_rows(doc):
    result={}
    for row in doc.iter(W+'tr'):
        cells=row.findall(W+'tc')
        if cells and re.fullmatch(r'\([A-F0-9]+-\d+[a-z]?\)',content(cells[-1])):
            result[content(cells[-1])]=E.tostring(row)
    return result


def numeric_cells(doc):
    result=[]
    for cell in doc.iter(W+'tc'):
        t=content(cell)
        if re.fullmatch(r'[\s0-9.,+−–%±\[\]()—]+',t) and re.search(r'\d',t):
            result.append(t)
    return result


with ZipFile(SOURCE) as a,ZipFile(OUTPUT) as b:
    old=E.fromstring(a.read('word/document.xml'))
    new=E.fromstring(b.read('word/document.xml'))
    assert SOURCE.read_bytes()==(HERE/'修改前.docx').read_bytes()
    for name in ['word/styles.xml','word/numbering.xml','word/settings.xml','word/_rels/document.xml.rels']:
        assert a.read(name)==b.read(name),name
    assert [E.tostring(x) for x in old.iter(W+'sectPr')]==[E.tostring(x) for x in new.iter(W+'sectPr')]
    equations=numbered_rows(old)
    assert equations==numbered_rows(new),'numbered equations changed'
    assert numeric_cells(old)==numeric_cells(new),'numerical table entries changed'
    embedded=[p for p in a.namelist() if p.startswith('word/embeddings/')]
    changed=[]
    for name in embedded:
        if a.read(name)!=b.read(name):
            changed.append(name)
            with ZipFile(BytesIO(a.read(name))) as x, ZipFile(BytesIO(b.read(name))) as y:
                assert x.namelist()==y.namelist()
                assert [p for p in x.namelist() if x.read(p)!=y.read(p)]==['visio/pages/page1.xml']
                before=E.fromstring(x.read('visio/pages/page1.xml'))
                after=E.fromstring(y.read('visio/pages/page1.xml'))
                vn='{http://schemas.microsoft.com/office/visio/2012/main}'
                for root in [before,after]:
                    for text in root.iter(vn+'Text'):
                        for node in text.iter():
                            node.text=None;node.tail=None
                assert E.tostring(before)==E.tostring(after),'Visio geometry changed'
    assert len(changed)==1,'only figure 2-3 should change'
    bookmarks=Counter(new.xpath('.//w:bookmarkStart/@w:name',namespaces=NS))
    assert all(count==1 for count in bookmarks.values()),'duplicate bookmarks'
    targets=set(new.xpath('.//w:hyperlink/@w:anchor',namespaces=NS))
    assert not targets-set(bookmarks),'broken internal hyperlinks'
    for code in new.xpath('.//w:instrText/text()',namespaces=NS):
        m=re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)',code)
        if m:
            assert m[1] in bookmarks,code
    bib=lambda d:[E.tostring(p) for p in d.iter(W+'p') if p.find(W+'pPr/'+W+'pStyle') is not None
                  and p.find(W+'pPr/'+W+'pStyle').get(W+'val')=='bibliography']
    assert bib(old)==bib(new),'bibliography changed'
    captions=lambda d:[re.match(r'^[图表][1-7A-F]-\d+',content(p))[0] for p in d.iter(W+'p')
                        if p.find(W+'pPr/'+W+'pStyle') is not None
                        and p.find(W+'pPr/'+W+'pStyle').get(W+'val') in {'figurecaption','tablecaption'}]
    assert captions(old)==captions(new),'figure/table numbers changed'
    residual=[content(p) for p in new.find(W+'body') if any(t in content(p) for t in ['早期','前期','前序'])]
    for name in b.namelist():
        if re.fullmatch(r'word/header\d+\.xml',name):
            label=content(E.fromstring(b.read(name)))
            if any(t in label for t in ['早期','前期','前序']):
                residual.append(label)
    assert not residual,residual
    report={'source_preserved':True,'numbered_equations_preserved':len(equations),
            'numeric_table_cells_preserved':len(numeric_cells(old)),
            'reference_entries_preserved':len(bib(old)),
            'styles_and_sections_preserved':True,'internal_links_valid':True,
            'embedded_objects':len(embedded),'unchanged_embedded_objects':len(embedded)-len(changed),
            'figure_2_3_native_geometry_preserved':True,'targeted_legacy_terms_remaining':0}
    (HERE/'成品检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))
