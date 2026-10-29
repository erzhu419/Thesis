"""Verify preservation and link targets before delivering the edited Word."""
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from lxml import etree as E
import json
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'workpapers/全文编号符号简称复核_20261011'))
import scan
from scan import W,content
sys.path.insert(0,str(ROOT/'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans

BASE=HERE/'修改前_用户已完成1至13项.docx'
OUT=ROOT/'output/排版完成_残留问题替换_20261011.docx'
STAGE=Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_replacements_20261011/toc_stage.docx')


def toc_cache():
    with ZipFile(STAGE) as z:
        saved=E.fromstring(z.read('word/document.xml')).find(W+'body')
    def style(p):
        s=p.find(W+'pPr/'+W+'pStyle')
        return s.get(W+'val') if s is not None else ''
    pages=[list(p.iter(W+'t'))[-1].text for p in saved if style(p) in ('TOC1','TOC2','TOC3')]
    with ZipFile(OUT) as z:
        raw=z.read('word/document.xml')
        body=E.fromstring(raw).find(W+'body')
        spans=body_spans(raw)
        indexes=[i for i,p in enumerate(body) if style(p) in ('TOC1','TOC2','TOC3')]
        assert len(pages)==len(indexes)==165
        result=raw
        for i,page in reversed(list(zip(indexes,pages))):
            assert page.isdigit()
            a,b=spans[i]
            fragment=raw[a:b]
            m=list(re.finditer(rb'(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)',fragment))[-1]
            assert m[2].decode().isdigit()
            fragment=fragment[:m.start(2)]+page.encode()+fragment[m.end(2):]
            result=result[:a]+fragment+result[b:]
        entries=[(info,result if info.filename=='word/document.xml' else z.read(info.filename))for info in z.infolist()]
    with ZipFile(OUT,'w') as z:
        for info,data in entries:z.writestr(info,data)
    print('165 TOC page results transferred; source XML and mathematical structures otherwise unchanged.')


def main():
    scan.HERE=HERE/'替换后'
    scan.HERE.mkdir(exist_ok=True)
    scan.run(OUT)
    prior=json.loads((HERE/'编号与引用检查.json').read_text())
    after=json.loads((scan.HERE/'编号与引用检查.json').read_text())
    with ZipFile(BASE) as a,ZipFile(OUT) as b:
        raw_a=a.read('word/document.xml');raw_b=b.read('word/document.xml')
        old=E.fromstring(raw_a).find(W+'body');new=E.fromstring(raw_b).find(W+'body')
        anchors=[n.get(W+'name')for n in new.iter(W+'bookmarkStart')]
        links=[n.get(W+'anchor')for n in new.iter(W+'hyperlink') if n.get(W+'anchor')]
        missing_links=sorted(set(links)-set(anchors))
        changed=[name for name in a.namelist() if a.read(name)!=b.read(name)]
        toc_a=[E.tostring(x)for x in list(old)[:80]]
        toc_b=[E.tostring(x)for x in list(new)[:80]]
        symold=old[81].findall(W+'tr')
        symnew=new[81].findall(W+'tr')
        def symbol_xml(row):
            cell=deepcopy(row.findall(W+'tc')[0])
            for spacing in cell.findall('.//'+W+'pPr/'+W+'spacing'):
                spacing.getparent().remove(spacing)
            return E.tostring(cell)
        diagrams=json.loads((HERE/'图替换关系.json').read_text())
        original_diagrams={r['figure']:all(a.read(r[k])==b.read(r[k])for k in ['embedding','preview'])for r in diagrams if 'embedding'in r}
        numeric_tables={}
        for caption in ['表F-1','表F-2']:
            def data(body):
                ix=next(i for i,p in enumerate(body) if content(p).startswith(caption+' '))
                return [[content(c)for c in row.findall(W+'tc')[1:]] for row in body[ix+1].findall(W+'tr')[1:]]
            numeric_tables[caption]=data(old)==data(new)
        result={
            'original_manuscript_unchanged':(ROOT/'output/排版完成_全文编号符号简称复核_20261011.docx').read_bytes()==BASE.read_bytes(),
            'abstract_and_frontmatter_through_abstract_unchanged':toc_a==toc_b,
            'completed_05_to_13_prose_preserved':all(any(content(p)==content(old[i])for p in new)for i in [303,341,523,930,949,986]),
            'symbol_column_content_and_fonts_unchanged':len(symold)==len(symnew) and all(symbol_xml(x)==symbol_xml(y)for x,y in zip(symold,symnew)),
            'equation_numbers_unchanged':list(prior['equations'])==list(after['equations']),
            'caption_numbers_and_text_unchanged':list(prior['captions'])==list(after['captions']) and all(prior['captions'][k]['text']==after['captions'][k]['text']for k in prior['captions']),
            'numbered_equations':len(after['equations']),'captions':len(after['captions']),
            'missing_hyperlink_targets':missing_links,'field_errors':after['errors'],
            'original_native_figures_unchanged':original_diagrams,
            'F_table_numerical_cells_unchanged':numeric_tables,
            'changed_package_parts':changed,
        }
        # Bibliography and section XML must survive local prose replacement.
        def bibliography(body):
            start=next(i for i,p in enumerate(body) if content(p)=='参考文献')
            end=next(i for i,p in enumerate(body) if i>start and content(p).startswith('附录A '))
            return [E.tostring(p)for p in list(body)[start+1:end]]
        result['bibliography_xml_unchanged']=bibliography(old)==bibliography(new)
        result['sections_unchanged']=[E.tostring(n)for n in old.iter(W+'sectPr')]==[E.tostring(n)for n in new.iter(W+'sectPr')]
        (HERE/'成品验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False))
        assert all(result[k] for k in ['original_manuscript_unchanged','abstract_and_frontmatter_through_abstract_unchanged','completed_05_to_13_prose_preserved','symbol_column_content_and_fonts_unchanged','equation_numbers_unchanged','caption_numbers_and_text_unchanged','bibliography_xml_unchanged','sections_unchanged'])
        assert not missing_links and not after['errors']
        assert all(original_diagrams.values()) and all(numeric_tables.values())


if __name__=='__main__':
    toc_cache() if len(sys.argv)>1 and sys.argv[1]=='toc' else main()
