from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import sys, json

work=Path(__file__).resolve().parent
target=Path(sys.argv[1])
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
q=lambda x:'{'+ns['w']+'}'+x
def style(node):
    p=node.find('w:pPr/w:pStyle',ns)
    return p.get(q('val')) if p is not None else ''
def text(node):return ''.join(node.xpath('.//w:t/text()|.//m:t/text()',namespaces=ns))
def main_text(doc):
    body=doc.find('w:body',ns);start=next(i for i,x in enumerate(body) if style(x)=='chapter')
    return [text(x) for x in list(body)[start:] if style(x) not in ['figure','figurecaption'] and text(x)]
def maths(node):
    if E.QName(node).namespace!=ns['m'] or E.QName(node).localname in ['ctrlPr','rPr']: return None
    return [E.QName(node).localname,sorted(node.attrib.items()),node.text if E.QName(node).localname=='t' else None,[s for c in node if (s:=maths(c)) is not None]]
with ZipFile(work/'修改前.docx') as old, ZipFile(target) as new:
    a=E.fromstring(old.read('word/document.xml'));b=E.fromstring(new.read('word/document.xml'))
    assert main_text(a)==main_text(b),'Body content/order changed beyond figure placement'
    abstract=lambda d:[text(p) for p in d.findall('.//w:p',ns) if style(p).startswith('abstract') or style(p).startswith('keywords')]
    assert abstract(a)==abstract(b),'Abstract text changed'
    ma=[maths(n) for n in a.findall('.//m:oMath',ns)];mb=[maths(n) for n in b.findall('.//m:oMath',ns)]
    assert ma==mb,'Mathematical expression content or structure changed'
    media=[p for p in old.namelist() if p.startswith('word/media/')]
    assert all(old.read(p)==new.read(p) for p in media),'Figure resource changed'
    figures=lambda d:[text(p) for p in d.findall('.//w:p',ns) if style(p)=='figurecaption']
    assert figures(a)==figures(b),'Figure captions/order changed'
    entries=[text(p) for p in b.findall('.//w:p',ns) if style(p).startswith('TOC')]
    assert not any(x.startswith(('摘要','Abstract','ABSTRACT','主要符号说明')) for x in entries)
    body=b.find('w:body',ns);styles=[style(p) for p in body]
    assert styles.index('symboltitle')<styles.index('toctitle')
    black=[];count=0
    for ti,t in enumerate(b.findall('.//w:tbl',ns)):
        if t.find('.//w:tblHeader',ns) is None:continue
        count+=1
        for r in t.find('w:tr',ns).findall('.//w:r',ns):
            rp=r.find('w:rPr',ns)
            if rp is None:continue
            bf=rp.find('w:b',ns);font=rp.find('w:rFonts',ns)
            if bf is not None and bf.get(q('val'),'1') not in ['0','false','off']:black.append(ti+1)
            if font is not None and font.get(q('eastAsia')) in ['黑体','SimHei']:black.append(ti+1)
    assert not black,black
    print(json.dumps({'main_text_and_order_unchanged':True,'abstract_unchanged':True,'native_equations_unchanged':len(ma),'image_resources_unchanged':len(media),'figure_captions_unchanged':len(figures(b)),'regular_table_headers':count,'toc_entries':len(entries),'toc_first':entries[0],'symbols_before_toc':True},ensure_ascii=False,indent=2))
