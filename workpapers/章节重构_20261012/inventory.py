"""Extract the current Word structure for scoped chapter migration."""
from pathlib import Path
from zipfile import ZipFile
import json
import re
from lxml import etree as E

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS={'w':W[1:-1],'m':M[1:-1]}
SOURCE=ROOT/'output/排版完成_研究展望替换_20261012.docx'
PLAN=ROOT/'markdown/第二章及第3至6章替换正文_20261012.docx'


def content(p):
    return ''.join(p.xpath('.//w:t/text() | .//m:t/text()',namespaces=NS))


def load(path):
    with ZipFile(path) as z:
        raw=z.read('word/document.xml')
    root=E.fromstring(raw)
    return raw,root,list(root.find(W+'body'))


def row(i,p):
    s=p.find(W+'pPr/'+W+'pStyle')
    v=content(p)
    return {'index':i,'kind':E.QName(p).localname,'style':s.get(W+'val') if s is not None else None,
            'text':v,'equation':bool(p.xpath('.//m:oMath',namespaces=NS)),
            'drawing':bool(p.xpath('.//w:drawing | .//w:object | .//w:pict',namespaces=NS)),
            'section':p.find('.//'+W+'sectPr') is not None}


if __name__=='__main__':
    HERE.mkdir(exist_ok=True)
    for path,name in [(SOURCE,'现稿结构'),(PLAN,'方案结构')]:
        raw,root,blocks=load(path)
        rows=[row(i,p) for i,p in enumerate(blocks)]
        (HERE/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        (HERE/f'{name}.txt').write_text('\n'.join(f"[{r['index']}] [{r['style'] or r['kind']}] {r['text']}" for r in rows))
        if name=='现稿结构':
            for r in rows:
                if r['style'] in {'h1','h2','h3','figurecaption','tablecaption','bibliographytitle'} or (r['index']<650 and r['equation'] and r['kind']=='tbl'):
                    print(f"[{r['index']}] [{r['style'] or r['kind']}] {r['text'][:170]}")
