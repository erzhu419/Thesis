"""Import only Word's updated TOC page caches, retaining the original package."""
from pathlib import Path
from zipfile import ZipFile
import re
import sys
from lxml import etree as E

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'workpapers/七章重写稿整合_20261010'))
from integrate import W, NS, body_spans
OUT=ROOT/'output/排版完成_双清单替换_20261011.docx'
STAGE=Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_two_lists_20261011/word_update.docx')


def key(p):
    s=''.join(p.xpath('.//w:instrText/text()',namespaces=NS))
    m=re.search(r'PAGEREF\s+(\S+)',s)
    return m[1] if m else None


with ZipFile(STAGE) as z:
    updated=E.fromstring(z.read('word/document.xml'))
    caches={key(p):p.xpath('.//w:t/text()',namespaces=NS)[-1]
            for p in updated.iter(W+'p') if key(p)}
with ZipFile(OUT) as z:
    raw=z.read('word/document.xml');body=E.fromstring(raw).find(W+'body')
    spans=body_spans(raw);count=0;edits=[]
    for i,p in enumerate(body):
        code=key(p)
        if code and code in caches:
            nodes=p.findall('.//'+W+'t')
            assert re.fullmatch(r'[0-9IVXLC]+',nodes[-1].text),nodes[-1].text
            nodes[-1].text=caches[code]
            a,b=spans[i];edits.append((a,b,E.tostring(p)));count+=1
    for a,b,data in reversed(edits):
        raw=raw[:a]+data+raw[b:]
    entries=[(i,raw if i.filename=='word/document.xml' else z.read(i.filename)) for i in z.infolist()]
with ZipFile(OUT,'w') as z:
    for info,data in entries:
        z.writestr(info,data)
print(f'{count} TOC page caches imported; embedded objects and non-TOC formatting retained.')
