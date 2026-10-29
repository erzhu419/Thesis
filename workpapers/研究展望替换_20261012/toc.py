"""Import only native Word's updated TOC page numbers, not its package rewrite."""
from pathlib import Path
from zipfile import ZipFile
import re
import sys
from lxml import etree as E

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans
from replace import OUTPUT, TEXTS, NS, W, text
STAGE=Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_outlook_20261012/word.docx')


def key(p):
    m=re.search(r'PAGEREF\s+(\S+)',''.join(p.xpath('.//w:instrText/text()',namespaces=NS)))
    return m[1] if m else None


with ZipFile(STAGE) as z:
    root=E.fromstring(z.read('word/document.xml'))
    pages={key(p):p.xpath('.//w:t/text()',namespaces=NS)[-1] for p in root.iter(W+'p') if key(p)}
with ZipFile(OUTPUT) as z:
    raw=z.read('word/document.xml')
    root=E.fromstring(raw)
    blocks=list(root.find(W+'body'))
    spans=body_spans(raw)
    edits=[]
    for i,p in enumerate(blocks):
        bookmark=key(p)
        if bookmark and bookmark in pages:
            nodes=p.findall('.//'+W+'t')
            assert re.fullmatch(r'[0-9IVXLC]+',nodes[-1].text)
            if nodes[-1].text!=pages[bookmark]:
                nodes[-1].text=pages[bookmark]
                a,b=spans[i]
                edits.append((a,b,E.tostring(p)))
    for a,b,data in reversed(edits):
        raw=raw[:a]+data+raw[b:]
    entries=[(info,raw if info.filename=='word/document.xml' else z.read(info.filename)) for info in z.infolist()]
with ZipFile(OUTPUT,'w') as z:
    for info,data in entries:
        z.writestr(info,data)
after=list(E.fromstring(raw).find(W+'body'))
start=next(i for i,p in enumerate(after) if text(p).strip()=='7.3 研究展望')+1
assert [text(p) for p in after[start:start+len(TEXTS)]]==TEXTS
print(f'{len(edits)} changed TOC page caches synchronized; supplied 7.3 text unchanged. No PDF generated.')
