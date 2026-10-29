"""Synchronize only native Word's directory page caches after reconstruction."""
from zipfile import ZipFile
from pathlib import Path
import re
from lxml import etree as E
from inventory import HERE,W,NS
from restructure import OUTPUT
from integrate import body_spans

STAGE=Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_restructure_20261012/word_ready.docx')


def key(p):
    hit=re.search(r'PAGEREF\s+(\S+)',''.join(p.xpath('.//w:instrText/text()',namespaces=NS)))
    return hit[1] if hit else None


with ZipFile(STAGE) as z:
    root=E.fromstring(z.read('word/document.xml'))
    caches={key(p):p.xpath('.//w:t/text()',namespaces=NS)[-1] for p in root.iter(W+'p') if key(p)}
with ZipFile(OUTPUT) as z:
    raw=z.read('word/document.xml');spans=body_spans(raw)
    blocks=list(E.fromstring(raw).find(W+'body'));edits=[]
    for i,p in enumerate(blocks):
        target=key(p)
        if target and target in caches:
            nodes=list(p.iter(W+'t'))
            assert re.fullmatch(r'[0-9IVXLC]+',nodes[-1].text)
            if nodes[-1].text!=caches[target]:
                nodes[-1].text=caches[target]
                a,b=spans[i];edits.append((a,b,E.tostring(p)))
    for a,b,data in reversed(edits):raw=raw[:a]+data+raw[b:]
    entries=[(info,raw if info.filename=='word/document.xml' else z.read(info.filename)) for info in z.infolist()]
with ZipFile(OUTPUT,'w') as z:
    for info,data in entries:z.writestr(info,data)
print(f'{len(caches)} TOC page targets checked; {len(edits)} changed caches synchronized.')
