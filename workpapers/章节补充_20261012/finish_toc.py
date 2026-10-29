"""Import only Word's refreshed directory page numbers, not its rewritten objects."""
from pathlib import Path
from zipfile import ZipFile
import re
from lxml import etree as E
from inventory import W, NS
from supplement import OUTPUT, body_spans

STAGE = Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_supplement_20261012/word_checked.docx')

def target(p):
    match = re.search(r'PAGEREF\s+(\S+)', ''.join(p.xpath('.//w:instrText/text()', namespaces=NS)))
    return match[1] if match else None

with ZipFile(STAGE) as z:
    root = E.fromstring(z.read('word/document.xml'))
    caches = {target(p): p.xpath('.//w:t/text()', namespaces=NS)[-1]
              for p in root.iter(W + 'p') if target(p)}
with ZipFile(OUTPUT) as z:
    raw = z.read('word/document.xml')
    spans = body_spans(raw)
    blocks = list(E.fromstring(raw).find(W + 'body'))
    edits = []
    for i, p in enumerate(blocks):
        key = target(p)
        if key and key in caches:
            nodes = list(p.iter(W + 't'))
            assert re.fullmatch(r'[0-9IVXLC]+', nodes[-1].text or ''), key
            if nodes[-1].text != caches[key]:
                nodes[-1].text = caches[key]
                a, b = spans[i]
                edits.append((a, b, E.tostring(p)))
    for a, b, data in reversed(edits):
        raw = raw[:a] + data + raw[b:]
    entries = [(info, raw if info.filename == 'word/document.xml' else z.read(info.filename))
               for info in z.infolist()]
with ZipFile(OUTPUT, 'w') as z:
    for info, data in entries:
        z.writestr(info, data)
print(f'{len(caches)} directory page targets checked; {len(edits)} caches synchronized.')
