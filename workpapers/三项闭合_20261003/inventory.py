"""Extract current Word evidence without touching the older manuscript sources."""
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'workpapers/全文深度校审_20261003'))
from extract_word import content, NS

with ZipFile(ROOT/'output/排版完成.docx') as z:
    doc = E.fromstring(z.read('word/document.xml'))
body = doc.find('w:body', NS)
blocks, references, uses = [], [], {}
section = '前置'
for i, b in enumerate(body):
    style = b.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
    style = style[0] if style else ''
    text = content(b)
    if style == 'chapter':
        section = text.strip()
    if style == 'bibliographytitle':
        section = '参考文献'
    bookmarks = b.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
    cites = b.xpath('.//w:hyperlink[starts-with(@w:anchor,"ref-")]/@w:anchor', namespaces=NS)
    row = {'block': i, 'section': section, 'style': style, 'text': text,
           'bookmarks': bookmarks, 'citations': [v[4:] for v in cites],
           'fields': b.xpath('.//w:instrText/text()', namespaces=NS)}
    if E.QName(b).localname == 'tbl':
        row['rows'] = [[content(c) for c in r.findall('w:tc', NS)] for r in b.findall('w:tr', NS)]
    blocks.append(row)
    for key in row['citations']:
        uses.setdefault(key, []).append({'block': i, 'section': section, 'text': text})
    if style == 'bibliography':
        key = next(v[4:] for v in bookmarks if v.startswith('ref-'))
        references.append({'key': key, 'block': i, 'text': text,
                           'number': int(re.match(r'\[(\d+)\]', text).group(1))})
for ref in references:
    ref['uses'] = uses.get(ref['key'], [])
(HERE/'current_blocks.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2)+'\n')
(HERE/'current_references.json').write_text(json.dumps(references, ensure_ascii=False, indent=2)+'\n')
metadata = {}
base = ROOT/'workpapers/架构重构_20261002/修改前/manuscript'
for filename in ['references.json', 'reference_preprints.json', 'reference_transport.json',
                 'reference_additions.json', 'reference_chinese.json', 'reference_metadata.json']:
    values = json.loads((base/filename).read_text())
    for value in values:
        metadata.setdefault(value['id'], {}).update(value)
(HERE/'inherited_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'blocks': len(blocks), 'references': len(references),
                  'arxiv_in_current': sum('arxiv.org' in r['text'] for r in references),
                  'inherited_records': len(metadata)}, ensure_ascii=False))
