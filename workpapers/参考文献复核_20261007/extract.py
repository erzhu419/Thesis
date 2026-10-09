"""Extract the current Word bibliography and the actual citation contexts."""
import importlib.util
import json
import re
import shutil
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path('/home/erzhu419/mine_code/Thesis')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_最终通读校订_20261007.docx'
BACKUP = HERE / '修改前.docx'
if not BACKUP.exists():
    shutil.copy2(SOURCE, BACKUP)
spec = importlib.util.spec_from_file_location('extract_helper', ROOT / 'workpapers/全文深度校审_20261003/extract_word.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
NS = helper.NS
original = helper.content


def content(node):
    if E.QName(node).namespace not in (NS['w'], NS['m']):
        return ''
    return original(node)


helper.content = content
with ZipFile(BACKUP) as archive:
    doc = E.fromstring(archive.read('word/document.xml'))
blocks, references, uses = [], [], {}
chapter = '前置部分'
for i, node in enumerate(doc.find('w:body', NS)):
    styles = node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
    style = styles[0] if styles else ''
    text = content(node)
    if style == 'chapter' or style == 'bibliographytitle':
        chapter = text
    bookmarks = node.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
    anchors = node.xpath('.//w:hyperlink/@w:anchor', namespaces=NS)
    block = {'index': i, 'style': style, 'chapter': chapter, 'text': text,
             'bookmarks': bookmarks, 'anchors': anchors}
    blocks.append(block)
    keys = [key for key in bookmarks if key.startswith('ref-')]
    if keys:
        match = re.match(r'\[(\d+)\]', text)
        references.append({'number': int(match.group(1)), 'block': i, 'key': keys[0], 'text': text})
    else:
        for key in dict.fromkeys(anchors):
            if key.startswith('ref-'):
                uses.setdefault(key, []).append(block)
for ref in references:
    ref['uses'] = uses.get(ref['key'], [])
    arxiv = re.search(r'https?://arxiv.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', ref['text'])
    doi = re.search(r'(?:DOI:|doi.org/)(10\.[^\s]+)', ref['text'], re.I)
    ref['arxiv'] = arxiv.group(1) if arxiv else ''
    ref['doi'] = doi.group(1).rstrip('.') if doi else ''
    ref['urls'] = re.findall(r'https?://[^\s]+', ref['text'])
(HERE / 'blocks.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding='utf-8')
(HERE / 'references.json').write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding='utf-8')
(HERE / '文献与引用原文.txt').write_text('\n\n'.join(
    ref['text'] + '\n' + '\n'.join(f"正文[{v['index']}] {v['chapter']}: {v['text']}" for v in ref['uses'])
    for ref in references), encoding='utf-8')
print(json.dumps({'references': len(references), 'arxiv': sum(bool(v['arxiv']) for v in references),
                  'doi': sum(bool(v['doi']) for v in references),
                  'uncited': [v['number'] for v in references if not v['uses']]}, ensure_ascii=False))
