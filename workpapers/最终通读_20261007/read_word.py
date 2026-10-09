"""Extract a read-only review copy, including native equation structure."""
import importlib.util
import json
import re
import shutil
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_自有论文与仓库补齐_20261007.docx'
BACKUP = WORK / '修改前.docx'
if not BACKUP.exists():
    shutil.copy2(SOURCE, BACKUP)
spec = importlib.util.spec_from_file_location('word_review_extract', ROOT / 'workpapers/全文深度校审_20261003/extract_word.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
NS = helper.NS
original_content = helper.content


def content(node):
    # Drawing/VML tags can share local names with OMML fractions and accents.
    if E.QName(node).namespace not in (NS['w'], NS['m']):
        return ''
    return original_content(node)


helper.content = content
with ZipFile(BACKUP) as archive:
    doc = E.fromstring(archive.read('word/document.xml'))
blocks, chapters = [], {}
chapter = '00_前置'
for i, node in enumerate(doc.find('w:body', NS)):
    s = node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
    style = s[0] if s else ''
    text = content(node)
    if style == 'chapter':
        chapter = re.sub(r'\s+', '_', text)
    elif style == 'bibliographytitle':
        chapter = '参考文献'
    block = {'index': i, 'style': style, 'text': text, 'chapter': chapter,
             'bookmarks': node.xpath('.//w:bookmarkStart/@w:name', namespaces=NS),
             'fields': node.xpath('.//w:instrText/text()', namespaces=NS)}
    if E.QName(node).localname == 'tbl':
        block['rows'] = [[content(c) for c in row.findall('w:tc', NS)] for row in node.findall('w:tr', NS)]
    blocks.append(block)
    lines = chapters.setdefault(chapter, [])
    if 'rows' in block:
        lines.append(f'[{i} table]')
        lines.extend(' | '.join(row) for row in block['rows'])
    else:
        lines.append(f'[{i} {style or "paragraph"}] {text}')
for name, lines in chapters.items():
    (WORK / (name + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
(WORK / 'blocks.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'source': str(SOURCE), 'paragraph_blocks': len(blocks),
                  'chapter_characters': {name: sum(len(t) for t in lines) for name, lines in chapters.items()}}, ensure_ascii=False, indent=2))
