"""Check preservation after Word layout changes, allowing Word's run merging."""
import copy
import json
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as E
from zipfile import ZipFile

WORK = Path(__file__).resolve().parent
SOURCE = WORK / '修改前.docx'
OUTPUT = WORK.parents[1] / 'output/排版完成_章内留白压缩_20261007.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'


def read(path):
    with ZipFile(path) as archive:
        return E.fromstring(archive.read('word/document.xml')), Counter(
            archive.read(n) for n in archive.namelist() if n.startswith('word/media/'))


def text(node):
    return ''.join(t.text or '' for t in node.iter(W + 't'))


def style(node):
    s = node.find(W + 'pPr/' + W + 'pStyle')
    return s.get(W + 'val') if s is not None else ''


def content(node):
    # Run boundaries and cached page-break markers are not manuscript content.
    return (text(node), ''.join(t.text or '' for t in node.iter(M + 't')),
            [tuple(sorted(t.attrib.items())) for t in node.iter(W + 'br')],
            sum(1 for _ in node.iter(W + 'tab')))


def math(node):
    node = copy.deepcopy(node)
    for parent in node.iter():
        for child in list(parent):
            if child.tag == W + 'lastRenderedPageBreak':
                parent.remove(child)
    return E.tostring(node)


def links(root):
    result = Counter(h.get(W + 'anchor') for h in root.iter(W + 'hyperlink'))
    for t in root.iter(W + 'instrText'):
        match = re.search(r'HYPERLINK\s+\\l\s+"([^"]+)"', t.text or '')
        if match:
            result[match.group(1)] += 1
    return result


a, media_a = read(SOURCE)
b, media_b = read(OUTPUT)
ab, bb = a.find(W + 'body'), b.find(W + 'body')
skip = {'figure', 'figurecaption', 'tablecaption', 'TOC1', 'TOC2', 'TOC3', 'TOC4'}
checks = {
    'prose_content_and_order': [content(p) for p in ab if p.tag == W + 'p' and style(p) not in skip]
    == [content(p) for p in bb if p.tag == W + 'p' and style(p) not in skip],
    'native_equations': Counter(math(n) for n in a.iter(M + 'oMath'))
    == Counter(math(n) for n in b.iter(M + 'oMath')),
    'sections': [E.tostring(n) for n in a.iter(W + 'sectPr')]
    == [E.tostring(n) for n in b.iter(W + 'sectPr')],
    'media_bytes': media_a == media_b,
    'hyperlink_targets': links(a) == links(b),
    'bookmarks': Counter(n.get(W + 'name') for n in a.iter(W + 'bookmarkStart'))
    == Counter(n.get(W + 'name') for n in b.iter(W + 'bookmarkStart')),
}
for ordinary in [False, True]:
    def tables(body):
        return [[content(c) for c in t.iter(W + 'tc')] for t in body
                if t.tag == W + 'tbl' and (t.find('.//' + W + 'tblHeader') is not None) == ordinary]
    checks['ordinary_tables' if ordinary else 'equation_tables'] = tables(ab) == tables(bb)
for s in ['figurecaption', 'tablecaption', 'reference']:
    checks[s + '_content_order'] = [text(p) for p in ab if style(p) == s] == [text(p) for p in bb if style(p) == s]
def fields(root):
    return Counter(n.text for n in root.iter(W + 'instrText') if not re.search(r'\bHYPERLINK\b', n.text or ''))
checks['non_hyperlink_field_instructions'] = fields(a) == fields(b)
native = json.loads((WORK / '完成分页.json').read_text(encoding='utf-8-sig'))
checks['figure_caption_same_page'] = all(n['CaptionPage'] == n['ImagePage'] for n in native['AfterFigures'])
checks['table_caption_with_first_row'] = all(n['CaptionPage'] == n['StartPage'] == n['FirstBodyPage'] for n in native['AfterTables'])
checks['figure_sizes'] = [(n['Bookmark'], n['Width'], n['Height']) for n in native['BeforeFigures']] == [(n['Bookmark'], n['Width'], n['Height']) for n in native['AfterFigures']]
checks['source_file_untouched'] = SOURCE.read_bytes() == (OUTPUT.parent / '排版完成_六图替换_20261007.docx').read_bytes()
(WORK / '保留检查.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False, indent=2))
assert all(checks.values()), 'Preservation or layout binding failed.'
