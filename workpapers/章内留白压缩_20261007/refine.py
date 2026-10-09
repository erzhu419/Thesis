"""Defer floating figure/table blocks within their existing chapter."""
import copy
import io
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as E
from zipfile import ZipFile

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
DOCX = ROOT / 'output/排版完成_章内留白压缩_20261007.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
native_path = WORK / (sys.argv[1] if len(sys.argv) > 1 else '第一步分页.json')
atomic_only = len(sys.argv) > 2 and sys.argv[2] == 'atomic'
suffix = native_path.stem + ('_公式衔接' if atomic_only else '')
native = json.loads(native_path.read_text(encoding='utf-8-sig'))
assert native['Status'] == 'completed'
shutil.copy2(DOCX, WORK / ('顺延前_' + suffix + '.docx'))
with ZipFile(DOCX) as z:
    infos = z.infolist()
    parts = {i.filename: z.read(i.filename) for i in infos}
raw = parts['word/document.xml']
for _, (prefix, uri) in E.iterparse(io.BytesIO(raw), events=['start-ns']):
    E.register_namespace(prefix, uri)
d = E.fromstring(raw)
before = copy.deepcopy(d)
body = d.find(W + 'body')
nodes = list(body)


def text(n):
    return ''.join(t.text or '' for t in n.iter(W + 't'))


def style(n):
    s = n.find(W + 'pPr/' + W + 'pStyle')
    return s.get(W + 'val') if s is not None else ''


def normal(s):
    return re.sub(r'\s+', '', s)


def block_end(index):
    if style(nodes[index]) == 'figure':
        assert style(nodes[index + 1]) == 'figurecaption'
        return index + 2
    if style(nodes[index]) == 'tablecaption':
        assert nodes[index + 1].tag == W + 'tbl'
        return index + 2
    return None


def height(n):
    if style(n) in ['h2', 'h3']:
        return 38
    prefix = normal(text(n))[:18]
    if len(prefix) >= 10:
        hits = [b['Height'] for g in native['AfterGeometry'] for b in g['Blocks']
                if normal(b['TextStart']).startswith(prefix)]
        if len(hits) == 1:
            return hits[0]
    if n.tag == W + 'tbl':
        return 40
    return max(20, math.ceil(len(text(n)) / 35) * 20)


chapter_pages = {c['Page'] for c in native['AfterChapters']}
moves, used = [], set()
for gap in ([] if atomic_only else native['AfterGeometry']):
    page = gap['Page']
    if page < 19 or gap['Gap'] < 100 or page + 1 in chapter_pages or page >= native['AfterPages']:
        continue
    figures = [f for f in native['AfterFigures'] if f['ImagePage'] == page + 1]
    if not figures:
        continue
    f = figures[0]
    label = '图' + f['Bookmark'][3:].replace('_', '-')
    caption = next(i for i, n in enumerate(nodes)
                   if style(n) == 'figurecaption' and text(n).startswith(label + ' '))
    start = caption - 1
    if start in used:
        continue
    cursor, objects, prose, labels = start, [], [], []
    gained, paragraphs = 0, 0
    while cursor < len(nodes) and len(labels) <= 3 and paragraphs < 6:
        n = nodes[cursor]
        if cursor in used or style(n) in ['chapter', 'bibliographytitle'] or n.tag == W + 'sectPr' or n.find(W + 'pPr/' + W + 'sectPr') is not None:
            break
        end = block_end(cursor)
        if end is not None:
            if len(labels) == 3:
                break
            cap = nodes[cursor + 1] if style(n) == 'figure' else n
            labels.append(text(cap).split()[0])
            objects.extend(nodes[cursor:end])
            cursor = end
            continue
        if style(n) in ['h2', 'h3']:
            if any(word in text(n) for word in ['小结', '总结', '局限']):
                break
            # Appendix F explicitly distinguishes independent experiment batches.
            if label.startswith('图F-'):
                break
        prose.append(n)
        gained += height(n)
        if style(n) == 'body' and text(n).strip():
            paragraphs += 1
        cursor += 1
        if paragraphs and gained >= gap['Gap'] - 30:
            break
    # Leave a heading next to its text, never immediately before a delayed figure.
    while prose and style(prose[-1]) in ['h2', 'h3']:
        gained -= height(prose.pop())
        cursor -= 1
    if not paragraphs or gained < 40 or not prose:
        continue
    assert nodes[start:cursor] == sorted(objects + prose, key=nodes.index)
    used.update(range(start, cursor))
    moves.append({'start': start, 'end': cursor, 'objects': objects, 'prose': prose,
                  'labels': labels, 'old_page': page + 1, 'old_gap_points': gap['Gap'],
                  'filling_points': gained})

for move in reversed(moves):
    replacement = move['prose'] + move['objects']
    assert len(replacement) == move['end'] - move['start']
    for n in nodes[move['start']:move['end']]:
        body.remove(n)
    for i, n in enumerate(replacement, move['start']):
        body.insert(i, n)

released = []
for n in body:
    flag = n.find(W + 'pPr/' + W + 'keepLines')
    if style(n) == 'body' and flag is not None and flag.get(W + 'val', '1') != '0' and len(text(n)) > 300:
        flag.set(W + 'val', '0')
        released.append(text(n)[:90])

# An illustration may float past prose, but not between a formula's original
# introductory sentence and the displayed formula or its symbol definition.
reference_path = WORK / '第一步完成.docx'
with ZipFile(reference_path) as z:
    reference = E.fromstring(z.read('word/document.xml')).find(W + 'body')
P_ID = '{http://schemas.microsoft.com/office/word/2010/wordml}paraId'
current_paras = {n.get(P_ID): n for n in body if n.tag == W + 'p' and n.get(P_ID)}


def equation_key(n):
    return tuple((t.tag, t.text or '') for t in n.iter() if t.tag in [W + 't', M + 't'])


current_equations = {equation_key(n): n for n in body if n.tag == W + 'tbl'
                     and n.find('.//' + M + 'oMath') is not None
                     and n.find('.//' + W + 'tblHeader') is None}
atomic_repairs = []
for i, eq in enumerate(reference):
    if eq.tag != W + 'tbl' or eq.find('.//' + M + 'oMath') is None or eq.find('.//' + W + 'tblHeader') is not None or i == 0:
        continue
    intro = reference[i - 1]
    if style(intro) != 'body' or intro.get(P_ID) not in current_paras:
        continue
    if equation_key(eq) not in current_equations:
        continue
    current_intro, current_eq = current_paras[intro.get(P_ID)], current_equations[equation_key(eq)]
    a, b = list(body).index(current_intro), list(body).index(current_eq)
    atoms = [current_eq]
    j = i + 1
    while j < len(reference) and (style(reference[j]) in ['figure', 'figurecaption', 'tablecaption'] or
                                  (reference[j].tag == W + 'tbl' and reference[j].find('.//' + W + 'tblHeader') is not None)):
        j += 1
    if j < len(reference):
        definition = reference[j]
        if style(definition) == 'body' and re.match(r'^(其中|式|公式|该式|上式|只有|第一项|第二项|上述等式|上述不等式)', text(definition)) and definition.get(P_ID) in current_paras:
            atoms.append(current_paras[definition.get(P_ID)])
            b = list(body).index(atoms[-1])
    segment = list(body)[a + 1:b + 1]
    floating = [n for n in segment if n not in atoms]
    if not floating:
        continue
    if not all(style(n) in ['figure', 'figurecaption', 'tablecaption'] or
               (n.tag == W + 'tbl' and n.find('.//' + W + 'tblHeader') is not None) for n in floating):
        continue
    for n in segment:
        body.remove(n)
    for j, n in enumerate(atoms + floating, a + 1):
        body.insert(j, n)
    atomic_repairs.append(text(eq))


def semantic(n):
    return [(t.tag, t.text or '', sorted(t.attrib.items())) for t in n.iter()
            if t.tag in [W + 't', M + 't', W + 'instrText', W + 'fldChar', W + 'tab', W + 'br']]


old_body = before.find(W + 'body')
assert len(old_body) == len(body)
excluded = {'figure', 'figurecaption', 'tablecaption'}
assert [semantic(p) for p in old_body if p.tag == W + 'p' and style(p) not in excluded] == [semantic(p) for p in body if p.tag == W + 'p' and style(p) not in excluded]
for data_table in [False, True]:
    assert [semantic(t) for t in old_body if t.tag == W + 'tbl' and (t.find('.//' + W + 'tblHeader') is not None) == data_table] == [semantic(t) for t in body if t.tag == W + 'tbl' and (t.find('.//' + W + 'tblHeader') is not None) == data_table]
for tag in [M + 'oMath', W + 'sectPr', W + 'bookmarkStart', W + 'bookmarkEnd', W + 'instrText']:
    assert Counter(E.tostring(t) for t in before.iter(tag)) == Counter(E.tostring(t) for t in d.iter(tag)), tag
for s in ['figurecaption', 'tablecaption']:
    assert [text(p) for p in old_body if style(p) == s] == [text(p) for p in body if style(p) == s], s

serialized = E.tostring(d, encoding='utf-8')
root_end = raw.index(b'>', raw.index(b'<w:document')) + 1
serialized_end = serialized.index(b'>') + 1
existing_prefixes = set(re.findall(rb'xmlns:([^=\s]+)=', raw[:root_end]))
extra_namespaces = b''.join(match.group(0) for match in re.finditer(rb' xmlns:([^=\s]+)="[^"]*"', serialized[:serialized_end])
                            if match.group(1) not in existing_prefixes)
parts['word/document.xml'] = raw[:root_end - 1] + extra_namespaces + b'>' + serialized[serialized_end:]
E.fromstring(parts['word/document.xml'])
with ZipFile(DOCX, 'w') as z:
    for info in infos:
        z.writestr(info, parts[info.filename])
report = [{'figures_tables': m['labels'], 'original_page': m['old_page'],
           'gap_points': round(m['old_gap_points'], 1), 'prose_height_points': round(m['filling_points'], 1),
           'last_prose': text(m['prose'][-1])[:90]} for m in moves]
(WORK / ('图表顺延记录_' + suffix + '.json')).write_text(json.dumps({'moves': report, 'long_paragraphs_released': released,
    'formula_intro_definition_adjacency_restored': atomic_repairs,
    'prose_order_table_data_equations_captions_bookmarks_preserved': True}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'moves': report, 'long_paragraphs_released': released, 'formula_groups_restored': atomic_repairs}, ensure_ascii=False, indent=2))
