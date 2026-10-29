"""Read-only Word text/notation inventory with native-math object locations."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json
import re

A = Path(__file__).resolve().parent
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
ast = json.loads((A / 'source_ast.json').read_text())

def walk(obj):
    if isinstance(obj, dict):
        if obj.get('t') == 'Math':
            yield obj['c'][1]
        else:
            for value in obj.values():
                yield from walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value)

tex = list(walk(ast))
with ZipFile(A / 'source_snapshot.docx') as z:
    root = E.fromstring(z.read('word/document.xml'))
maths = list(root.iter(M + 'oMath'))
assert len(tex) == len(maths)
math_id = {m: i for i, m in enumerate(maths, 1)}
tables = list(root.iter(W + 'tbl'))
table_id = {t: i for i, t in enumerate(tables, 1)}

def plain(node):
    return ''.join(n.text or '' for n in node.iter() if n.tag in (W + 't', M + 't'))

def rendered(node):
    if node.tag == M + 'oMath':
        return '$' + tex[math_id[node] - 1] + '$'
    if node.tag == W + 't':
        return node.text or ''
    if node.tag in (W + 'tab', W + 'br'):
        return ' '
    return ''.join(rendered(child) for child in node)

paragraphs, formulas, groups = [], [], {}
chapter, section = '前置页', '前置页'
for ident, p in enumerate(root.iter(W + 'p'), 1):
    style = p.find(W + 'pPr/' + W + 'pStyle')
    style = style.get(W + 'val', '') if style is not None else ''
    raw, text = plain(p), rendered(p)
    if style == '1':
        chapter = raw.strip()
        section = chapter
    elif style in ('2', '3'):
        section = raw.strip()
    # Body headings in the inherited template are also plain paragraphs.
    elif re.match(r'^[2-7]\.\d+(?:\.\d+)?\s+[^=]', raw) and len(raw) < 100:
        section = raw.strip()
    table = next(p.iterancestors(W + 'tbl'), None)
    row = next(p.iterancestors(W + 'tr'), None)
    label = None
    if row is not None:
        last = plain(row.findall(W + 'tc')[-1]).strip()
        if re.fullmatch(r'[（(][1-7A-G]-\d+[a-z]*[）)]', last):
            label = last.strip('()（）')
    indices = [math_id[m] for m in p.iter(M + 'oMath')]
    item = {'paragraph': ident, 'chapter': chapter, 'section': section,
            'style': style, 'formula_number': label, 'math_ids': indices,
            'text': text, 'raw_text': raw, 'table': table_id.get(table)}
    paragraphs.append(item)
    groups.setdefault(chapter, []).append(item)
    for index in indices:
        formulas.append({'math_id': index, 'paragraph': ident, 'chapter': chapter,
                         'section': section, 'formula_number': label, 'tex': tex[index - 1]})

symbol_title = next(p for p in root.iter(W + 'p') if plain(p) == '主要符号说明')
symbol_table = next(t for t in tables if symbol_title.sourceline is None or
                    len(t.findall(W + 'tr')) > 80)
symbols = []
for row_index, row in enumerate(symbol_table.findall(W + 'tr')):
    cells = row.findall(W + 'tc')
    if row_index == 0:
        continue
    symbols.append({'row': row_index, 'symbol': rendered(cells[0]),
                    'definition': rendered(cells[1]),
                    'math_ids': [math_id[m] for m in cells[0].iter(M + 'oMath')]})

for filename, data in [('paragraphs.json', paragraphs), ('formulas.json', formulas),
                       ('symbol_table.json', symbols)]:
    (A / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
for chapter, rows in groups.items():
    name = re.sub(r'[^\w\u4e00-\u9fff-]', '_', chapter)[:90]
    lines = [f'# {chapter}']
    for row in rows:
        if row['text'].strip():
            loc = f"P{row['paragraph']:04d}"
            if row['formula_number']:
                loc += f" 式({row['formula_number']})"
            if row['math_ids']:
                loc += ' ' + ','.join(f'M{i:06d}' for i in row['math_ids'])
            lines.extend([f'\n[{loc}] {row["text"]}'])
    (A / (name + '.md')).write_text('\n'.join(lines), encoding='utf-8')
(A / '主要符号表.md').write_text('\n'.join(
    ['|行|符号|含义|', '|---|---|---|'] +
    [f"|{s['row']}|{s['symbol'].replace('|', '∣')}|{s['definition'].replace('|', '∣')}|" for s in symbols]), encoding='utf-8')
print(json.dumps({'math_objects': len(formulas), 'paragraphs': len(paragraphs),
                  'symbol_rows': len(symbols), 'chapter_files': list(groups)}, ensure_ascii=False))
