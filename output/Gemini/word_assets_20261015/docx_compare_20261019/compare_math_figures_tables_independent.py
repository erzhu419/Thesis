"""Read-only OOXML comparison against the user-selected D-drive baseline."""
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
import difflib
import importlib.util
import json
import re
from lxml import etree as E

R = Path('/home/erzhu419/mine_code/Thesis/output/Gemini')
T = R / 'word_assets_20261015/docx_compare_20261019'
BASE = Path('/mnt/d/Paper/Thesis/Gemini/重构论文_导言小结同步_20261018.docx')
CHECK = R / '调整完格式的.docx'
helper = R / 'word_assets_20261015/exact_library_v2_20261016/prepare_math_verification.py'
spec = importlib.util.spec_from_file_location('settled_math_canonical', helper)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
W, M = h.W, h.M
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
WP = '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}'
RR = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
LABEL = r'\s*[（(]([1-7A-F]-\d+[a-z]*)[）)]\s*'

def compact(s):
    return ''.join(c for c in s if not c.isspace() and c not in '\u200b\u2061\u2062')

def math_content(nodes):
    values = []
    for node in nodes:
        node = deepcopy(node)
        # These ampersands are TeX equation-array alignment markers, not symbols.
        for arr in node.iter(M + 'eqArr'):
            for t in arr.iter(M + 't'):
                t.text = (t.text or '').replace('&', '')
        values.append(h.canon(node))
    return h.sequence(values)

def ancestors(e, tag):
    return next((p for p in e.iterancestors() if p.tag == tag), None)

def numbered_labels(table):
    out = []
    for tr in table.findall(W + 'tr'):
        cells = tr.findall(W + 'tc')
        match = re.fullmatch(LABEL, h.text(cells[-1])) if len(cells) > 1 else None
        if match:
            out.append(match[1])
    return out

def table_cells(table):
    return [[h.text(tc) for tc in tr.findall(W + 'tc')] for tr in table.findall(W + 'tr')]

def table_grid(table):
    return [[{'span': tc.find(W + 'tcPr/' + W + 'gridSpan').get(W + 'val') if tc.find(W + 'tcPr/' + W + 'gridSpan') is not None else '1',
              'vmerge': tc.find(W + 'tcPr/' + W + 'vMerge').get(W + 'val', 'continue') if tc.find(W + 'tcPr/' + W + 'vMerge') is not None else None}
             for tc in tr.findall(W + 'tc')] for tr in table.findall(W + 'tr')]

def load(path):
    with ZipFile(path) as z:
        root = E.fromstring(z.read('word/document.xml'))
        rels = {r.get('Id'): r.get('Target') for r in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        media = {n: z.read(n) for n in z.namelist() if n.startswith('word/media/')}
        other_math = {}
        for n in z.namelist():
            if n.startswith('word/') and n.endswith('.xml') and n != 'word/document.xml':
                data = z.read(n)
                if b'oMath' in data:
                    part = E.fromstring(data)
                    if list(part.iter(M + 'oMath')):
                        other_math[n] = list(part.iter(M + 'oMath'))
    body = root.find(W + 'body')
    figcaps, tablecaps, tables, cover_tables, draws = [], [], [], [], []
    last_table_caption = ''
    for bi, block in enumerate(body):
        text = h.text(block)
        if block.tag == W + 'p':
            sid = block.find(W + 'pPr/' + W + 'pStyle')
            sid = sid.get(W + 'val') if sid is not None else ''
            if sid in ('figurecaption', 'affc') and re.match(r'^图\d|^图[A-F]', text):
                figcaps.append({'body_index': bi, 'text': text, 'style': sid})
            if sid == 'tablecaption':
                tablecaps.append({'body_index': bi, 'text': text, 'style': sid})
                last_table_caption = text
        if block.tag == W + 'tbl' and not numbered_labels(block):
            cells = table_cells(block)
            record = {'body_index': bi, 'caption': last_table_caption or '主要符号表', 'node': block}
            if not last_table_caption and cells and compact(cells[0][0]) != '符号':
                cover_tables.append({'body_index': bi, 'cells': cells})
            else:
                tables.append(record)
        for d in block.iter(W + 'drawing'):
            ref = [{'id': b.get(RR + 'embed'), 'target': rels.get(b.get(RR + 'embed'))} for b in d.iter(A + 'blip')]
            draws.append({'body_index': bi, 'reference': ref,
                          'extent': [dict(e.attrib) for e in d.iter(WP + 'extent')],
                          'image_extent': [dict(e.attrib) for e in d.iter(A + 'ext') if 'cx' in e.attrib],
                          'crop': [dict(e.attrib) for e in d.iter(A + 'srcRect')],
                          'placement': [E.QName(e).localname for e in d],
                          'transform': [dict(e.attrib) for e in d.iter(A + 'xfrm')]})
    return dict(root=root, body=body, media=media, other_math=other_math, figures=figcaps, captions=tablecaps, tables=tables, cover_tables=cover_tables, drawings=draws)

old, new = load(BASE), load(CHECK)
old_rows, new_rows = h.word_rows(BASE), h.word_rows(CHECK)
formula_checks, formula_changes = [], []
for a, b in zip(old_rows, new_rows):
    raw_a = [E.tostring(x) for x in a['maths']]
    raw_b = [E.tostring(x) for x in b['maths']]
    match = math_content(a['maths']) == math_content(b['maths'])
    record = {'label': a['label'], 'ordinal': a['ordinal'], 'same_label': a['label'] == b['label'],
              'content_and_math_structure_equal': match, 'old_oMath_count': len(a['maths']), 'new_oMath_count': len(b['maths'])}
    formula_checks.append(record)
    if raw_a != raw_b:
        record = dict(record, kind='layout_run_or_format_change' if match else 'math_content_or_structure_change',
                      old_text=''.join(h.text(m) for m in a['maths']), new_text=''.join(h.text(m) for m in b['maths']))
        if not match:
            record.update(old_content=math_content(a['maths']), new_content=math_content(b['maths']))
        formula_changes.append(record)

old_maths, new_maths = list(old['root'].iter(M + 'oMath')), list(new['root'].iter(M + 'oMath'))
old_keys, new_keys = [math_content([m]) for m in old_maths], [math_content([m]) for m in new_maths]
math_stream_changes = []
new_ps = list(new['root'].iter(W + 'p'))
for op, i, j, k, l in difflib.SequenceMatcher(None, old_keys, new_keys, autojunk=False).get_opcodes():
    if op == 'equal':
        continue
    a, b = old_maths[i:j], new_maths[k:l]
    rec = {'operation': op, 'old_oMath_range_0based': [i, j], 'new_oMath_range_0based': [k, l],
           'old_texts': [h.text(m) for m in a], 'new_texts': [h.text(m) for m in b],
           'joined_math_structure_equal': math_content(a) == math_content(b)}
    if a:
        tr = ancestors(a[0], W + 'tr')
        cells = tr.findall(W + 'tc') if tr is not None else []
        label = re.fullmatch(LABEL, h.text(cells[-1])) if cells else None
        rec['label'] = label[1] if label else None
    if op == 'delete':
        conversions = []
        for m in a:
            p = ancestors(m, W + 'p')
            before = h.text(p)
            candidates = [q for q in new_ps if compact(h.text(q)).startswith(compact(before)[:60])]
            source_expression = h.text(m)
            plain_texts = [h.text(q) for q in candidates]
            if re.search(r'\d\s*s$', source_expression):
                value = source_expression.strip().rsplit('s', 1)[0].strip()
                matches = [re.search(re.escape(value) + r'\s*(?:"\s*s"|s\b|秒)', text) for text in plain_texts]
                literal = next((match[0].strip() for match in matches if match), None)
                extra_quotes = literal is not None and '"' in literal
                kept = literal is not None
            else:
                literal = source_expression if any(compact(source_expression) in compact(text) for text in plain_texts) else None
                extra_quotes = False
                kept = literal is not None
            conversions.append({'old_expression': source_expression, 'new_plain_expression': literal,
                                'value_and_unit_present_in_new_plain_text': kept,
                                'extra_literal_unit_quotes': extra_quotes,
                                'old_paragraph': before, 'new_paragraphs': [h.text(q) for q in candidates],
                                'new_native_math_counts': [len(list(q.iter(M + 'oMath'))) for q in candidates]})
        rec.update(kind='native_inline_math_to_plain_text', conversions=conversions,
                   all_values_and_units_preserved=all(c['value_and_unit_present_in_new_plain_text'] for c in conversions),
                   extra_literal_unit_quote_count=sum(c['extra_literal_unit_quotes'] for c in conversions))
    else:
        rec['kind'] = 'layout_math_segmentation' if rec['joined_math_structure_equal'] else 'math_content_or_structure_change'
    math_stream_changes.append(rec)

figure_changes, figure_checks = [], []
for i, (a, b) in enumerate(zip(old['drawings'], new['drawings'])):
    cap_a, cap_b = old['figures'][i], new['figures'][i]
    target_a, target_b = [r['target'] for r in a['reference']], [r['target'] for r in b['reference']]
    equal = target_a == target_b
    rec = {'ordinal': i + 1, 'figure': re.match(r'图([1-7A-F]-\d+)', cap_a['text'])[1],
           'media_targets_equal': equal, 'image_bytes_equal': all(old['media'].get('word/' + t) == new['media'].get('word/' + t) for t in target_a),
           'caption_text_equal': cap_a['text'] == cap_b['text']}
    figure_checks.append(rec)
    changes = {field: {'before': a[field], 'after': b[field]} for field in ('reference', 'extent', 'image_extent', 'crop', 'placement', 'transform') if a[field] != b[field]}
    if cap_a['text'] != cap_b['text'] or cap_a['style'] != cap_b['style']:
        changes['caption'] = {'before': cap_a, 'after': cap_b}
    if changes:
        figure_changes.append(dict(rec, changes=changes))

table_checks, table_changes = [], []
numeric = re.compile(r'[+\-−]?\d[\d,]*(?:\.\d+)?(?:[eE][+\-]?\d+)?')
new_table_map = {t['caption']: t for t in new['tables']}
assert len(new_table_map) == len(new['tables'])
for i, a in enumerate(old['tables']):
    b = new_table_map[a['caption']]
    cells_a, cells_b = table_cells(a['node']), table_cells(b['node'])
    raw_equal = cells_a == cells_b
    whitespace_equal = [[compact(v) for v in row] for row in cells_a] == [[compact(v) for v in row] for row in cells_b]
    numbers_equal = [[numeric.findall(v) for v in row] for row in cells_a] == [[numeric.findall(v) for v in row] for row in cells_b]
    grid_equal = table_grid(a['node']) == table_grid(b['node'])
    record = {'ordinal': i + 1, 'caption': a['caption'], 'old_body_index': a['body_index'], 'new_body_index': b['body_index'],
              'row_cell_grid_equal': grid_equal, 'cell_text_exact_equal': raw_equal, 'cell_text_equal_ignoring_whitespace': whitespace_equal,
              'numeric_tokens_per_cell_equal': numbers_equal}
    table_checks.append(record)
    diffs = []
    for ri, (ar, br) in enumerate(zip(cells_a, cells_b)):
        for ci, (av, bv) in enumerate(zip(ar, br)):
            if av != bv:
                diffs.append({'row_1based': ri + 1, 'cell_1based': ci + 1, 'before': av, 'after': bv,
                              'kind': 'whitespace_only' if compact(av) == compact(bv) else 'text_or_math_change',
                              'numeric_tokens_equal': numeric.findall(av) == numeric.findall(bv)})
    if not raw_equal or not grid_equal:
        table_changes.append(dict(record, changed_cells=diffs))

caption_changes = []
for kind, left, right in [('figure', old['figures'], new['figures']), ('table', old['captions'], new['captions'])]:
    for i, (a, b) in enumerate(zip(left, right), 1):
        if a['text'] != b['text'] or a['style'] != b['style']:
            caption_changes.append({'kind': kind, 'ordinal': i, 'before': a, 'after': b})

report = {
    'baseline': str(BASE), 'baseline_authority': '用户最终指定D盘版；WSL试算不纳入本报告', 'compared': str(CHECK), 'read_only': True,
    'method': '直接比较ZIP媒体字节（未哈希）；编号公式逐号及全部oMath有序内容规范树；真实表格逐行逐单元格文本、数值和合并结构；图像引用目标、尺寸、裁切及图表题。',
    'counts': {'old_numbered_formulas': len(old_rows), 'new_numbered_formulas': len(new_rows), 'old_oMath': len(old_maths), 'new_oMath': len(new_maths),
               'old_figures': len(old['drawings']), 'new_figures': len(new['drawings']), 'old_figure_captions': len(old['figures']), 'new_figure_captions': len(new['figures']),
               'old_table_captions': len(old['captions']), 'new_table_captions': len(new['captions']), 'old_real_tables_including_symbols': len(old['tables']), 'new_real_tables_including_symbols': len(new['tables'])},
    'all_376_numbered_formula_content_and_structure_equal': len(old_rows) == len(new_rows) == 376 and all(r['same_label'] and r['content_and_math_structure_equal'] for r in formula_checks),
    'numbered_formula_checks': formula_checks, 'numbered_formula_xml_changes': formula_changes,
    'all_math_stream_changes': math_stream_changes, 'other_word_parts_with_math': {'before': {n: len(m) for n, m in old['other_math'].items()}, 'after': {n: len(m) for n, m in new['other_math'].items()}},
    'all_53_media_bytes_equal': old['media'].keys() == new['media'].keys() and all(old['media'][n] == new['media'][n] for n in old['media']),
    'figure_checks': figure_checks, 'figure_differences': figure_changes, 'caption_differences': caption_changes,
    'table_checks': table_checks, 'table_differences': table_changes,
    'front_matter_unlabelled_tables': {'before': old['cover_tables'], 'after': new['cover_tables']},
    'all_real_table_numeric_tokens_and_grid_equal': len(old['tables']) == len(new['tables']) and all(t['numeric_tokens_per_cell_equal'] and t['row_cell_grid_equal'] for t in table_checks),
}
T.mkdir(parents=True, exist_ok=True)
(T / 'math_figures_tables_independent.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'counts': report['counts'], 'numbered_formulas_equal': report['all_376_numbered_formula_content_and_structure_equal'],
                  'math_stream_changes': len(math_stream_changes), 'media_bytes_equal': report['all_53_media_bytes_equal'],
                  'figure_differences': len(figure_changes), 'caption_differences': len(caption_changes),
                  'table_differences': len(table_changes), 'table_numeric_grid_equal': report['all_real_table_numeric_tokens_and_grid_equal']}, ensure_ascii=False))
