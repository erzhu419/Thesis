"""Static source inventory; no document edits and no Word automation."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from lxml import etree as E
import json
import re
import unicodedata as U
import pypandoc

A = Path(__file__).resolve().parent
R = A.parents[1]
SOURCE = R / '调整完格式的_括号正体与公式校对修复_20261019.docx'
MD = R / '重构论文_替换库二次修订_20261016.md'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
with ZipFile(SOURCE) as z:
    root = E.fromstring(z.read('word/document.xml'))
tree = root.getroottree()
paragraphs = {p: i for i, p in enumerate(root.iter(W+'p'))}
all_text_operators = Counter()
all_property_operators = Counter()
normal_operators = []
italic_operators = []
first_candidates = []
normal_count = 0
math_style_counts = Counter()
structures = Counter()
named_functions = Counter()

def text(e):
    return ''.join(t.text or '' for t in e.iter() if t.tag in (W+'t', M+'t'))

def on(parent, key):
    e = parent.find(key) if parent is not None else None
    return None if e is None else e.get(W+'val', e.get(M+'val', '1')) not in ('0', 'off', 'false')

def role(run, math):
    return [E.QName(p).localname for p in run.iterancestors() if p is not math and p.tag.startswith(M)]

def operator(c):
    return U.category(c) == 'Sm' or c in '+-*/=<>|&:·'

def props(run):
    wp, mp = run.find(W+'rPr'), run.find(M+'rPr')
    sty = mp.find(M+'sty') if mp is not None else None
    scr = mp.find(M+'scr') if mp is not None else None
    fonts = wp.find(W+'rFonts') if wp is not None else None
    return {'normal_text': on(mp, M+'nor'), 'math_style': sty.get(M+'val') if sty is not None else None,
            'math_script': scr.get(M+'val') if scr is not None else None,
            'direct_italic': on(wp, W+'i'), 'direct_bold': on(wp, W+'b'),
            'direct_italic_cs': on(wp, W+'iCs'), 'direct_bold_cs': on(wp, W+'bCs'),
            'fonts': {E.QName(k).localname: v for k, v in fonts.attrib.items()} if fonts is not None else {}}

inventory = []
for ordinal, math in enumerate(root.iter(M+'oMath'), 1):
    ident = f'M{ordinal:06d}'
    para = next((p for p in math.iterancestors() if p.tag == W+'p'), None)
    row = next((p for p in math.iterancestors() if p.tag == W+'tr'), None)
    cells = row.findall(W+'tc') if row is not None else []
    label = text(cells[-1]).strip() if cells else ''
    label = label[1:-1] if re.fullmatch(r'[（(][1-7A-F]-\d+[a-z]*[）)]', label) else None
    has_delimiter = any(True for _ in math.iter(M+'d'))
    runs = []
    for index, run in enumerate(math.iter(M+'r'), 1):
        value = text(run)
        properties = props(run)
        normal_count += properties['normal_text'] is True
        math_style_counts[properties['math_style'] or '(unset)'] += 1
        operators = Counter(c for c in value if operator(c))
        all_text_operators.update(operators)
        item = {'run': index, 'text': value, 'script_context': role(run, math), **properties}
        runs.append(item)
        loc = {'id': ident, 'ordinal': ordinal, 'label': label, 'run': index,
               'runXPath': tree.getpath(run), 'text': value, 'operator_characters': dict(operators),
               'alignment_marker_only': bool(any(p.tag == M+'eqArr' for p in run.iterancestors()) and re.sub('[\s\u200b]', '', value) == '&'), **properties}
        if operators and properties['normal_text'] is True:
            normal_operators.append(loc)
        if operators and properties['direct_italic'] is True:
            italic_operators.append(loc)
    property_chars = []
    for node in math.iter():
        if node.tag in (M+'chr', M+'sepChr', M+'begChr', M+'endChr'):
            value = node.get(M+'val', '')
            if value:
                property_chars.append({'property': E.QName(node).localname, 'value': value})
                all_property_operators.update(c for c in value if operator(c))
        elif node.tag == M+'nary' and node.find(M+'naryPr/'+M+'chr') is None:
            property_chars.append({'property': 'default_nary_chr', 'value': '∫'})
            all_property_operators['∫'] += 1
        elif node.tag == M+'rad':
            property_chars.append({'property': 'implicit_radical_glyph', 'value': '√'})
            all_property_operators['√'] += 1
        elif node.tag == M+'fName':
            named_functions[re.sub('[\s\u200b\u2061\u2062]', '', text(node))] += 1
    count = Counter(E.QName(n).localname for n in math.iter() if n.tag.startswith(M) and n.tag not in (M+'r', M+'t') and not E.QName(n).localname.endswith('Pr'))
    structures.update(count)
    first = next((x for x in runs if re.sub('[\s\u200b\u2061\u2062]', '', x['text'])), None)
    if first and first['direct_italic'] is True and not has_delimiter:
        first_candidates.append({'id': ident, 'ordinal': ordinal, 'label': label,
                                 'mathXPath': tree.getpath(math), 'first_run': first,
                                 'math_text': text(math), 'normal_text': first['normal_text'],
                                 'contains_only_identifier_characters': all(U.category(c)[0] in ('L', 'N', 'M') or c in "'′_" for c in first['text'] if not c.isspace()),
                                 'contains_identifier_letter': any(U.category(c).startswith('L') for c in first['text']),
                                 'numeric_literal_only': bool(re.fullmatch(r'\d+(?:\.\d+)?', re.sub(r'\s', '', first['text'])))})
    inventory.append({'id': ident, 'ordinal': ordinal, 'mathXPath': tree.getpath(math),
                      'paragraph_index_zero_based': paragraphs.get(para), 'paragraphXPath': tree.getpath(para) if para is not None else None,
                      'paragraph_excerpt': text(para)[:130] if para is not None else None,
                      'numbered_display_label': label, 'contains_native_delimiter': has_delimiter,
                      'text': text(math), 'structure_counts': dict(count), 'structural_glyph_properties': property_chars,
                      'run_count': len(runs), 'normal_text_run_count': sum(x['normal_text'] is True for x in runs),
                      'direct_italic_run_count': sum(x['direct_italic'] is True for x in runs), 'runs': runs})

def walk(x):
    if isinstance(x, dict):
        if x.get('t') == 'Math':
            yield x['c']
        else:
            for value in x.values():
                yield from walk(value)
    elif isinstance(x, list):
        for value in x:
            yield from walk(value)

ast = json.loads(pypandoc.convert_text(MD.read_text(), 'json', format='markdown+tex_math_dollars-smart'))
md = []
for bi, block in enumerate(ast['blocks']):
    for formula in walk(block):
        md.append({'id': f'MD{len(md)+1:06d}', 'block_index_zero_based': bi,
                   'type': formula[0]['t'], 'tex': formula[1], 'labels': re.findall(r'\\tag\{([^}]+)\}', formula[1])})
display = [x for x in inventory if x['numbered_display_label']]
labels = list(dict.fromkeys(x['numbered_display_label'] for x in display))
md_display = [x for x in md if x['type'] == 'DisplayMath']
md_labels = [x['labels'][0] for x in md_display if len(x['labels']) == 1]
glyphs = [{'character': c, 'codepoint': f'U+{ord(c):04X}', 'unicode_name': U.name(c, '(unnamed)'),
           'text_occurrences': all_text_operators[c], 'structural_property_occurrences': all_property_operators[c],
           'normal_text_occurrences': sum(x['operator_characters'].get(c, 0) for x in normal_operators),
           'direct_italic_occurrences': sum(x['operator_characters'].get(c, 0) for x in italic_operators)}
          for c in sorted(set(all_text_operators) | set(all_property_operators), key=ord)]
matching = {'current_word_math_objects': len(inventory), 'MD_display': len(md_display),
            'MD_inline': sum(x['type'] == 'InlineMath' for x in md), 'MD_total': len(md),
            'Word_numbered_display_labels': len(labels), 'Word_numbered_display_objects': len(display),
            'Word_other_inline_or_frontmatter_objects': len(inventory)-len(display),
            'MD_display_labels_same_order_as_Word': md_labels == labels,
            'direct_positional_one_to_one_matching_safe': False,
            'reason': 'Word含MD未纳入的前置材料，且用户将部分长公式分成多个oMath；显示式376标签对应411个Word对象。行内变量重复，不能只按全局序号或相同文字强行匹配。',
            'semantic_reference_rule': '当前Word自身的4738对象及已有D基准canonical是保留用户修改的主对照；MD显示式可按标签聚合后比数学结构，行内需段落上下文对齐。不从MD重建覆盖现稿。'}
summary = {'source': str(SOURCE), 'math_objects': len(inventory), 'run_count': sum(x['run_count'] for x in inventory),
           'normal_text_run_count': normal_count, 'math_style_counts': dict(math_style_counts),
           'operator_character_set_size': len(glyphs), 'normal_text_operator_run_count': len(normal_operators),
           'direct_italic_operator_run_count': len(italic_operators),
           'normal_text_and_direct_italic_operator_runs': sum(x['normal_text'] is True for x in italic_operators),
           'direct_italic_alignment_marker_runs': sum(x['alignment_marker_only'] for x in italic_operators),
           'direct_italic_actual_operator_runs_excluding_alignment': sum(not x['alignment_marker_only'] for x in italic_operators),
           'direct_italic_actual_operator_formula_count': len({x['id'] for x in italic_operators if not x['alignment_marker_only']}),
           'first_direct_italic_run_without_native_delimiter': len(first_candidates),
           'first_normal_text_direct_italic_without_native_delimiter': sum(x['normal_text'] is True for x in first_candidates),
           'first_identifier_only_candidate_count': sum(x['normal_text'] is True and x['contains_only_identifier_characters'] and x['contains_identifier_letter'] for x in first_candidates),
           'first_numeric_literal_count': sum(x['numeric_literal_only'] for x in first_candidates),
           'named_function_counts_from_native_fName': dict(named_functions),
           'math_structure_counts': {k: structures[k] for k in ('f','rad','nary','func','limLow','limUpp','sSub','sSup','sSubSup','sPre','d','m','eqArr') if structures[k]},
           'important_boundary': 'normal_text/direct_italic仅是样式模式计数，不直接认定为错误或实际方框；显示异常由原生Word预览确认。',
           'minimal_native_conversion_rule': '首run直接斜体且无m:d本身不构成显示错误。如原生预览证实仍需转换，先只处理m:nor=1且w:i=1、首个有效运行仅含变量标识字符并至少含一个字母的候选：保留原文字/脚本层级/m:scr，以m:sty=i或bi承接原斜粗体，不重建公式；数值、&对齐标记、普通函数名、中文条件或已显式正体算子不按此规则批量转换。',
           'source_modified': False}
for name, data in [('operator_characters.json', glyphs), ('normal_text_operator_runs.json', normal_operators),
                   ('direct_italic_operator_runs.json', italic_operators), ('first_run_native_candidates.json', first_candidates),
                   ('actual_operator_italic_candidates.json', [x for x in italic_operators if not x['alignment_marker_only']]),
                   ('display_label_mapping.json', [{'label': label, 'Word_ids': [x['id'] for x in display if x['numbered_display_label'] == label], 'MD': next(x for x in md_display if x['labels'] == [label])} for label in labels]),
                   ('md_matchability.json', matching), ('inventory_summary.json', summary)]:
    (A/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
with (A/'all_4738_math_inventory.jsonl').open('w') as f:
    for row in inventory:
        f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
with (A/'md_formula_inventory.jsonl').open('w') as f:
    for row in md:
        f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
print(json.dumps({'summary': summary, 'MD_matchability': matching}, ensure_ascii=False))
