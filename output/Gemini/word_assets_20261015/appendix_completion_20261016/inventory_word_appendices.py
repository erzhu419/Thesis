"""独立只读盘点附录A–F，为Word到Markdown补全提供验证目标。"""
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from xml.etree import ElementTree as E
import json
import re

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parents[1] / '重构论文_替换库修订_20261015.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
P = '{http://schemas.openxmlformats.org/package/2006/relationships}'
LABEL = re.compile(r'[（(]((?:\d+|[A-Z])-\d+[a-z]?)[）)]')


def text(element):
    return ''.join(n.text or '' for n in element.iter() if n.tag in (W + 't', M + 't'))


def style(element):
    n = element.find(W + 'pPr/' + W + 'pStyle')
    return n.get(W + 'val', '') if n is not None else ''


def equation_labels(table):
    result = []
    for row in table.findall(W + 'tr'):
        cells = row.findall(W + 'tc')
        if len(cells) < 2:
            continue
        raw = re.sub(r'\s+', '', text(cells[-1]))
        if LABEL.findall(raw) and not LABEL.sub('', raw):
            result.extend(LABEL.findall(raw))
    return result


with ZipFile(SOURCE) as archive:
    root = E.fromstring(archive.read('word/document.xml'))
    rels = E.fromstring(archive.read('word/_rels/document.xml.rels'))
relationships = {r.get('Id'): r.get('Target') for r in rels.findall(P + 'Relationship')}
body = list(root.find(W + 'body'))
boundaries = [(re.match(r'^附录([A-G])\s', text(b))[1], n, text(b)) for n, b in enumerate(body) if style(b) == 'chapter' and re.match(r'^附录([A-G])\s', text(b))]
number_keys = {number: key for key, number in json.loads((BASE.parent / 'terminal_cleanup_20261015/reference_numbers_final.json').read_text()).items()}
reference_texts = {}
for block in body:
    for bookmark in block.iter(W + 'bookmarkStart'):
        match = re.fullmatch(r'latest_ref_(\d+)', bookmark.get(W + 'name', ''))
        if match:
            reference_texts[int(match[1])] = text(block)

appendices = []
for offset, (letter, start, title) in enumerate(boundaries[:-1]):
    end = boundaries[offset + 1][1]
    if letter not in 'ABCDEF':
        continue
    blocks = body[start:end]
    labels, equation_tables, ordinary_tables, figures, hints, citations, plain_brackets, crossrefs = [], [], [], [], [], [], [], []
    for index in range(start, end):
        block = body[index]
        raw = text(block)
        hint = re.match(r'^([A-F]\.\d+)\s+([^。]+。)', raw)
        if hint:
            hints.append({'id': hint[1], 'lead': hint[0], 'word_block': index, 'paragraph_style': style(block)})
        if block.tag == W + 'tbl':
            tags = equation_labels(block)
            if tags:
                labels.extend(tags)
                equation_tables.append({'word_block': index, 'labels': tags, 'native_display_math_count': sum(1 for _ in block.iter(M + 'oMathPara')), 'math_text': ''.join(n.text or '' for n in block.iter(M + 't'))})
            else:
                rows = [[text(cell) for cell in row.findall(W + 'tc')] for row in block.findall(W + 'tr')]
                ordinary_tables.append({'word_block': index, 'row_count': len(rows), 'column_counts': [len(row) for row in rows], 'rows': rows})
        for blip in block.iter(A + 'blip'):
            rid = blip.get(R + 'embed')
            figures.append({'word_block': index, 'relationship': rid, 'docx_media': str(PurePosixPath('word') / relationships[rid]) if rid in relationships else None})
        for link in block.iter(W + 'hyperlink'):
            match = re.fullmatch(r'latest_ref_(\d+)', link.get(W + 'anchor', ''))
            if match:
                number = int(match[1])
                citations.append({'word_block': index, 'number': number, 'display': text(link), 'anchor': link.get(W + 'anchor'), 'key': number_keys.get(number), 'reference': reference_texts.get(number), 'context': raw})
        plain = ''.join(n.text or '' for n in block.iter(W + 't'))
        for match in re.finditer(r'\[([1-9]\d{0,2})\]', plain):
            plain_brackets.append({'word_block': index, 'number': int(match[1]), 'plain_text': plain})
        for instruction in block.iter(W + 'instrText'):
            match = re.search(r'\b(?:REF|PAGEREF)\s+(\S+)', instruction.text or '')
            if match:
                crossrefs.append({'word_block': index, 'instruction': instruction.text, 'target': match[1]})
    paragraphs = [p for block in blocks for p in block.iter(W + 'p')]
    appendix = {'letter': letter, 'title': title, 'word_block_start_inclusive': start, 'word_block_end_exclusive': end, 'top_level_block_count': len(blocks), 'top_level_paragraph_count': sum(b.tag == W + 'p' for b in blocks), 'all_paragraph_count_including_cells': len(paragraphs), 'nonempty_paragraph_count_including_cells': sum(bool(text(p).strip()) for p in paragraphs), 'all_native_math_count': sum(1 for b in blocks for _ in b.iter(M + 'oMath')), 'native_display_math_count': sum(1 for b in blocks for _ in b.iter(M + 'oMathPara')), 'equation_table_count': len(equation_tables), 'equation_label_count': len(labels), 'equation_labels': labels, 'equations': equation_tables, 'ordinary_table_count': len(ordinary_tables), 'ordinary_tables': ordinary_tables, 'figure_count': len(figures), 'figures': figures, 'captions': [{'word_block': index, 'style': style(body[index]), 'text': text(body[index])} for index in range(start, end) if style(body[index]) in ('figurecaption', 'tablecaption')], 'hint_count': len(hints), 'hints': hints, 'citation_link_count': len(citations), 'citations': citations, 'numeric_brackets_in_plain_runs': plain_brackets, 'REF_fields': crossrefs}
    appendices.append(appendix)

totals = {key: sum(a[key] for a in appendices) for key in ['top_level_block_count', 'top_level_paragraph_count', 'all_paragraph_count_including_cells', 'nonempty_paragraph_count_including_cells', 'all_native_math_count', 'native_display_math_count', 'equation_table_count', 'equation_label_count', 'ordinary_table_count', 'figure_count', 'hint_count', 'citation_link_count']}
targets = ['附录A–F各出现一次，顺序位于附录G之前；现有MD字符与章节不改。', '50条A.1–F.4段内提示的定位号与标题文字按Word保留，不重编号。', '全部234个附录显示公式标签逐项/顺序保留；原生数学对象为转换来源，不以扁平math_text替代公式。', 'A-3（Word块1397）和C-9（块1669）各含两个显示数学节点共用一个标签；两个表达式都保留，236显示节点不能只按标签数取234个。', '两张附录F数据表逐单元格保留（含空白、单位、区间和正负号）。', '三张附录F图片与图题逐一对应，图片使用本版Word中的媒体资源。', '四处Word文献引用的号、key与原书目身份保持对应；附录内部REF的目标按原定位保留。', '段内普通序号、定理/引理/证明标识和段落顺序沿用Word，不润色或合并重排。']
report = {'source_word': str(SOURCE), 'index_convention': '主文w:body子元素0基；start含、end不含，包含公式表与空段', 'appendix_G_boundary': next(n for letter, n, _ in boundaries if letter == 'G'), 'totals': totals, 'validation_targets': targets, 'appendices': appendices}
(BASE / 'word_appendix_inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
lines = ['# 附录A–F只读盘点', '', '| 附录 | Word块范围（0基，右端不含） | 主文段落 | 式表/标签 | 数据表/图 | 段内提示 | 文献链接 |', '| --- | --- | --- | --- | --- | --- | --- |']
for a in appendices:
    lines.append(f"| {a['letter']} | {a['word_block_start_inclusive']}–{a['word_block_end_exclusive']} | {a['top_level_paragraph_count']} | {a['equation_table_count']}/{a['equation_label_count']} | {a['ordinary_table_count']}/{a['figure_count']} | {a['hint_count']} | {a['citation_link_count']} |")
lines += ['', f"合计{totals['top_level_block_count']}个主文块、{totals['equation_label_count']}个附录公式标签、{totals['hint_count']}条段内提示、{totals['ordinary_table_count']}张数据表、{totals['figure_count']}张图、{totals['citation_link_count']}处文献链接。", '', '验证目标：', '']
lines += [f'- {target}' for target in targets]
lines += ['', '引用身份：', '']
for a in appendices:
    for citation in a['citations']:
        lines.append(f"- 附录{a['letter']}，Word块{citation['word_block']}：[{citation['number']}] → {citation['key']}")
lines += ['', '完整50条提示、234公式标签、逐单元格表格及图片资源路径见 word_appendix_inventory.json。', '']
(BASE / 'word_appendix_inventory.md').write_text('\n'.join(lines))
print(json.dumps({'totals': totals, 'by_appendix': [{k: a[k] for k in ['letter', 'top_level_block_count', 'equation_label_count', 'ordinary_table_count', 'figure_count', 'hint_count', 'citation_link_count']} for a in appendices]}, ensure_ascii=False))
