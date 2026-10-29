"""Match the 16 observed PDF .notdef positions to source OMML; no source edits."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json
import re
import fitz
from lxml import etree as E

T = Path(__file__).resolve().parent
P = T / 'pdf_reported/pdf_glyph0_diagnosis.json'
report = json.loads(P.read_text())
source = T / 'source_snapshot_current.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with ZipFile(source) as z:
    root = E.fromstring(z.read('word/document.xml'))
maths = root.xpath('//m:oMath', namespaces=NS)
doc = fitz.open(report['pdf'])
# PDF reading order, matched by the complete local formula and source context.
# (math index, run index, character offset, exact source, expected visual, label)
matches = [
    (480, 7, 0, '∈', '∈', '2-35'),
    (1564, 23, 0, '∈', '∈', '4-5'),
    (1567, 10, 0, '∈', '∈', '4-6'),
    (1570, 2, 0, '∈', '∈', None),
    (1570, 3, 0, 'Z', '𝒵', None),
    (1829, 3, 1, '∈', '∈', '4-27'),
    (1851, 4, 0, '∈', '∈', None),
    (1851, 5, 0, 'Z', '𝒵', None),
    (1852, 0, 1, '∈', '∈', None),
    (1852, 1, 0, 'Z', '𝒵', None),
    (2422, 8, 0, '∈', '∈', '5-5'),
    (2422, 22, 0, '∈', '∈', '5-5'),
    (2453, 1, 0, '∈', '∈', '5-7'),
    (2453, 35, 0, '∈', '∈', '5-7'),
    (4176, 13, 0, '∈', '∈', 'D-4b'),
    (4176, 33, 0, '∈', '∈', 'D-4b'),
]
occurrences = report['visible_gid0_groups'][0]['occurrences']
assert len(matches) == len(occurrences) == 16
for occurrence, (mi, ri, ci, char, visual, label) in zip(occurrences, matches):
    math = maths[mi]
    run = math.xpath('.//m:r', namespaces=NS)[ri]
    text = ''.join(run.xpath('./m:t/text()', namespaces=NS))
    assert text[ci] == char, (mi, ri, text)
    if char == 'Z':
        assert run.xpath('./m:rPr/m:scr[@m:val="script"]', namespaces=NS)
    para = run.xpath('ancestor::w:p[1]', namespaces=NS)[0]
    occurrence['source_OMML'] = {
        'math_index_0based': mi, 'run_index_0based': ri,
        'character_index_in_run': ci, 'run_text': text,
        'source_character': char, 'source_codepoint': f'U+{ord(char):04X}',
        'expected_visual_character': visual,
        'expected_visual_codepoint': f'U+{ord(visual):04X}',
        'script': 'script' if char == 'Z' else None,
        'kind': 'numbered' if label else 'inline', 'exact_equation_label': label,
        'math_text': ''.join(math.xpath('.//m:t/text()', namespaces=NS)),
        'paragraph_text': ''.join(para.xpath('.//w:t/text()|.//m:t/text()', namespaces=NS)),
        'run_xpath': root.getroottree().getpath(run),
    }
    # Exact display-number lines exclude an in-text equation reference at the margin.
    page = doc[occurrence['physical_page_1based'] - 1]
    tags = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            line_text = ''.join(s['text'] for s in line['spans']).strip()
            match = re.fullmatch(r'[（(]([1-7A-F]-\d+[a-z]*)[）)]', line_text)
            if match and line['bbox'][0] > 460:
                tags.append((match.group(1), line['bbox']))
    if tags:
        nearest = min(tags, key=lambda a: abs((a[1][1]+a[1][3])/2-occurrence['origin_pt'][1]))
        occurrence['nearest_right_margin_equation_label'] = nearest[0]
        occurrence['vertical_distance_pt'] = round(abs((nearest[1][1]+nearest[1][3])/2-occurrence['origin_pt'][1]), 2)
    else:
        occurrence.pop('nearest_right_margin_equation_label', None)
        occurrence.pop('vertical_distance_pt', None)
report['visible_gid0_groups'][0]['nearby_numbered_formulas_within_35pt'] = sorted({
    o['nearest_right_margin_equation_label'] for o in occurrences
    if o.get('vertical_distance_pt', 999) < 35
})
report['source_context_mapping'] = {
    'source': str(source), 'source_size': source.stat().st_size,
    'method': 'Match complete formula/inline paragraph context and reading order to actual PDF glyph0 positions; no font-cmap inference replaces PDF evidence.',
    'source_character_counts': dict(Counter(m[3] for m in matches)),
    'intended_visual_counts': dict(Counter(m[4] for m in matches)),
    'exact_numbered_equations': sorted({m[5] for m in matches if m[5]}),
    'inline_math_indices_0based': sorted({m[0] for m in matches if m[5] is None}),
    'note': 'PDF和DOCX保存时间不同；这里只按相同局部公式/段落内容定位源run，不声称PDF由此快照直接导出。',
}
for page_no, clip in [(119, (85, 253, 505, 275)), (129, (85, 90, 505, 119))]:
    output = T / f'pdf_reported/inline_missing_page_{page_no:03d}.png'
    doc[page_no-1].get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(clip), alpha=False).save(output)
report['inline_missing_crops'] = [str(T/f'pdf_reported/inline_missing_page_{p:03d}.png') for p in (119,129)]
report['font_object_evidence']['interpretation'] = 'PDF中字符码0000经Identity映射绘制glyph0=.notdef，ToUnicode又把0000抽取为∈。16个可见缺字位置结合源上下文对应13个∈与3个花体𝒵；文字可抽取∈并不代表其原始符号一定是∈。'
P.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
membership_path = T / 'membership_diagnosis.json'
membership = json.loads(membership_path.read_text())
membership['actual_reported_pdf_evidence'] = {
    'report': str(P), 'visible_gid0_count': 16,
    'pages_1based': [63,119,128,129,159,160,265],
    'source_context_mapping_counts': {'∈': 13, '𝒵': 3},
    'formulas_with_actual_gid0': ['2-35','4-5','4-6','4-27','5-5','5-7','D-4b'],
    'inline_math_indices_0based': [1570,1851,1852],
    'formula_3_9_actual_gid0': 0,
    'interpretation': '实际缺字是16个glyph0；PDF ToUnicode统一抽成U+2208，按源局部上下文区分为13个∈和3个script Z（应显示𝒵）。其余字体cmap缺字候选不等同于PDF实际故障。',
}
membership_path.write_text(json.dumps(membership, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report['source_context_mapping'], ensure_ascii=False))
