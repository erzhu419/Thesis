"""Inspect the supplied PDF only; export PNG evidence, never create a PDF."""
from pathlib import Path
from io import BytesIO
from collections import defaultdict
import json
import re
import unicodedata
import fitz
from fontTools.ttLib import TTFont

T = Path(__file__).resolve().parent
OUT = T / 'pdf_reported'
OUT.mkdir(exist_ok=True)
SOURCE = Path('/home/erzhu419/mine_code/Thesis/output/Gemini/调整完格式的.pdf')
doc = fitz.open(SOURCE)
# The installed C fast-path crashed with none_dealloc during this PDF scan.
# Use the library's existing Python texttrace device without changing packages.
trace_ns = fitz.Page.get_texttrace.__globals__
mu = trace_ns['mupdf']
def texttrace(page):
    result = []
    device = trace_ns['JM_new_texttrace_device'](result)
    rect = mu.fz_bound_page(page.this)
    device.ptm = mu.FzMatrix(1, 0, 0, -1, 0, rect.y1)
    mu.fz_run_page(page.this, device, mu.FzMatrix(), mu.FzCookie())
    mu.fz_close_device(device)
    return result
cases = {'2-35': (63, (140, 600, 515, 649)), '3-9': (74, (85, 458, 515, 518)), '4-6': (119, (220, 196, 515, 233))}
targets = {}
for label, (page_num, coords) in cases.items():
    page = doc[page_num - 1]
    clip = fitz.Rect(coords)
    page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(OUT / f'page_{page_num:03d}.png')
    page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=clip, alpha=False).save(OUT / f'eq_{label}.png')
    chars = []
    for span in texttrace(page):
        for cp, gid, origin, bbox in span['chars']:
            if fitz.Rect(bbox).intersects(clip):
                chars.append({'char': chr(cp), 'codepoint': f'U+{cp:04X}', 'gid': gid, 'font': span['font'], 'size': span['size'], 'origin': origin, 'bbox': bbox})
    targets[label] = {'physical_page_1based': page_num, 'crop': str(OUT / f'eq_{label}.png'), 'full_page': str(OUT / f'page_{page_num:03d}.png'),
                      'clip_pt': coords, 'visible_gid0': [c for c in chars if c['gid'] == 0 and not c['char'].isspace() and unicodedata.category(c['char']) not in ('Cf', 'Cc')],
                      'membership_chars': [c for c in chars if c['char'] == '∈'],
                      'omega_chars': [c for c in chars if c['char'] == 'ω']}

grouped = defaultdict(lambda: {'count': 0, 'pages': set(), 'nearby_numbered_formulas': set(), 'occurrences': []})
for page_index, page in enumerate(doc):
    traces = texttrace(page)
    bad = [(span, c) for span in traces for c in span['chars']
           if c[1] == 0 and not chr(c[0]).isspace() and unicodedata.category(chr(c[0])) not in ('Cf', 'Cc')]
    if not bad:
        continue
    tags = []
    for label in set(re.findall(r'[（(]([1-7A-F]-\d+[a-z]*)[）)]', page.get_text())):
        for spelling in (f'({label})', f'（{label}）'):
            for rect in page.search_for(spelling):
                if rect.x0 >= 460 and rect.width < 58:
                    tags.append((label, rect))
    for span, (cp, gid, origin, bbox) in bad:
        key = (cp, span['font'])
        g = grouped[key]
        g['count'] += 1
        g['pages'].add(page_index + 1)
        closest = min(tags, key=lambda t: abs((t[1].y0 + t[1].y1) / 2 - origin[1])) if tags else None
        occurrence = {'physical_page_1based': page_index + 1, 'origin_pt': origin, 'bbox_pt': bbox}
        if closest:
            label, rect = closest
            distance = abs((rect.y0 + rect.y1) / 2 - origin[1])
            occurrence.update(nearest_right_margin_equation_label=label, vertical_distance_pt=round(distance, 2))
            if distance < 35:
                g['nearby_numbered_formulas'].add(label)
        g['occurrences'].append(occurrence)

groups = [{'char': chr(cp), 'codepoint': f'U+{cp:04X}', 'unicode_name': unicodedata.name(chr(cp), ''), 'font': font,
           'count': g['count'], 'physical_pages_1based': sorted(g['pages']), 'nearby_numbered_formulas_within_35pt': sorted(g['nearby_numbered_formulas']),
           'occurrences': g['occurrences']} for (cp, font), g in sorted(grouped.items())]
name, ext, kind, font_bytes = doc.extract_font(30)
font = TTFont(BytesIO(font_bytes))
unicode_xref = int(doc.xref_get_key(30, 'ToUnicode')[1].split()[0])
mapping = doc.xref_stream(unicode_xref).decode('ascii', 'replace')
proof = {'font_xref': 30, 'base_font': name, 'subtype': kind, 'encoding': 'Identity-H', 'descendant_font_xref': 32,
         'CIDToGIDMap': 'Identity', 'ToUnicode_xref': unicode_xref,
         'U2208_mapping_lines': [line for line in mapping.splitlines() if '2208' in line.lower()],
         'embedded_font_bytes': len(font_bytes), 'embedded_font_glyph_zero_name': font.getGlyphOrder()[0],
         'embedded_font_cmap_U2208': font.getBestCmap().get(0x2208),
         'interpretation': 'U+2208的PDF源字符码为0000，并经Identity映射绘制glyph0=.notdef；所以可抽取∈，实际却画缺字框。该结论直接来自PDF实际字符轨迹和字体对象，不是仅凭DOCX字体推测。'}
report = {'pdf': str(SOURCE), 'page_count': len(doc), 'read_only': True, 'no_pdf_generated': True,
          'formula_evidence': targets, 'visible_gid0_total': sum(g['count'] for g in groups), 'visible_gid0_groups': groups,
          'font_object_evidence': proof,
          'scope_notes': ['只统计实际PDF gid0可见Unicode字符；跳过Unicode空白、控制/格式字符（含U+200B）。',
                          '邻近式号按右页边编号与纵向距离给出；正文行内缺字也保留真实页码和坐标，不强行归属最近公式。',
                          '式3-9的ω与公式区域未发现gid0，不能把它判为本次∈同类缺字；其实际布局已提供完整crop。',
                          '同一符号在Cambria Math或其他有效字形处不列入修复范围；依照实际gid0字符定位最小字体修复。']}
(OUT / 'pdf_glyph0_diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'visible_gid0_total': report['visible_gid0_total'], 'groups': [{'char': g['char'], 'codepoint': g['codepoint'], 'font': g['font'], 'count': g['count'], 'pages': len(g['physical_pages_1based'])} for g in groups],
                  'target_gid0': {label: [(c['char'], c['codepoint'], c['font']) for c in v['visible_gid0']] for label, v in targets.items()}, 'output': str(OUT / 'pdf_glyph0_diagnosis.json')}, ensure_ascii=False))
