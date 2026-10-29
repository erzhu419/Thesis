"""Read-only character and font diagnosis for 4-6, 2-35, and 3-9."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter, defaultdict
import importlib.util
import json
import re
import unicodedata

R = Path('/home/erzhu419/mine_code/Thesis/output/Gemini')
T = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('math_helper', R / 'word_assets_20261015/exact_library_v2_20261016/prepare_math_verification.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
files = {'latest': R / '调整完格式的.docx', 'D_baseline': Path('/mnt/d/Paper/Thesis/Gemini/重构论文_导言小结同步_20261018.docx')}
labels = ('4-6', '2-35', '3-9')
rows = {name: h.word_rows(path) for name, path in files.items()}

def props(run):
    fonts = [{E.QName(k).localname: v for k, v in node.attrib.items()} for node in run.findall(h.W + 'rPr/' + h.W + 'rFonts')]
    return {'m_nor': run.find(h.M + 'rPr/' + h.M + 'nor') is not None,
            'm_scr': [n.get(h.M + 'val') for n in run.findall(h.M + 'rPr/' + h.M + 'scr')],
            'm_sty': [n.get(h.M + 'val') for n in run.findall(h.M + 'rPr/' + h.M + 'sty')],
            'w_italic': run.find(h.W + 'rPr/' + h.W + 'i') is not None,
            'run_fonts': fonts}

def runs_of(row):
    return [r for m in row['maths'] for r in m.iter(h.M + 'r')]

target = []
for label in labels:
    pair = {name: next(r for r in rws if r['label'] == label) for name, rws in rows.items()}
    item = {'label': label, 'math_content_and_structure_equal': h.content(pair['latest']['maths']) == h.content(pair['D_baseline']['maths']), 'versions': {}}
    for name, row in pair.items():
        rr = runs_of(row)
        selected = []
        for run in rr:
            txt = h.text(run)
            if any(c in txt for c in ('∈', '口', '□', 'ω', 'Ψ')) or txt == 'Z':
                selected.append({'text': txt, 'codepoints': ['U+%04X' % ord(c) for c in txt], 'parent': E.QName(run.getparent()).localname, **props(run)})
        text = ''.join(h.text(m) for m in row['maths'])
        item['versions'][name] = {'text': text, 'U2208_count': text.count('∈'), 'U53E3_count': text.count('口'), 'U25A1_count': text.count('□'),
                                  'U200B_count': text.count('\u200b'), 'special_runs': selected,
                                  'spacing_char_counts': {'U+%04X' % ord(c): text.count(c) for c in set(text) if unicodedata.category(c).startswith('Z') or c == '\u200b'}}
    target.append(item)

whole = {}
for name, path in files.items():
    with ZipFile(path) as z:
        root = E.fromstring(z.read('word/document.xml'))
        styles = E.fromstring(z.read('word/styles.xml'))
        settings = E.fromstring(z.read('word/settings.xml'))
    membership = []
    symbols = defaultdict(lambda: {'count': 0, 'normal_text_count': 0, 'normal_text_without_explicit_cambria_math_count': 0, 'font_settings': Counter(), 'numbered_labels': set()})
    for math in root.iter(h.M + 'oMath'):
        tr = next((a for a in math.iterancestors() if a.tag == h.W + 'tr'), None)
        cells = tr.findall(h.W + 'tc') if tr is not None else []
        match = re.fullmatch(r'\s*[（(]([1-7A-F]-\d+[a-z]*)[）)]\s*', h.text(cells[-1])) if cells else None
        label = match[1] if match else None
        for run in math.iter(h.M + 'r'):
            text, pr = h.text(run), props(run)
            fontkey = json.dumps(pr['run_fonts'], ensure_ascii=False, sort_keys=True)
            explicit_cambria = any(v == 'Cambria Math' for f in pr['run_fonts'] for k, v in f.items() if k in ('ascii', 'hAnsi'))
            if '∈' in text:
                p = next((a for a in run.iterancestors() if a.tag == h.W + 'p'), None)
                membership.append({'numbered_label': label, 'text': text, 'count': text.count('∈'),
                                   'context': h.text(p)[:220] if p is not None else '', **pr})
            for c in text:
                if ord(c) < 128 or not (unicodedata.category(c) == 'Sm' or 0x370 <= ord(c) <= 0x3ff):
                    continue
                d = symbols[c]
                d['count'] += 1
                d['normal_text_count'] += int(pr['m_nor'])
                d['normal_text_without_explicit_cambria_math_count'] += int(pr['m_nor'] and not explicit_cambria)
                d['font_settings'][fontkey] += 1
                if label:
                    d['numbered_labels'].add(label)
    symbol_data = [{'symbol': c, 'codepoint': 'U+%04X' % ord(c), 'unicode_name': unicodedata.name(c, ''),
                    **{k: v for k, v in d.items() if k not in ('font_settings', 'numbered_labels')},
                    'font_settings': [{'run_fonts': json.loads(k), 'count': v} for k, v in d['font_settings'].items()],
                    'numbered_labels': sorted(d['numbered_labels'])} for c, d in symbols.items()]
    body_style = next(s for s in styles.findall(h.W + 'style') if s.get(h.W + 'styleId') == 'body')
    whole[name] = {'membership_U2208_total': sum(e['count'] for e in membership),
                   'membership_U2208_normal_text_total': sum(e['count'] for e in membership if e['m_nor']),
                   'membership_numbered_labels': list(dict.fromkeys(e['numbered_label'] for e in membership if e['numbered_label'])),
                   'membership_occurrences': membership,
                   'other_nonASCII_math_and_Greek_symbol_font_inventory': symbol_data,
                   'body_style_fonts': [{E.QName(k).localname: v for k, v in e.attrib.items()} for e in body_style.findall(h.W + 'rPr/' + h.W + 'rFonts')],
                   'document_math_font': [e.get(h.M + 'val') for e in settings.iter(h.M + 'mathFont')]}

source = R / '重构论文_替换库二次修订_20261016.md'
md = [e for e in h.md_entries(source) if e['label'] in labels]
report = {'read_only': True, 'files': {k: str(v) for k, v in files.items()}, 'MD': {'path': str(source), 'formulas': md},
          'target_formulas': target,
          'conclusion': [
              '4-6与2-35存储的成员符号都为U+2208；两式中没有U+53E3口或U+25A1方框，数学内容与基准一致。',
              '4-6的∈从Cambria Math数学正体run改为m:nor普通文本且无显式ascii/hAnsi字体；正文样式继承Times New Roman与宋体，实际字形选取不再由原数学字体run锁定。花体Z保留m:scr=script及Cambria Math，不是Z字符被换。',
              '2-35的∈在最新版被显式指定宋体ascii/hAnsi/cs并启用m:nor；旧版是Cambria Math和m:sty=p。',
              '3-9没有成员符号∈；ω仍是U+03C9，变为m:nor、无显式Cambria Math，且包含U+2001/U+2005/U+2009等数学间距。其用户所见异常不能归为∈字符问题。',
              '用户已确认4-6在PDF中的方框为实际显示故障；XML证明字符未替换，故应修数学run模式/符号字体链，不应把口替换成∈或改公式内容。'
          ],
          'rendering_boundary': '全文清点仅提供同模式、同字体配置的潜在影响范围，未把XML中存在符号认定为已目见缺字；具体实际PDF字体及fallback需原生渲染或PDF字体映射确认。E-46用户已确认PDF正常，单列为编辑占位问题，不与∈打印缺字合并。',
          'whole_document_inventory': whole,
          'repair_scope_recommendation': '优先在临时副本将4-6/2-35的∈恢复Cambria Math数学正体run并原生验证；3-9按实际异常位置修对应ω/间距或run模式。保留所有中文、编号及数学符号/结构。未修改文件。'}
(T / 'membership_diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'target_math_unchanged': {x['label']: x['math_content_and_structure_equal'] for x in target},
                  'U2208_counts': {k: [v['membership_U2208_total'], v['membership_U2208_normal_text_total']] for k, v in whole.items()},
                  'latest_membership_numbered_labels': whole['latest']['membership_numbered_labels'], 'output': str(T / 'membership_diagnosis.json')}, ensure_ascii=False))
