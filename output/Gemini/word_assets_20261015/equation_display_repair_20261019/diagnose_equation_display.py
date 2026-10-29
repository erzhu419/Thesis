"""Read-only E-46 display diagnosis; no DOCX or Markdown is modified."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import importlib.util
import json
import re

R = Path('/home/erzhu419/mine_code/Thesis/output/Gemini')
T = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('settled_math_canonical', R / 'word_assets_20261015/exact_library_v2_20261016/prepare_math_verification.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
paths = {
    'latest_0055': R / '调整完格式的.docx',
    'D_baseline': Path('/mnt/d/Paper/Thesis/Gemini/重构论文_导言小结同步_20261018.docx'),
    'old_WSL_format_restore': R / '重构论文_格式恢复_20261016.docx',
    'three_line_donor': R / 'word_assets_20261015/md_sync_latest_20261016/math_three_layout_donor.docx',
}

def info(nodes):
    runs = [r for m in nodes for r in m.iter(h.M + 'r')]
    text = ''.join(h.text(m) for m in nodes)
    specials = []
    for r in runs:
        t = h.text(r)
        if not any(c in t for c in ('\u200b', '&', '#', '=')):
            continue
        specials.append({'text': t, 'codepoints': ['U+%04X' % ord(c) for c in t],
                         'm_nor': r.find(h.M + 'rPr/' + h.M + 'nor') is not None,
                         'm_aln': r.find(h.M + 'rPr/' + h.M + 'aln') is not None,
                         'm_sty': [v.get(h.M + 'val') for v in r.findall(h.M + 'rPr/' + h.M + 'sty')],
                         'fonts': [{E.QName(k).localname: v for k, v in n.attrib.items()} for n in r.findall(h.W + 'rPr/' + h.W + 'rFonts')],
                         'w_italic': r.find(h.W + 'rPr/' + h.W + 'i') is not None})
    return {'oMath_count': len(nodes), 'run_count': len(runs),
            'm_nor_count': sum(len(list(m.iter(h.M + 'nor'))) for m in nodes),
            'm_sty_count': sum(len(list(m.iter(h.M + 'sty'))) for m in nodes),
            'm_aln_count': sum(len(list(m.iter(h.M + 'aln'))) for m in nodes),
            'U200B_count': text.count('\u200b'), 'literal_hash_count': text.count('#'),
            'literal_ampersand_count': text.count('&'), 'ASCII_equals_count': text.count('='),
            'eqArr_lines': [h.text(e) for m in nodes for arr in m.iter(h.M + 'eqArr') for e in arr.findall(h.M + 'e')],
            'special_runs': specials}

all_rows, e46, metadata = {}, {}, {}
for name, path in paths.items():
    with ZipFile(path) as z:
        root = E.fromstring(z.read('word/document.xml'))
        settings = E.fromstring(z.read('word/settings.xml'))
    if name == 'three_line_donor':
        active = False
        nodes = []
        for block in root.find(h.W + 'body'):
            if h.text(block) == 'EXPECT_SLOT_0376':
                active = True
                continue
            if active:
                nodes.extend(block.iter(h.M + 'oMath'))
    else:
        all_rows[name] = h.word_rows(path)
        nodes = next(r['maths'] for r in all_rows[name] if r['label'] == 'E-46')
    e46[name] = nodes
    metadata[name] = {'path': str(path), **info(nodes),
                      'document_math_font': [v.get(h.M + 'val') for v in settings.iter(h.M + 'mathFont')]}

old_map = {r['label']: r for r in all_rows['D_baseline']}
same_class = []
for row in all_rows['latest_0055']:
    zero_runs = [r for m in row['maths'] for r in m.iter(h.M + 'r') if '\u200b' in h.text(r)]
    if zero_runs:
        old = old_map[row['label']]
        same_class.append({'label': row['label'], 'U200B_count': sum(h.text(r).count('\u200b') for r in zero_runs),
                           'zero_width_runs': len(zero_runs),
                           'zero_width_runs_now_normal_text': sum(r.find(h.M + 'rPr/' + h.M + 'nor') is not None for r in zero_runs),
                           'zero_width_runs_before_normal_text': sum(r.find(h.M + 'rPr/' + h.M + 'nor') is not None for m in old['maths'] for r in m.iter(h.M + 'r') if '\u200b' in h.text(r))})

source = R / '重构论文_替换库二次修订_20261016.md'
entry = next(e for e in h.md_entries(source) if e['label'] == 'E-46')
donor_content = h.content(e46['three_line_donor'], '\\begin{aligned}')
report = {
    'source_md': str(source), 'source_line': entry['source_line'], 'source_tex': entry['tex'],
    'screenshot': '/mnt/c/Users/erzhu/AppData/Local/Temp/codex-clipboard-24638776-7469-40d6-bc43-449ca44ce530.png',
    'read_only': True,
    'E46_versions': metadata,
    'canonical_math_matches_donor': {name: h.content(nodes, '\\begin{aligned}') == donor_content for name, nodes in e46.items()},
    'confirmed_findings': [
        'E-46三行数学结构、求和上下限、帽/波浪符与两个ASCII等号保持；没有字面井号或新数学符号。',
        '最新00:55版E-46的68个数学run全部设置m:nor，旧D及旧WSL版均未设置；原m:sty与显式Cambria Math的ascii/hAnsi run字体被普通文本/斜体设置取代。',
        '4个U+200B在旧版即存在：第1/3行等号前、第2行首、求和隐藏上限占位。三行donor原有3个孤立&对齐占位与1个求和隐藏上限U+200B。',
        '比较的两个已原生保存旧版与最新版本m:aln均为0；不是最新格式调整才删除m:aln。历史套式脚本把donor的&转为m:aln加U+200B，原生保存后m:aln已不在而U+200B保留。',
        '99个编号式含U+200B；它们的所有零宽run目前均设置m:nor，旧D版这些零宽run都未设置m:nor。此清单是同类结构风险，并不等于已目见99式均显示异常。'
    ],
    'root_cause': {
        'assessment': '高可信格式根因：全局普通文本化改变数学run排版处理，使遗留U+200B占位与等号/求和位置产生可见编辑占位或字形异常；等号数学字符本身正确。',
        'evidence_scope': '结构差异已核实；占位可见与m:nor之间的因果尚需原生临时副本A/B隔离确认。截图有灰色手动换行箭头，编辑标记处于可见状态，不能仅据截图断言打印亦异常。',
        'm_nor_definition_source': 'https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.math.normaltext?view=openxml-3.0.1',
        'definition_summary': '微软OMML定义：m:nor启用普通文本处理，取消数学斜体与数学间距。'
    },
    'same_class_numbered_formula_count': len(same_class),
    'same_class_numbered_formula_labels': [r['label'] for r in same_class],
    'same_class_details': same_class,
    'repair_boundary': '只处理公式排版属性及无数学含义占位。先在临时副本A/B恢复E-46数学run模式或去除其零宽占位，确认等号与求和显示；保持三行、编号、真实运算符/上下标/括号和所有中文不变。未执行修订。',
}
(T / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'E46_math_content_same': report['canonical_math_matches_donor'], 'same_class_count': len(same_class), 'output': str(T / 'diagnosis.json')}, ensure_ascii=False))
