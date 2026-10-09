"""Check preservation and record full-corpus symbols and evidence locations."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workpapers/全文深度校审_20261003'))
from extract_word import NS, content

def read(path):
    z = ZipFile(path)
    doc = E.fromstring(z.read('word/document.xml'))
    return z, doc, list(doc.find('w:body', NS))

oldz, old, before = read(HERE / '排版完成_修改前.docx')
newz, new, after = read(ROOT / 'output/排版完成.docx')
approved = {v['block']: v for v in json.loads((HERE / 'word_changes.json').read_text())}
assert len(before) == len(after) == 3224
unexpected = []
for i, (a, b) in enumerate(zip(before, after)):
    style = a.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
    if style and style[0].startswith('TOC'):
        continue
    if i in approved:
        assert content(a) == approved[i]['old'], i
        assert content(b) == approved[i]['new'], i
    elif content(a) != content(b):
        unexpected.append(i)
assert not unexpected, unexpected
media = [n for n in oldz.namelist() if n.startswith('word/media/')]
assert all(n in newz.namelist() and oldz.read(n) == newz.read(n) for n in media)
for prefix in ['word/footnotes.xml', 'word/endnotes.xml', 'word/embeddings/']:
    for name in [n for n in oldz.namelist() if n.startswith(prefix)]:
        assert oldz.read(name) == newz.read(name), name
assert oldz.read('word/styles.xml') == newz.read('word/styles.xml')
assert oldz.read('word/numbering.xml') == newz.read('word/numbering.xml')
assert len(new.xpath('.//w:sectPr', namespaces=NS)) == 16
names = new.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
assert len(names) == len(set(names))
preserved = {n for n in old.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
             if not n.startswith('_Toc')}
assert preserved.issubset(names)
for target in new.xpath('.//w:hyperlink/@w:anchor', namespaces=NS):
    assert target in names, target
counts = {kind: len(new.xpath(f'.//{kind}', namespaces=NS))
          for kind in ['m:oMath', 'w:tbl', 'w:sectPr']}

symbols = defaultdict(list)
sentences = []
heading = '前置'
for i, block in enumerate(after):
    style = block.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
    style = style[0] if style else ''
    if style in ['chapter', 'h2', 'h3', 'h4']:
        heading = content(block)
    if style in ['abstractcntitle', 'abstractentitle', 'symboltitle']:
        heading = content(block)
    if style.startswith('TOC') or style in ['bibliography', 'bibliographytitle']:
        continue
    for n in block.xpath('.//m:sSub | .//m:sSubSup | .//m:sSup | .//m:acc', namespaces=NS):
        value = content(n)
        if value not in [''] and len(value) < 120:
            symbols[value].append({'block': i, 'section': heading})
    if style != 'body' and not style.startswith(('abstractcn', 'abstracten')):
        continue
    if i < 63 or style.endswith('title'):
        continue
    text = content(block)
    citations = block.xpath('.//w:hyperlink[starts-with(@w:anchor,"ref-")]/@w:anchor', namespaces=NS)
    explicit = re.findall(r'(?:图|表)[A-F\d]+-\d+|式[（(][A-F\d]+-\d+[）)]|(?:定理|引理)[A-F]\.\d+', text)
    splitter = r'(?<=[.!?])\s+(?=[A-Z])' if style.startswith('abstracten') else r'(?<=[。！？])'
    for j, sentence in enumerate(re.split(splitter, text)):
        if not sentence.strip():
            continue
        sentences.append({'block': i, 'sentence': j + 1, 'section': heading,
                          'text': sentence, 'paragraph_citations': citations,
                          'paragraph_explicit_evidence': explicit})
(HERE / '全书符号出现位置.json').write_text(json.dumps(symbols, ensure_ascii=False, indent=2), encoding='utf-8')
(HERE / '全书逐句证据索引.json').write_text(json.dumps(sentences, ensure_ascii=False, indent=2), encoding='utf-8')

# Recalculate the current joint-action comparisons from frozen summaries.
data = json.loads((ROOT / 'workpapers/Word结果整合_20261003/integration_values.json').read_text())
def find(obj, scene, arm):
    return next(r for r in data['dual'] if (r['objective'], r['scene'], r['arm']) == (obj, scene, arm))
checks = []
for obj in ['passenger', 'headway']:
    for scene in ['calibrated', 'joint75']:
        mc, h2o = find(obj, scene, 'mcwm'), find(obj, scene, 'h2oplus')
        field = 'cost' if obj == 'passenger' else 'ridge'
        reference = data['daganzo'][field]
        reduction_h2o = 100 * (abs(h2o[field]) - abs(mc[field])) / abs(h2o[field])
        reduction_rule = 100 * (abs(reference) - abs(mc[field])) / abs(reference)
        checks.append({'objective': obj, 'scene': scene,
                       'vs_h2o_percent': round(reduction_h2o, 2),
                       'vs_rule_percent': round(reduction_rule, 2)})
expected = [(3.48, 1.94), (4.81, 4.13), (0.42, 4.87), (0.80, 3.18)]
assert [(c['vs_h2o_percent'], c['vs_rule_percent']) for c in checks] == expected
confirmation = json.loads((ROOT / 'experiments/mcwm_seed_confirmation_v13/completed_analysis.json').read_text())
assert confirmation['complete'] is True
seed_checks = []
for scenario, values in {
        'calibrated': [(-51.14, 16), (-57.67, 19)],
        'joint50': [(-69.93, 19), (-70.67, 20)],
        'joint75': [(-87.05, 19), (-76.26, 20)]}.items():
    for reference, expected in zip(['h2o_matched', 'daganzo_duet'], values):
        row = next(r for r in confirmation['comparisons']
                   if (r['scenario'], r['arm'], r['reference']) ==
                   (scenario, 'fixed0250', reference))
        actual = (round(row['cost_delta_seconds']['mean'], 2), row['better_seeds'])
        assert actual == expected, (scenario, reference, actual)
        assert row['cost_delta_seconds']['n'] == 20
        seed_checks.append({'scenario': scenario, 'reference': reference,
                            'mean_delta_seconds': actual[0], 'better_seeds': actual[1]})
result = {'changed_blocks': len(approved), 'unexpected_content_changes': unexpected,
          'media_preserved': len(media), 'styles_preserved': True, 'numbering_preserved': True,
          'existing_non_toc_bookmarks_preserved': len(preserved), 'native_objects': counts,
          'indexed_sentences': len(sentences), 'indexed_composite_symbols': len(symbols),
          'joint_action_numeric_checks': checks,
          'twenty_seed_numeric_checks': seed_checks,
          'scope': 'The exhaustive indexes locate every covered item; automatic extraction does not certify every sentence semantics.'}
(HERE / 'content_acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
