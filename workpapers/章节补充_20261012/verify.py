"""Detect missing supplied text, native objects, results, and broken reference targets."""
from collections import Counter
from zipfile import ZipFile
import json
import re
from lxml import etree as E
from inventory import HERE, SOURCE, PLAN, W, M, NS, text, read, style
from supplement import OUTPUT

_, old, before = read(SOURCE)
_, new, after = read(OUTPUT)
_, _, plan = read(PLAN)
record = json.loads((HERE / '整合记录.json').read_text())
mapping = record['labels']

assert SOURCE.read_bytes() == (HERE / '修改前.docx').read_bytes(), 'baseline changed'
with ZipFile(SOURCE) as a, ZipFile(OUTPUT) as b:
    assert a.namelist() == b.namelist(), 'package objects removed or added'
    changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
    assert changed == ['word/document.xml'], changed
    native = [n for n in a.namelist() if n.startswith(('word/media/', 'word/embeddings/'))]
    assert all(a.read(n) == b.read(n) for n in native), 'image or OLE object changed'

def structure(node):
    return (node.tag, tuple(sorted(node.attrib.items())), node.text, tuple(structure(c) for c in node))

assert [structure(s) for s in old.iter(W + 'sectPr')] == [structure(s) for s in new.iter(W + 'sectPr')]
names = Counter(new.xpath('.//w:bookmarkStart/@w:name', namespaces=NS))
assert all(n == 1 for n in names.values()), 'duplicate bookmark'
assert Counter(new.xpath('.//w:bookmarkStart/@w:id', namespaces=NS)) == Counter(new.xpath('.//w:bookmarkEnd/@w:id', namespaces=NS)), 'unpaired bookmark'
targets = set(new.xpath('.//w:hyperlink/@w:anchor', namespaces=NS))
for code in new.xpath('.//w:instrText/text()', namespaces=NS):
    m = re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)', code)
    if m:
        targets.add(m[1])
assert not targets - set(names), ('missing target', targets - set(names))
assert [text(p) for p in before if style(p) == 'bibliography'] == [text(p) for p in after if style(p) == 'bibliography']
pattern = re.compile(r'\([2-6]-\d+[a-z]?\)|[图表][2-6]-\d+')
section = re.compile(r'5\.4\.([1-5])(?!\d)')
def normalize(value):
    value = value.replace('Bsoft', 'BSAC').replace('cobs', 'c')
    value = re.sub(r'\[(\d+(?:,\d+)+)\]',
                   lambda m: ''.join('[' + n + ']' for n in m[1].split(',')), value)
    return re.sub(r'\s+', '', value)

# Only numbering changes may differ from the supplied replacement wording.
visible = [normalize(text(p)) for p in after]
preserved = []
for i in record['plan_paragraphs']:
    p = plan[i]
    if p.find(M + 'oMathPara') is not None:
        original = normalize(text(p))
        assert any(original in value for value in visible), ('equation not retained', i)
    else:
        value = section.sub(lambda m: f'5.4.{int(m[1])+1}', text(p))
        value = pattern.sub(lambda m: mapping.get(m[0], m[0]), value)
        value = normalize(value)
        assert any(value in s for s in visible), ('replacement wording changed', i, value[:120])
    preserved.append(i)

# Result data stay intact; only the three protocol/configuration tables are replaced.
results = []
for i, p in enumerate(before):
    if style(p) != 'tablecaption' or not re.match(r'表[3-6]-', text(p)) or i in {1119, 1265, 1520}:
        continue
    table = E.fromstring(E.tostring(before[i + 1]))
    for node in table.iter(W + 't'):
        if node.text:
            node.text = section.sub(lambda m: f'5.4.{int(m[1])+1}', node.text)
    results.append(structure(table))
newtables = {structure(p) for p in after if p.tag == W + 'tbl'}
assert all(p in newtables for p in results), 'existing result table changed'

def part(blocks, start, end):
    a = next(i for i, p in enumerate(blocks) if text(p) == start)
    b = next(i for i, p in enumerate(blocks) if text(p) == end)
    return [text(p) for p in blocks[a:b]]

assert part(before, '7.3 研究展望', '参考文献') == part(after, '7.3 研究展望', '参考文献')
assert text(before[64]).split('本文以公交到站')[0] == text(after[64]).split('本文以公交到站')[0], 'advisor background changed'
titles = [text(p) for p in after if style(p) in {'chapter', 'h2', 'h3'}]
for value in ['2.6 公交仿真平台与场景资料', '2.7 本章小结',
              '3.2.5 共享策略与鲁棒价值模型', '4.2.3 近期反馈与符号约束的策略价值模型',
              '5.2.4 混合训练的策略与价值模型', '6.2.3 模型修正与策略价值协同模型',
              '5.4.1 仿真标定与实验设置', '5.4.6 结果讨论']:
    assert value in titles, value
assert len(titles) == len(set(titles)), 'duplicate section title'
checks = {'baseline_preserved': True, 'native_package_parts_preserved': len(native),
          'original_sections_preserved': len(list(old.iter(W + 'sectPr'))),
          'supplied_paragraphs_and_equations_preserved': len(preserved),
          'existing_result_tables_preserved': len(results),
          'references_unchanged': len([p for p in after if style(p) == 'bibliography']),
          'internal_reference_targets_valid': True, 'outlook_preserved': True,
          'advisor_abstract_background_preserved': True, 'counts': record['counts']}
(HERE / '内容与结构核对.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
print(json.dumps(checks, ensure_ascii=False))
