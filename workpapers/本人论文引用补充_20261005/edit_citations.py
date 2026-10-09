"""Add the author's five papers in-place, preserving native Word objects."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import json, re

HERE = Path(__file__).resolve().parent
TARGET = HERE.parents[1] / 'output/排版完成_导师意见复核_20261005.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W = lambda tag: '{' + NS['w'] + '}' + tag
with ZipFile(HERE / '修改前.docx') as z:
    package = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(package['word/document.xml'])
original = deepcopy(doc)
body = doc.find('w:body', NS)
ps = body.findall('w:p', NS)
changes = []

def text(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=NS))

def replace(p, old, new):
    nodes = p.xpath('.//w:t', namespaces=NS)
    before = ''.join(t.text or '' for t in nodes)
    assert before.count(old) == 1, (old, before[:120])
    start = before.index(old); end = start + len(old); offset = 0
    for t in nodes:
        value = t.text or ''; stop = offset + len(value)
        if offset <= start < stop:
            t.text = value[:start-offset] + new + (value[end-offset:] if end <= stop else '')
        elif start < offset < end:
            t.text = value[end-offset:] if end <= stop else ''
        offset = stop
    assert text(p) == before.replace(old, new)

def edit(i, old, new):
    replace(ps[i], old, new)
    changes.append({'paragraph_before': i, 'old': old, 'new': new})

# Names refer to the published predecessor, not the later renamed BAPR draft.
edit(65, '贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR）',
     '贝叶斯遗忘分段鲁棒SAC（Bayesian Amnesic Piecewise-Robust SAC，BAPR）')
edit(65, '因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）',
     '因果符号BAPR（Causal-Symbolic BAPR，CS-BAPR）')
edit(75, 'Bayesian Adaptive Policy Regularization (BAPR)', 'Bayesian Amnesic Piecewise-Robust SAC (BAPR)')
edit(75, 'Causal-Symbolic Bayesian Adaptive Policy Regularization (CS-BAPR)', 'Causal-Symbolic BAPR (CS-BAPR)')
edit(509, '相关单智能体公交研究将车辆、站点和服务情境输入同一决策映射',
     '作者与合作者的前期单智能体公交研究将类别特征嵌入与连续状态结合，使车辆、站点和服务情境进入同一决策映射')
edit(518, '本文提出的RE-SAC', '作者前期提出的RE-SAC')
edit(522, '本文提出BAPR，用环境识别与模式置信更新调节策略学习；进一步提出CS-BAPR',
     '作者的早期BAPR研究用环境识别与模式置信更新调节策略学习。@@bapr@@本文在此基础上进一步研究CS-BAPR')
edit(539, '本文的公交跨保真研究沿用这一数据分工', '作者与合作者的公交跨保真研究@@bus_h2o@@沿用这一数据分工')
edit(551, '本文提出的MC-WM', '作者提出的MC-WM@@mcwm@@')
edit(571, '构建处理车队异步决策事件的共享策略。', '构建处理车队异步决策事件的共享策略。@@categorical@@')
edit(571, '在SAC基础上提出RE-SAC，', '在SAC基础上提出RE-SAC@@resac@@，')
edit(573, '由BAPR的置信调节出发', '由早期BAPR@@bapr@@的置信调节出发')
edit(575, '构建公交驻站的跨保真训练方法。', '构建公交驻站的跨保真训练方法。@@bus_h2o@@')
edit(577, '提出MC-WM，', '提出MC-WM@@mcwm@@，')
edit(847, '本章首先建立类别特征与连续状态结合的事件决策模型，再提出RE-SAC，以集成价值估计和保守正则化调节动作选择。',
     '本章整合作者与合作者的类别特征嵌入共享公交控制研究@@categorical@@以及RE-SAC方法@@resac@@。首先建立类别特征与连续状态结合的事件决策模型，再以集成价值估计和保守正则化调节动作选择。')
edit(868, '本章将这一表示用于公交策略和价值函数，使站点、方向等身份信息与控制目标共同学习。',
     '作者前期将这一表示用于公交策略和价值函数，使站点、方向等身份信息与控制目标共同学习。@@categorical@@本章沿用该共享表示。')
edit(928, 'RE-SAC以SAC', 'RE-SAC@@resac@@以SAC')
edit(936, 'Embedding-SAC） 使用', 'Embedding-SAC）@@categorical@@使用')
edit(936, '后续 RE-SAC 公交实验', '后续RE-SAC研究@@resac@@的公交实验')
edit(1005, '原研究', '作者与合作者的前期研究')
edit(1163, '本章以隐含运行模式描述条件变化，由BAPR的置信调节',
     '本章以隐含运行模式描述条件变化，由作者早期BAPR研究@@bapr@@的置信调节')
edit(1374, '其次，基于H2O+', '其次，结合作者与合作者的公交跨保真研究@@bus_h2o@@，基于H2O+')
edit(1442, '这是本文RE-SAC在长沙多线路SUMO中的离线适配',
     '这是作者前期RE-SAC方法@@resac@@在长沙多线路SUMO中的离线适配')
edit(1456, '本节在第5.2节公交状态与动作模型上，',
     '作者与合作者在此基础上研究了公交驻站控制的跨保真训练与行程完成保障。@@bus_h2o@@本节在第5.2节公交状态与动作模型上，')
edit(1679, '本章提出MC-WM，', '本章以作者提出的MC-WM@@mcwm@@为基础，')
edit(1704, '算法包含四个共同环节', 'MC-WM算法@@mcwm@@包含四个共同环节')
edit(2439, '早期 BAPR 围绕分段平稳环境中的置信更新与保守价值学习展开。',
     '作者的早期BAPR研究围绕分段平稳环境中的置信更新与保守价值学习展开。@@bapr@@')

bibs = body.xpath('w:p[w:pPr/w:pStyle/@w:val="bibliography"]', namespaces=NS)
by_anchor = {}
old_number = {}
for p in bibs:
    anchor = next(s.get(W('name')) for s in p.findall('w:bookmarkStart', NS) if s.get(W('name')).startswith('ref-'))
    by_anchor[anchor] = p
    old_number[int(re.match(r'\[(\d+)\]', text(p))[1])] = anchor
assert len(bibs) == 154
keys = {'categorical': 'ref-zhang2026single', 'resac': 'ref-resac2026',
        'bapr': 'ref-zhang2026bapr', 'bus_h2o': 'ref-zhang2026busH2O', 'mcwm': 'ref-zhang2026mcwm'}

# Update the existing RE-SAC entry to the exact verified version, no duplicate.
replace(by_anchor[keys['resac']], '2026[2026-09-29]', '2026[2026-10-05]')
replace(by_anchor[keys['resac']], 'https://arxiv.org/abs/2603.18396', 'https://arxiv.org/abs/2603.18396v4')
rels = E.fromstring(package['word/_rels/document.xml.rels'])
for rel in rels:
    if rel.get('Target') == 'https://arxiv.org/abs/2603.18396':
        rel.set('Target', 'https://arxiv.org/abs/2603.18396v4')

# Keep reference bookmark ends with their own entries when reordering entries.
end_by_id = {x.get(W('id')): x for x in doc.xpath('//w:bookmarkEnd', namespaces=NS)}
for anchor, p in by_anchor.items():
    start = next(x for x in p.findall('w:bookmarkStart', NS) if x.get(W('name')) == anchor)
    end = end_by_id[start.get(W('id'))]
    end.getparent().remove(end)
    p.append(end)

new_entries = {
    'bapr': 'ZHANG Y, ZHENG L. BAPR: Bayesian amnesic piecewise-robust reinforcement learning for non-stationary continuous control[EB/OL]. arXiv, 2026[2026-10-05]. https://arxiv.org/abs/2605.16170v2.',
    'bus_h2o': 'ZHANG Y, ZHANG Q, ZHENG L. Completion-Aware Cross-Fidelity Offline-to-Online Reinforcement Learning for Multi-Line Bus Holding[EB/OL]. arXiv, 2026[2026-10-05]. https://arxiv.org/abs/2609.39868v1.',
    'mcwm': 'ZHANG Y, ZHENG L. Calibration-risk routing for controlled world-model adaptation[EB/OL]. arXiv, 2026[2026-10-05]. https://arxiv.org/abs/2610.01001v1.'}
next_id = max(int(x.get(W('id'))) for x in doc.xpath('//w:bookmarkStart', namespaces=NS)) + 1
for offset, (key, value) in enumerate(new_entries.items()):
    p = E.Element(W('p')); p.append(deepcopy(bibs[-1].find('w:pPr', NS)))
    E.SubElement(p, W('bookmarkStart'), {W('id'): str(next_id+offset), W('name'): keys[key]})
    r = E.SubElement(p, W('r')); E.SubElement(r, W('t')).text = f'[{155+offset}] '
    r = E.SubElement(p, W('r')); E.SubElement(r, W('tab')); E.SubElement(r, W('t')).text = value
    E.SubElement(p, W('bookmarkEnd'), {W('id'): str(next_id+offset)})
    by_anchor[keys[key]] = p
    old_number[155+offset] = keys[key]

def run(value, superscript=False):
    r = E.Element(W('r'))
    if superscript:
        E.SubElement(E.SubElement(r, W('rPr')), W('vertAlign'), {W('val'): 'superscript'})
    t = E.SubElement(r, W('t')); t.text = value
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return r

def cite(anchors):
    result = [run('[', True)]
    for i, anchor in enumerate(anchors):
        if i: result.append(run(',', True))
        h = E.Element(W('hyperlink'), {W('anchor'): anchor})
        h.append(run('0', True)); result.append(h)
    result.append(run(']', True))
    return result

def expand_markers(p, pattern, resolve):
    for t in list(p.xpath('.//w:t', namespaces=NS)):
        if not pattern.search(t.text or ''): continue
        r = t.getparent()
        assert r.tag == W('r') and r.getparent() == p
        assert all(c.tag in (W('rPr'), W('t'), W('lastRenderedPageBreak')) for c in r), E.tostring(r)
        output = []; last = 0
        for m in pattern.finditer(t.text):
            if m.start() > last: output.append(run(t.text[last:m.start()]))
            output.extend(cite(resolve(m)))
            last = m.end()
        if last < len(t.text): output.append(run(t.text[last:]))
        pos = p.index(r); p.remove(r)
        for n, element in enumerate(output): p.insert(pos+n, element)

for p in ps:
    expand_markers(p, re.compile(r'@@(\w+)@@'), lambda m: [keys[m[1]]])
# Restore the three existing plain-text reference groups to linked citations.
for i in [479, 480, 1005]:
    expand_markers(ps[i], re.compile(r'\[(\d+(?:,\d+)*)\]'),
                   lambda m: [old_number[int(n)] for n in m[1].split(',')])

# Assign reference numbers by first use, including paragraphs inside tables.
order = list(dict.fromkeys(doc.xpath('//w:hyperlink[starts-with(@w:anchor,"ref-")]/@w:anchor', namespaces=NS)))
assert set(order) == set(by_anchor), (set(by_anchor)-set(order), set(order)-set(by_anchor))
number = {anchor: i+1 for i, anchor in enumerate(order)}
for h in doc.xpath('//w:hyperlink[starts-with(@w:anchor,"ref-")]', namespaces=NS):
    ts = h.xpath('.//w:t', namespaces=NS); assert ts
    ts[0].text = str(number[h.get(W('anchor'))])
    for t in ts[1:]: t.text = ''
insert_at = body.index(bibs[0])
for p in bibs: body.remove(p)
for i, anchor in enumerate(order):
    p = by_anchor[anchor]
    old = re.match(r'\[\d+\]', text(p))[0]
    replace(p, old, '['+str(number[anchor])+']')
    body.insert(insert_at+i, p)

# These checks detect damaged equations, data, or citation destinations.
for query in ['//m:oMath', '//w:drawing', '//w:sectPr']:
    assert [E.tostring(x) for x in original.xpath(query, namespaces=NS)] == [E.tostring(x) for x in doc.xpath(query, namespaces=NS)], query
for old, new in zip(original.xpath('//w:tbl', namespaces=NS), doc.xpath('//w:tbl', namespaces=NS)):
    for el in (old, new):
        for h in el.xpath('.//w:hyperlink[starts-with(@w:anchor,"ref-")]', namespaces=NS):
            for t in h.xpath('.//w:t', namespaces=NS): t.text = '#CITATION#'
    assert E.tostring(old) == E.tostring(new), 'table data changed'
before_ids = set(original.xpath('//w:bookmarkStart/@w:id', namespaces=NS))
after_ids = doc.xpath('//w:bookmarkStart/@w:id', namespaces=NS)
assert before_ids <= set(after_ids) and len(after_ids) == len(set(after_ids))
assert sorted(doc.xpath('//w:bookmarkStart/@w:id', namespaces=NS)) == sorted(doc.xpath('//w:bookmarkEnd/@w:id', namespaces=NS))
assert '@@' not in text(doc) and 'Adaptive Policy Regularization' not in text(doc)
assert len(order) == 157

# Restore numeric citation text in tables after the comparison normalization.
for h in doc.xpath('//w:hyperlink[starts-with(@w:anchor,"ref-")]', namespaces=NS):
    ts = h.xpath('.//w:t', namespaces=NS); ts[0].text = str(number[h.get(W('anchor'))])
    for t in ts[1:]: t.text = ''
package['word/document.xml'] = E.tostring(doc, encoding='UTF-8', xml_declaration=True, standalone=True)
package['word/_rels/document.xml.rels'] = E.tostring(rels, encoding='UTF-8', xml_declaration=True, standalone=True)
with ZipFile(TARGET, 'w', ZIP_DEFLATED) as z:
    for name, data in package.items(): z.writestr(name, data)
(HERE/'修改记录.json').write_text(json.dumps({'edits': changes, 'new_papers': 3,
    'reference_count': 157, 'self_reference_numbers': {k: number[v] for k,v in keys.items()},
    'number_map': {n: number[a] for n,a in old_number.items()}}, ensure_ascii=False, indent=2))
print(json.dumps({'saved': str(TARGET), 'text_edits': len(changes), 'references': len(order),
    'self_reference_numbers': {k: number[v] for k,v in keys.items()}}, ensure_ascii=False))
