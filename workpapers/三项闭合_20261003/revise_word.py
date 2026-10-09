"""Patch the authoritative Word package while retaining native objects."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workpapers/全文深度校审_20261003'))
from extract_word import NS, content

BASE = ROOT / 'output/排版完成.docx'
BACKUP = HERE / '排版完成_修改前.docx'


def q(tag):
    prefix, local = tag.split(':')
    return '{' + NS[prefix] + '}' + local


def mr(text, upright=False):
    run = E.Element(q('m:r'))
    if upright:
        E.SubElement(E.SubElement(run, q('m:rPr')), q('m:sty')).set(q('m:val'), 'p')
    fonts = E.SubElement(E.SubElement(run, q('w:rPr')), q('w:rFonts'))
    fonts.set(q('w:ascii'), 'Cambria Math')
    fonts.set(q('w:hAnsi'), 'Cambria Math')
    E.SubElement(run, q('m:t')).text = text
    return run


def sub(base, label):
    node = E.Element(q('m:sSub'))
    E.SubElement(node, q('m:e')).append(mr(base))
    E.SubElement(node, q('m:sub')).append(mr(label, upright=len(label) > 1))
    return node


with ZipFile(BASE) as z:
    parts = {name: z.read(name) for name in z.namelist()}
with ZipFile(BACKUP) as z:
    assert parts['word/document.xml'] == z.read('word/document.xml'), 'Baseline changed; reread before applying.'
doc = E.fromstring(parts['word/document.xml'])
blocks = list(doc.find('w:body', NS))
assert len(blocks) == 3224
before = [content(b) for b in blocks]


def replace(node, old, new):
    ts = node.xpath('.//w:t', namespaces=NS)
    text = ''.join(t.text or '' for t in ts)
    start = text.find(old)
    assert start >= 0, (old, content(node))
    end, offset = start + len(old), 0
    for t in ts:
        value = t.text or ''
        lo, hi = offset, offset + len(value)
        if lo < end and hi > start:
            left, right = max(start - lo, 0), min(end - lo, len(value))
            t.text = value[:left] + (new if lo <= start < hi else '') + value[right:]
        offset = hi


def tokens(node, old, base, label, exclude_sub=False):
    count = 0
    for t in list(node.xpath('.//m:t', namespaces=NS)):
        if t.text == old:
            run = t.getparent()
            container = run.getparent()
            if exclude_sub and container.tag == q('m:sub'):
                continue
            if (container.tag == q('m:e') and container.getparent().tag == q('m:sSub')
                    and content(container.getparent().find('m:sub', NS)).startswith(label)):
                continue
            run.getparent().replace(run, sub(base, label))
            count += 1
    return count


def named_subs(node, base, label, new_label):
    count = 0
    for n in node.xpath('.//m:sSub | .//m:sSubSup', namespaces=NS):
        if content(n.find('m:e', NS)) == base and content(n.find('m:sub', NS)) == label:
            target = n.find('m:sub', NS)
            for child in list(target):
                target.remove(child)
            target.append(mr(new_label, upright=len(new_label) > 1))
            count += 1
    return count


trip_blocks = [553, 556, 557, 562, 563, 564, 565, 566, 568, 569, 572, 573, 608, 609]
for i in trip_blocks:
    assert tokens(blocks[i], 'j', 'j', 'tr') > 0, i
    for t in blocks[i].xpath('.//w:t', namespaces=NS):
        t.text = re.sub(r'(?<![A-Za-z])j(?![A-Za-z])', 'j_tr', t.text or '')

rows = blocks[414].findall('w:tr', NS)
trip_row = next(r for r in rows if '车辆、班次和线路内站点位置索引' in content(r))
assert tokens(trip_row, 'j', 'j', 'tr') == 1
time_row = next(r for r in rows if '班次计划与实际发车时刻' in content(r))
assert tokens(time_row, 'j', 'j', 'tr') == 2
critic_row = next(r for r in rows if '固定策略下第k个成员' in content(r))
assert named_subs(critic_row, 'c', 'k', 'j') == 1
for t in critic_row.xpath('.//m:t', namespaces=NS):
    if t.text == 'k':
        t.text = 'j'

assert named_subs(blocks[879], 'r', 'b,k+1', 'b,k') == 1
replace(blocks[878], '待同一车辆的后续观测和反馈出现后，再形成',
        '待同一车辆的后续观测和反馈出现后，再形成如下转移。奖励下标对应被评价的第k次动作，而非反馈到达时刻：')
assert tokens(blocks[881], 'B', 'B', 'batch') == 1
assert named_subs(blocks[882], 'B', 'b', 'batch,b') == 1
assert tokens(blocks[882], 'B', 'B', 'batch') == 1
replace(blocks[881], '期望样本数为', '期望样本数记为批内该车样本数的期望，满足')

# Keep the powers and accents of the value-error decomposition intact.
for i in [1088, 1089, 1090]:
    assert named_subs(blocks[i], 'e', 'i', 'Q,i') > 0
    for acc in list(blocks[i].xpath('.//m:acc', namespaces=NS)):
        if content(acc.find('m:e', NS)) == 'e':
            parent = acc.getparent()
            wrapper = E.Element(q('m:sSub'))
            target = E.SubElement(wrapper, q('m:e'))
            parent.replace(acc, wrapper)
            target.append(acc)
            E.SubElement(wrapper, q('m:sub')).append(mr('Q'))

assert tokens(blocks[1835], 'ψ', 'ψ', 's') == 1
assert tokens(blocks[1835], 'ζ', 'ψ', 'r') == 1
assert tokens(blocks[1841], 'ζ', 'ψ', 'r') == 1
assert tokens(blocks[1843], 'ψ', 'ψ', 's') == 1
assert named_subs(blocks[1860], 'e', 'i', 'mdl,i') == 2
for i in [1889, 1890]:
    assert tokens(blocks[i], 'e', 'e', 'mdl', exclude_sub=True) == 1

model_row = next(r for r in rows if '状态增量残差预测与奖励残差预测' in content(r))
math = E.SubElement(model_row.findall('w:tc', NS)[0].find('w:p', NS), q('m:oMath'))
math.append(mr(','))
math.append(sub('ψ', 's'))
math.append(mr(','))
math.append(sub('ψ', 'r'))
replace(model_row, '状态增量残差预测与奖励残差预测', '状态、奖励残差预测及两类神经修正参数')

error_row = next(r for r in rows if '预测误差尺度、局部置信信号' in content(r))
math = E.SubElement(error_row.findall('w:tc', NS)[0].find('w:p', NS), q('m:oMath'))
math.append(mr(','))
math.append(sub('e', 'mdl,i'))
replace(error_row, '预测误差尺度、局部置信信号、对数置信评价器及模型样本权重',
        '误差尺度、局部置信、置信评价、样本权重及状态预测误差')

replace(blocks[2482], '有界累计成本', '有界可测累计成本')
replace(blocks[2484], '非负重加权密度', '可测的非负重加权密度')
replace(blocks[2490], '取一个分位阈值',
        '累计成本的分布函数单调且右连续，在两端分别趋于0和1，因而存在一个分位阈值')
replace(blocks[2490], '该密度可行，且使上式取得等号。',
        '该密度可测且可行；密度有界、成本有界，故相应乘积可积，并使上式取得等号。')
replace(blocks[2493], '以上结论针对有限状态动作、精确期望及固定参数下的价值算子。',
        '本附录的价值算子结论针对有限状态动作、精确期望及固定参数；第A.7.3节的风险包络表示则适用于一般概率空间上的有界可测累计成本。')

ref127 = next(b for b in blocks if content(b).startswith('[127]'))
replace(ref127, 'LANIER J,', 'LANIER J B,')

changes = [{'block': i, 'old': old, 'new': content(blocks[i])}
           for i, old in enumerate(before) if old != content(blocks[i])]
assert len(changes) >= 25
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
output = HERE / '排版完成_三项校订.docx'
with ZipFile(output, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
BASE.write_bytes(output.read_bytes())
(HERE / 'word_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'changed_blocks': len(changes), 'output': str(BASE)}, ensure_ascii=False))
