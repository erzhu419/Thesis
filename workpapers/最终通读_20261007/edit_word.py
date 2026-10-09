"""Apply the final reading corrections to the current editable Word copy."""
import copy
import json
import re
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
SOURCE = WORK / '修改前.docx'
OUTPUT = ROOT / 'output/排版完成_最终通读校订_20261007.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'w': W, 'm': M, 'r': R}
q = lambda name: '{' + W + '}' + name
mq = lambda name: '{' + M + '}' + name
with ZipFile(SOURCE) as archive:
    infos = archive.infolist()
    parts = {item.filename: archive.read(item.filename) for item in infos}
doc = E.fromstring(parts['word/document.xml'])
body = doc.find('w:body', NS)
original = copy.deepcopy(body)
rels = E.fromstring(parts['word/_rels/document.xml.rels'])
changes = []
touched = set()


def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))


def signature(node):
    return (node.tag, tuple(sorted(node.attrib.items())), node.text, node.tail,
            tuple(signature(child) for child in node))


def replace(index, old, new):
    nodes = body[index].xpath('.//w:t', namespaces=NS)
    full = ''.join(n.text or '' for n in nodes)
    assert full.count(old) == 1, (index, old)
    start = full.index(old)
    stop = start + len(old)
    offset = 0
    inserted = False
    for n in nodes:
        value = n.text or ''
        end = offset + len(value)
        if end > start and offset < stop:
            a, b = max(0, start - offset), min(len(value), stop - offset)
            n.text = value[:a] + (new if not inserted else '') + value[b:]
            inserted = True
            n.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        offset = end
    touched.add(index)
    changes.append({'block': index, 'before': old, 'after': new})


def run(value):
    n = E.Element(q('r'))
    t = E.SubElement(n, q('t'))
    t.text = value
    if value != value.strip():
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return n


replace(66,
    '单一驻站动作、车头时距目标下，RE-SAC相对主流对照在所测随机环境中具有更低的计划时距偏差和更稳定的价值评价。结果表明，在高不确定性环境中，引入保守偏移以提高价值评价稳定性，可以带来更好的控制效果；但不同运营目标仍存在取舍。',
    '单一驻站动作、车头时距目标下，多线路比较中RE-SAC相对SAC的平均计划时距偏差和前后时距不均衡分别降低5.00秒和7.36秒。上述结果支持类别共享与保守动作评价在所测运行条件下的作用，但不同运营目标仍存在取舍。')
replace(76,
    'With a single holding action and a headway objective, RE-SAC achieves lower planned-headway deviation and more stable value evaluation than the tested mainstream baselines. These results show that introducing a conservative bias to stabilize value evaluation can improve control under high uncertainty, while different operating objectives retain tradeoffs.',
    'In the multi-line holding comparison, RE-SAC reduces mean planned-headway deviation and forward-backward imbalance by 5.00 and 7.36 seconds relative to SAC under the headway objective, while operating objectives retain tradeoffs.')
replace(67, '针对运行公交系统环境的随机性随时间变化、', '针对公交客流与路况的统计规律随时间变化、')
replace(67,
    '改用乘客时间目标后，服务成本差异缩小，说明时距目标收益不能直接解释为同等幅度的乘客时间收益。',
    '改用乘客时间目标后，人均广义乘客时间相对前序BAPR增加0.068%至0.268%，表明该组时距控制收益未转化为乘客时间收益。')
replace(77,
    'Under a passenger-time objective, service-cost differences narrow, so headway gains cannot be interpreted as equal reductions in passenger time.',
    'With passenger-time training, generalized passenger time increases by 0.068% to 0.268% relative to the preceding BAPR configuration, so the headway gains do not extend to passenger time.')
replace(68, '车辆不额外停站或增加驻站时长的单一驻站动作，共同构成控制器。',
    '车辆不额外停站或增加驻站时长的单一驻站动作，以及行程完成保障，共同构成控制器。')
replace(68, '在相同离线数据需求下的比较中，混合训练方案相对纯离线方案有正收益。',
    '在使用同一批目标离线数据、策略与评论家更新预算匹配的比较中，混合训练方案相对所测纯离线方案有正收益。')
replace(78, 'An update-budget-matched comparison supports hybrid training over the tested offline-only scheme.',
    'With the same target offline data and matched actor and critic update budgets, hybrid training improves over the tested offline-only scheme.')
replace(69, '源域和目标域环境不一致导致的车辆动力学失配', '源域和目标域环境不一致导致的公交运行转移规律失配')
replace(69,
    '车头时距目标下，高偏差条件下MC-WM相对H2O+也显示出改善。结果支持利用目标离线数据修正源仿真训练，并显示这一作用可服务于两种运营目标。 ',
    '采用车头时距奖励训练时，联合75%偏差下完成保障执行的人均广义乘客时间相对H2O+平均减少13.36秒，但时距奖励未改善。上述结果支持目标离线数据辅助源仿真训练的服务作用，并表明该作用受到训练目标与执行规则的共同影响。')
replace(79,
    'With a headway objective, MC-WM also improves over H2O+ under large source mismatch. These results support using target offline data to correct source-simulator training for both operating objectives.',
    'Under headway-reward training and joint 75% mismatch, MC-WM reduces generalized passenger time by 13.36 seconds relative to H2O+ with the completion reserve, but does not improve the headway reward. Benefits thus depend on both training and execution.')
replace(80,
    'The results support replacing SUMO offline data with real bus-operation records and continuing training in an interactive simulator as a pathway from simulation toward real bus holding control.',
    'These findings provide a basis for future bus-control studies combining real operation records with simulator interaction.')
replace(332, '精确零/正停站混合动作及执行保障', '可直接选择零驻站的混合动作及执行保障')
replace(1312,
    'RLPD-E10 的更新数据比4，即每个新增交互事件对应4次学习更新，属于独立敏感性设置。',
    '另将RLPD-E10的更新数据比增至4，即每个新增交互事件对应4次学习更新，作为独立敏感性设置。')

# Move the existing native symbol to its definition without flattening equations.
p = body[1052]
symbol = p.find('m:oMath', NS)
assert ''.join(symbol.xpath('.//m:t/text()', namespaces=NS)) == 'gθ'
r = symbol.getnext()
t = r.find('w:t', NS)
prefix = '驻站策略通过状态特征提取与动作输出两部分构建。'
assert t.text.startswith(prefix + '表示')
new_run = copy.deepcopy(r)
new_run.find('w:t', NS).text = prefix
t.text = t.text[len(prefix):]
p.insert(p.index(symbol), new_run)
touched.add(1052)
changes.append({'block': 1052, 'repair': '将g_theta移至状态特征网络的定义之前；保留原生公式。'})

# Bookmark just the displayed section number, rather than the whole appendix title.
bid = max(int(n.get(q('id'))) for n in doc.findall('.//w:bookmarkStart', NS)) + 1
for j, (index, number) in enumerate([(2895, 'F.1'), (2905, 'F.2'), (2914, 'F.3')]):
    p = body[index]
    r = next(n for n in p if n.tag == q('r') and text(n).startswith(number))
    assert text(r) == number + ' '
    r.find('w:t', NS).text = number
    location = p.index(r)
    p.insert(location, E.Element(q('bookmarkStart'), {q('id'): str(bid+j), q('name'): 'app'+number.replace('.', '')+'num'}))
    p.insert(location+2, E.Element(q('bookmarkEnd'), {q('id'): str(bid+j)}))
    p.insert(location+3, run(' '))
    touched.add(index)


def ref_runs(target, displayed):
    a, c, e = run(''), run(''), run('')
    for r, kind in [(a, 'begin'), (c, 'separate'), (e, 'end')]:
        r.remove(r.find('w:t', NS))
        E.SubElement(r, q('fldChar'), {q('fldCharType'): kind})
    instruction = E.Element(q('r'))
    E.SubElement(instruction, q('instrText'), {'{http://www.w3.org/XML/1998/namespace}space': 'preserve'}).text = f' REF {target} \\h \\* CHARFORMAT '
    return [a, instruction, c, run(displayed), e]


for index, number, suffix in [(1488, 'F.2', '.2'), (1657, 'F.1', '.1')]:
    p = body[index]
    instruction = next(n for n in p if n.xpath('.//w:instrText[contains(., " REF appF ")]', namespaces=NS))
    first = p.index(instruction)-1
    end = p.index(instruction)+1
    while not p[end].xpath('.//w:fldChar[@w:fldCharType="end"]', namespaces=NS):
        end += 1
    assert text(p[end+1]) == suffix
    for n in list(p)[first:end+2]:
        p.remove(n)
    for k, n in enumerate([run('附录')]+ref_runs('app'+number.replace('.', '')+'num', number)):
        p.insert(first+k, n)
    if index == 1657:
        r = next(n for n in p if text(n) == 'F.3')
        location = p.index(r)
        p.remove(r)
        for k, n in enumerate(ref_runs('appF3num', 'F.3')):
            p.insert(location+k, n)
    touched.add(index)
    changes.append({'block': index, 'repair': '正文附录引用改为对应小节的可更新REF域。'})

# Distinguish the evidence normalization scalar from the operating-mode set Z.
for index in [2485, 2487, 2490, 2696]:
    for r in list(body[index].xpath('.//m:r[m:t="Z"]', namespaces=NS)):
        parent = r.getparent()
        position = parent.index(r)
        sub = E.Element(mq('sSub'))
        e = E.SubElement(sub, mq('e'))
        parent.remove(r)
        e.append(r)
        sr = E.SubElement(E.SubElement(sub, mq('sub')), mq('r'))
        E.SubElement(E.SubElement(sr, mq('rPr')), mq('sty'), {mq('val'): 'p'})
        if r.find('w:rPr', NS) is not None:
            sr.append(copy.deepcopy(r.find('w:rPr', NS)))
        E.SubElement(sr, mq('t')).text = 'norm'
        parent.insert(position, sub)
    touched.add(index)
changes.append({'blocks': [2485, 2487, 2490, 2696], 'repair': '归一化常数统一记为Z_norm，模式集合仍记为Z。'})

# Make the soft-value exponent's argument explicit; the mathematical value is unchanged.
fraction = body[2189].find('.//m:f', NS)
parent = fraction.getparent()
position = parent.index(fraction)
delimiter = E.Element(mq('d'))
pr = E.SubElement(delimiter, mq('dPr'))
E.SubElement(pr, mq('begChr'), {mq('val'): '('})
E.SubElement(pr, mq('endChr'), {mq('val'): ')'})
e = E.SubElement(delimiter, mq('e'))
parent.remove(fraction)
e.append(fraction)
parent.insert(position, delimiter)
touched.add(2189)
changes.append({'block': 2189, 'repair': '为exp的分式自变量补足括号；不改变软价值定义。'})

replace(1970, '[C/OL]//Advances in Neural Information Processing Systems. 2022, 35. DOI:10.52202/068431-2652.',
    '[EB/OL]. arXiv, 2022[2026-10-07]. https://arxiv.org/abs/2206.13464.')
replace(1989, '[C/OL]//Proceedings of the 35th International Conference on Machine Learning. PMLR, 2018, 80: 1861-1870.',
    '[EB/OL]. arXiv, 2018[2026-10-07].')
replace(1989, 'https://proceedings.mlr.press/v80/haarnoja18b.html', 'https://arxiv.org/abs/1801.01290')
link = body[1989].find('w:hyperlink', NS)
rid = link.get('{'+R+'}id')
next(n for n in rels if n.get('Id') == rid).set('Target', 'https://arxiv.org/abs/1801.01290')
url = 'https://arxiv.org/abs/2206.13464'
r = next(n for n in body[1970].iter(q('r')) if url in text(n))
t = r.find('w:t', NS)
before, after = t.text.split(url)
t.text = before
rid = 'rId'+str(max(int(n.get('Id')[3:]) for n in rels if re.fullmatch(r'rId\d+', n.get('Id', '')))+1)
link = E.Element(q('hyperlink'), {'{'+R+'}id': rid, q('history'): '1'})
link.append(run(url))
p = r.getparent()
location = p.index(r)
p.insert(location+1, link)
p.insert(location+2, run(after))
E.SubElement(rels, '{'+REL+'}Relationship', Id=rid, Type=R+'/hyperlink', Target=url, TargetMode='External')

parts['word/media/image44.emf'] = (WORK/'图6-1标记校订.emf').read_bytes()
parts['word/embeddings/Microsoft_Visio_Drawing5.vsdx'] = (WORK/'图6-1标记校订.vsdx').read_bytes()
changes.append({'figure': '6-1', 'repair': '下分图改为(b)模型生成与置信度加权；同步可编辑Visio和Word矢量预览缓存。'})

assert len(body) == len(original)
for i, (before, after) in enumerate(zip(original, body)):
    if i not in touched:
        assert signature(before) == signature(after), i
assert signature(body[64]) == signature(original[64]), 'Author-edited opening paragraph changed.'
assert len(doc.xpath('//m:oMath', namespaces=NS)) == 2086
assert len(doc.xpath('//w:sectPr', namespaces=NS)) == 16
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
parts['word/_rels/document.xml.rels'] = E.tostring(rels, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(OUTPUT, 'w') as archive:
    for item in infos:
        archive.writestr(item, parts[item.filename])
(WORK/'修改记录.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
(WORK/'修改位置.json').write_text(json.dumps(sorted(touched), indent=2), encoding='utf-8')
print(json.dumps({'output': str(OUTPUT), 'modified_body_blocks': sorted(touched),
                  'unmodified_blocks_preserved': len(body)-len(touched), 'native_math': 2086}, ensure_ascii=False))
