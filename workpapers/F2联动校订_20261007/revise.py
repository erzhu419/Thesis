"""Carry the completed F-2 evidence into its related thesis sections."""
import copy
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / '排版完成_F2最终控制器结果更新_20261006.docx'
TARGET = ROOT / 'output/排版完成_F2联动校订_20261007.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W = '{' + NS['w'] + '}'

with ZipFile(SOURCE) as z:
    infos = z.infolist()
    parts = {info.filename: z.read(info.filename) for info in infos}
doc = E.fromstring(parts['word/document.xml'])
before = copy.deepcopy(doc)
body = doc.find('w:body', NS)
changes = []


def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))


def paragraph(prefix):
    hits = [p for p in body.findall('w:p', NS) if text(p).startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def replace(node, old, new):
    original = text(node)
    assert original.count(old) == 1, old
    start, end = original.index(old), original.index(old) + len(old)
    offset, inserted = 0, False
    for t in node.xpath('.//w:t', namespaces=NS):
        value = t.text or ''
        stop = offset + len(value)
        if offset < end and stop > start:
            left = value[:max(0, start - offset)]
            right = value[max(0, end - offset):] if stop > end else ''
            t.text = left + (new if not inserted else '') + right
            inserted = True
        offset = stop
    assert text(node) == original.replace(old, new, 1)
    changes.append({'body_index': body.index(node) if node.getparent() is body else None,
                    'before': original, 'after': text(node)})


def run(value):
    r = E.Element(W + 'r')
    t = E.SubElement(r, W + 't')
    t.text = value
    if value.startswith(' ') or value.endswith(' '):
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return r


def append(node, value):
    original = text(node)
    pieces = value.split('图F-2')
    for i, piece in enumerate(pieces):
        if i:
            field = E.SubElement(node, W + 'fldSimple', {W + 'instr': ' REF figF_2 \\h '})
            field.append(run('图F-2'))
        node.append(run(piece))
    changes.append({'body_index': body.index(node), 'before': original, 'after': text(node)})


# The final comparison supplements the existing mismatch results. Keep the
# original abstract's argument and the author's directly edited wording.
replace(paragraph('（3）针对现实目标环境难以持续通过试错'),
        '在十个新环境块、三档需求和九种控制器的1350次组合评价中，使窗口内',
        '在十个新环境块、三档需求和九种控制器的1350次组合评价中，完成保障相对相同学习策略的直接参照，使窗口内')
append(paragraph('（4）针对源域和目标域环境不一致'),
       ' 共同源采样与执行条件下的完整控制器补充比较中，MC-WM相对H2O+和纯源SAC的人均广义乘客时间分别平均减少23.02秒和40.25秒，乘客完成率分别提高0.1581和0.2626个百分点。')
append(paragraph('To address source-target transition mismatch'),
       ' A supplementary final-controller comparison with common source sampling and execution rules yields mean reductions of 23.02 and 40.25 seconds in generalized passenger time relative to H2O+ and source-only SAC, with completion increases of 0.1581 and 0.2626 percentage points, respectively.')

# Retain the protocol table's dimensions, adding the new final comparison to
# the existing full-schedule column rather than widening the page.
p = paragraph('表2-7汇总')
replace(p, '汇总第5、6章采用的四种实验安排。',
        '汇总第5、6章采用的四种基本实验安排，附录F.2另以完整训练日程补充最终控制器比较。')
table = body[627]
assert table.tag == W + 'tbl' and text(table).startswith('项目正式控制器')
cell_updates = {
    '结果位置': '第6.4.3至6.4.5节；附录F.2',
    '训练重复': '早期2—3个；主比较20个；最终比较5个',
    '离线初始化': '50,000次IQL；最终比较纯源组无',
    '策略/评论家更新': '模型续训20,000次；纯源按原日程',
    '最终评价回合': '早期180、270、400；主比较4800；最终225',
}
for row in table.findall('w:tr', NS):
    cells = row.findall('w:tc', NS)
    label = text(cells[0])
    if label in cell_updates:
        replace(cells[-1], text(cells[-1]), cell_updates[label])
append(paragraph('各批次训练、初始化复用与评价回合记录集中列于附录'),
       ' 附录F.2的三种最终控制器采用共同的源生成规则、名义25,000次源事件预算及停站执行规则，在五个新运行日和三档需求下完成225回合评价；纯源SAC没有离线预训练，各方法保留自身训练日程。')

# The historical nine-controller figure is valid for its original comparison;
# direct readers to the new paired comparison when discussing source-only SAC.
p = paragraph('完整控制器的人均广义时间为2392.73秒')
replace(p, '相对纯在线SAC的2593.85秒', '相对原执行方式下纯源SAC的2593.85秒')
append(p, ' 这里比较的是原确认批次中的完整方案。共同停站执行规则下的最终控制器比较见附录F.2及图F-2，H2O+相对纯源SAC平均减少17.23秒。')
replace(paragraph('图5-9 九种冻结控制器'), '九种冻结控制器', '原确认批次中九种冻结控制器')
append(paragraph('图5-9将完成保障控制器与其余八个控制器统一比较。'),
       ' 原表图中的纯在线SAC仅使用源仿真训练数据，即纯源SAC；该图与图F-2的测试运行日和参照执行方式不同，分别报告各自批次的结果。')
append(paragraph('完整控制器由学习策略、停站幅度调整和行程完成保障组成。'),
       ' 附录F.2进一步保留共同执行规则，在新运行日比较最终H2O+、MC-WM与纯源SAC，补充完整控制器层面的平均差异。')
p = paragraph('后一任务以广义乘客时间为目标，公交H2O+')
replace(p, '相对所列纯在线SAC的描述性时间降幅为7.75%。',
        '相对原执行方式下纯源SAC的描述性时间降幅为7.75%。')
append(paragraph('360次学习与执行交叉评价表明'),
       ' 在附录F.2的新运行日完整控制器比较中，共同执行规则下H2O+相对纯源SAC平均减少17.23秒，但五训练种子的配对区间包含零。')

# The twenty-initialization experiment retains its matched model-free
# comparator. The five-seed final-controller result answers another question.
p = paragraph('本节在固定SUMO离线数据和新增源仿真经验下评价模型辅助驻站。')
replace(p, '各组使用各自规定的训练与评价条件，早期接入和偏差探索见附录F。',
        '各组使用各自规定的训练与评价条件。附录F.1与F.3保留早期接入和偏差探索，F.2补充共同源采样与执行条件下的最终控制器比较。')
append(paragraph('“固定0.25”与“固定0.30”指评论家损失'),
       ' 附录F.2另行复用正式H2O+策略，并按其源生成规则与需求轮换计划续训MC-WM，在新测试日开展完整控制器比较。')
append(paragraph('受控任务说明模型类别选择可以影响最终策略'),
       ' 共同源采样与执行条件下，图F-2中的最终MC-WM相对正式H2O+与纯源SAC分别平均减少23.02秒和40.25秒人均广义时间，相应完成率提高0.1581和0.2626个百分点。该结果补充了完整控制器的服务比较。')
append(paragraph('不同训练目标、模型项系数以及训练和评价批次分别报告。'),
       ' 附录F.2采用另外五个训练种子及五个新测试运行日，纯源SAC不含离线预训练，各方法保留自身日程；MC-WM相对H2O+的平均时间差为−23.02秒，95%配对区间为[−69.23，23.19]秒，尚不足以确定稳定优势。')
append(paragraph('单一驻站的公交任务中，20个独立初始化'),
       ' 在共同源采样与执行条件下的最终控制器补充比较中，MC-WM相对正式H2O+平均减少23.02秒，相对纯源SAC平均减少40.25秒，见图F-2。')

# Keep the conclusion's old complete-controller result, identifying its
# original execution protocol and adding the newly matched comparison.
p = paragraph('针对难以持续在部署环境中试错的情况，公交H2O+')
replace(p, '相对所列纯在线SAC参照，时间的描述性降幅为7.75%。',
        '相对原执行方式下纯源SAC参照，时间的描述性降幅为7.75%。')
append(p, ' 附录F.2在新运行日采用共同执行规则，H2O+相对纯源SAC平均减少17.23秒（0.72%），这一平均差异小于原完整方案间的差异。')
append(paragraph('归一化置信更新对当前与后续对数置信加权求和'),
       ' 最终控制器的新运行日比较进一步得到MC-WM相对H2O+和纯源SAC分别平均减少23.02秒（0.97%）和40.25秒（1.68%），见图F-2。')
append(paragraph('跨保真学习的收益依赖数据与执行条件。'),
       ' 最终控制器比较新增五条MC-WM续训，复用对应的H2O+与纯源策略；MC-WM相对纯源的五种子配对区间低于零，相对H2O+则仍跨零。该组训练日程不同，完整方案差异尚不能单独归因于世界模型。')

# A final-controller comparison no longer fits an appendix titled wholly
# exploratory. Label the early plots explicitly without changing their data.
for old, new in (
    ('附录F 公交世界模型的探索性比较', '附录F 公交世界模型的补充实验'),
    ('F.1 初步实验', 'F.1 早期接入检验'),
    ('F.2 控制效果', 'F.2 最终控制器比较'),
    ('F.3 环境偏差', 'F.3 早期标定偏差探索'),
):
    for node in body.findall('w:p', NS):
        if text(node).startswith(old):
            replace(node, old, new)
replace(paragraph('本附录汇集公交世界模型的接入检验'),
        '第F.2节在统一训练与执行条件下，比较五个训练种子的最终控制器。',
        '第F.2节在共同源采样与执行条件下，比较五个训练种子的最终控制器，各方法保留自身完整训练日程。')
replace(paragraph('图F-1 共同离线初始化'), '共同离线初始化下不同续训方式',
        '早期共同离线初始化下不同续训方式')
replace(paragraph('图F-2 统一条件下'), '统一条件下不同控制方法',
        '共同源采样与执行条件下最终控制器')
append(paragraph('仿真标定误差涉及状态转移规律。'),
       ' 本节保留归一化置信改进前的单训练种子探索，训练预算、源采样和目标测试日均与第F.2节不同。')
replace(paragraph('图F-3 不同源转移偏差下'), '不同源转移偏差下', '早期不同源转移偏差下')
replace(paragraph('表F-2 世界模型辅助'), '世界模型辅助相对H2O+的时间与完成率变化',
        '早期世界模型辅助相对H2O+的时间与完成率变化')

# Detect unexpected edits to mathematical objects, section rules, images and
# fields. New references are added, while all existing fields remain intact.
for xpath in ('.//m:oMath', './/w:sectPr', './/w:bookmarkStart', './/w:bookmarkEnd',
              './/w:instrText', './/w:drawing', './/w:fldSimple'):
    a, b = before.xpath(xpath, namespaces=NS), doc.xpath(xpath, namespaces=NS)
    if xpath == './/w:fldSimple':
        continue
    assert [E.tostring(x) for x in a] == [E.tostring(x) for x in b], xpath
allowed = {row['body_index'] for row in changes if row['body_index'] is not None} | {627}
old_body = before.find('w:body', NS)
assert len(old_body) == len(body)
for i, (old, new) in enumerate(zip(old_body, body)):
    if i not in allowed:
        assert E.tostring(old) == E.tostring(new), i
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(TARGET, 'w') as z:
    for info in infos:
        z.writestr(info, parts[info.filename])
(WORK / '修改记录.json').write_text(json.dumps({'source': str(SOURCE), 'target': str(TARGET),
    'changes': changes, 'existing_equations_fields_bookmarks_sections_images_preserved': True},
    ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output': str(TARGET), 'changes': len(changes)}, ensure_ascii=False))
