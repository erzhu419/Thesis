"""Insert the supplied Word passages without rebuilding existing native objects."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import json
import re
import sys
from lxml import etree as E

from inventory import HERE, ROOT, SOURCE, PLAN, W, M, NS, text, read, style

sys.path.insert(0, str(ROOT / 'workpapers/章节重构_20261012'))
import restructure as previous
from integrate import body_spans

previous.SOURCE, previous.PLAN = SOURCE, PLAN
OUTPUT = ROOT / 'output/排版完成_章节补充整合_20261012.docx'
SPACE = '{http://www.w3.org/XML/1998/namespace}space'


class Supplement(previous.Manuscript):
    def __init__(self):
        super().__init__()
        self.templates['body'] = self.old[352]
        self.templates['abstractcnbody'] = self.old[65]
        self.templates['abstractenbody'] = self.old[73]
        self.eq_template = self.old[460]
        self.eq_aliases = {}
        self.cap_aliases = {}
        self.plan_insertions = []
        self.math_changes = []
        self.new_headings = []

    def equation(self, index, key):
        table = deepcopy(self.eq_template)
        for tag in ['bookmarkStart', 'bookmarkEnd']:
            for n in list(table.iter(W + tag)):
                n.getparent().remove(n)
        cells = table.find(W + 'tr').findall(W + 'tc')
        p = cells[-2].find(W + 'p')
        props = deepcopy(p.find(W + 'pPr'))
        for n in list(p):
            p.remove(n)
        if props is not None:
            p.append(props)
        mp = deepcopy(self.plan[index].find(M + 'oMathPara'))
        assert mp is not None, index
        for r in mp.iter(M + 'r'):
            rp = r.find(W + 'rPr')
            if rp is not None:
                r.remove(rp)
            rp = E.Element(W + 'rPr')
            E.SubElement(rp, W + 'rFonts', {
                W + 'ascii': 'Cambria Math', W + 'hAnsi': 'Cambria Math', W + 'eastAsia': '宋体'})
            r.insert(1 if r.find(M + 'rPr') is not None else 0, rp)
        p.append(mp)
        self.align_math_names(table, index)
        label = cells[-1].find(W + 'p')
        self.label(label, '(0-', 'Eq0', 0)
        self.bookmark(label, 'supp_eq_' + key)
        self.new_eq[id(table)] = key
        return table

    def align_math_names(self, node, index=None):
        for sub in node.iter(M + 'sSub'):
            base = sub.find(M + 'e')
            suffix = sub.find(M + 'sub')
            if base is not None and suffix is not None and text(base) == 'B' and text(suffix) == 'soft':
                runs = list(suffix.iter(M + 't'))
                runs[0].text = 'SAC'
                for r in runs[1:]:
                    r.text = ''
        if index in {141, 142}:
            for r in list(node.iter(M + 'r')):
                if text(r) != 'c':
                    continue
                sub = E.Element(M + 'sSub')
                expr = E.SubElement(sub, M + 'e')
                index_node = E.SubElement(sub, M + 'sub')
                suffix = E.SubElement(index_node, M + 'r')
                rp = E.SubElement(suffix, M + 'rPr')
                E.SubElement(rp, M + 'sty', {M + 'val': 'p'})
                E.SubElement(suffix, M + 't').text = 'obs'
                r.getparent().replace(r, sub)
                expr.append(r)
        # Keep the original distinction between raw and normalized confidence.
        if index in {325, 330}:
            bases = {'Q', 'y', 'w'} if index == 325 else {'w'}
            for r in list(node.iter(M + 'r')):
                if text(r) not in bases:
                    continue
                parent = r.getparent()
                if parent.tag == M + 'e' and parent.getparent().tag == M + 'acc':
                    mark = parent.getparent().find(M + 'accPr/' + M + 'chr')
                    mark.set(M + 'val', '\u0305')
                    continue
                acc = E.Element(M + 'acc')
                pr = E.SubElement(acc, M + 'accPr')
                E.SubElement(pr, M + 'chr', {M + 'val': '\u0305'})
                expr = E.SubElement(acc, M + 'e')
                r.getparent().replace(r, acc)
                expr.append(r)

    def table(self, original, widths):
        table = deepcopy(original)
        template = self.old[1036]
        table.find(W + 'tblPr').getparent().replace(
            table.find(W + 'tblPr'), deepcopy(template.find(W + 'tblPr')))
        total = sum(int(x.get(W + 'w')) for x in template.find(W + 'tblGrid'))
        sizes = [round(total * w / sum(widths)) for w in widths]
        sizes[-1] = total - sum(sizes[:-1])
        grid = table.find(W + 'tblGrid')
        for n in list(grid):
            grid.remove(n)
        for width in sizes:
            E.SubElement(grid, W + 'gridCol', {W + 'w': str(width)})
        rows = template.findall(W + 'tr')
        for ri, row in enumerate(table.findall(W + 'tr')):
            trp = row.find(W + 'trPr')
            if trp is None:
                trp = E.Element(W + 'trPr')
                row.insert(0, trp)
            for n in list(trp):
                trp.remove(n)
            E.SubElement(trp, W + 'cantSplit')
            if ri == 0:
                E.SubElement(trp, W + 'tblHeader')
            for ci, cell in enumerate(row.findall(W + 'tc')):
                prop = cell.find(W + 'tcPr')
                if prop is None:
                    prop = E.Element(W + 'tcPr')
                    cell.insert(0, prop)
                for n in list(prop):
                    prop.remove(n)
                E.SubElement(prop, W + 'tcW', {W + 'w': str(sizes[ci]), W + 'type': 'dxa'})
                E.SubElement(prop, W + 'vAlign', {W + 'val': 'center'})
                sample = rows[0 if ri == 0 else 1].findall(W + 'tc')[0].find(W + 'p')
                for p in cell.findall(W + 'p'):
                    pp = p.find(W + 'pPr')
                    if pp is not None:
                        p.remove(pp)
                    pp = deepcopy(sample.find(W + 'pPr'))
                    for tag in ['keepNext', 'pageBreakBefore', 'sectPr']:
                        for n in list(pp.findall(W + tag)):
                            pp.remove(n)
                    p.insert(0, pp)
                    sr = sample.find(W + 'r/' + W + 'rPr')
                    for r in p.iter(W + 'r'):
                        rp = r.find(W + 'rPr')
                        if rp is not None:
                            r.remove(rp)
                        if sr is not None:
                            r.insert(0, deepcopy(sr))
        for tag in ['bookmarkStart', 'bookmarkEnd']:
            for node in list(table.iter(W + tag)):
                node.getparent().remove(node)
        return table

    def caption(self, title, key, original=None):
        p = self.paragraph(title, 'tablecaption')
        if original is not None:
            for n in self.old[original]:
                if n.tag in {W + 'bookmarkStart', W + 'bookmarkEnd'}:
                    p.append(deepcopy(n))
            self.cap_aliases[self.cap_orig[id(self.old[original])]] = key
        self.new_cap[id(p)] = key
        self.bookmark(p, 'supp_table_' + key)
        return p

    def blocks(self, indices, prefix='p'):
        result = []
        for index in indices:
            p = self.plan[index]
            assert p.tag == W + 'p', (index, p.tag)
            assert text(p).strip(), index
            assert not text(p).startswith(('插入位置', '替换范围', '编辑核对', '另外插入', '操作：'))
            if p.find(M + 'oMathPara') is not None:
                new = self.equation(index, f'{prefix}{index}')
            else:
                new = self.formatted(p, 'body')
                self.align_math_names(new, index)
            result.append(new)
            self.plan_insertions.append(index)
        return result

    def heading(self, value, level='h3'):
        p = self.paragraph(value, level)
        self.bookmark(p, 'supp_heading_' + re.match(r'[\d.]+', value)[0].replace('.', '_'))
        self.new_headings.append(p)
        return p

    def alias(self, label, key):
        self.eq_aliases[label] = key

    def add_table(self, index, title, key, widths, original=None):
        return [self.caption(title, key, original), self.table(self.plan[index], widths)]

    def local_features(self, chapter):
        rows = [
            ['序号', '局部特征', '输入尺度、范围与形成方式'],
            ['1', '线路—方向编码', '0至11的类别编码，不按编号大小解释物理距离。'],
            ['2', '线路内站点进度', '当前站点序号除以该线路最大站点序号，截断至[0,1]。'],
            ['3、4', '前向、后向间隔比例', '观测间隔除以对应相邻班次计划间隔，截断至[0,5]；邻车缺失时比例取1。'],
            ['5、6', '前向、后向计划间隔', '以秒计的计划间隔分别除以720，不与线路静态间隔混用。'],
            ['7', '本车可服务候车负荷', '可由本线路服务的滞留人数除以车辆容量，截断至[0,2]。'],
            ['8', '载客率', '车内人数除以车辆容量，截断至[0,1]。'],
            ['9', '计划间隔内的需求负荷', '可服务客流到达率乘前后计划间隔均值，再除以车辆容量，截断至[0,2]。'],
            ['10', '必要服务时间', '已完成上下客耗时除以130秒，截断至[0,2]。'],
            ['11、12', '前车、后随车辆存在标记', '存在取1，否则取0，与缺失时补齐的间隔数值配合使用。'],
            ['附加两项', '有效车辆标记与同批已选动作', '有效位置取1、补齐位置取0；已选动作截断至[-1,1]，未执行的后果不进入输入。'],
            ['五项全局量', '物理时刻、批次规模、决策序号、系统候车与车内人数', '依次除以24000、16、15、20000、20000；前三项截断至[0,1]，人数截断至[0,1.25]。'],
        ]
        table = E.Element(W + 'tbl')
        table.append(E.Element(W + 'tblPr'))
        table.append(E.Element(W + 'tblGrid'))
        for values in rows:
            row = E.SubElement(table, W + 'tr')
            for value in values:
                cell = E.SubElement(row, W + 'tc')
                p = E.SubElement(cell, W + 'p')
                r = E.SubElement(p, W + 'r')
                E.SubElement(r, W + 't').text = value
        table = self.table(table, [12, 28, 60])
        return [self.caption('公交批次状态的局部特征与归一化', f'features{chapter}'), table]

    def split_paragraph(self, source, marker):
        left = deepcopy(source)
        right = deepcopy(source)
        for p in [left, right]:
            for n in list(p):
                if n.tag != W + 'pPr':
                    p.remove(n)
        split = False
        for child in source:
            if child.tag == W + 'pPr':
                continue
            value = text(child)
            if not split and marker in value:
                assert child.tag == W + 'r' and len(child.findall(W + 't')) == 1
                a, b = value.split(marker, 1)
                if a:
                    part = deepcopy(child)
                    part.find(W + 't').text = a
                    left.append(part)
                part = deepcopy(child)
                part.find(W + 't').text = marker + b
                right.append(part)
                split = True
            else:
                (right if split else left).append(deepcopy(child))
        assert split
        return left, right

    def build(self):
        # Chapter two contains platform background, not a common experimental protocol.
        self.replace(350, 351, self.blocks([30]), '第二章章首引言原文替换')
        platform = [self.heading('2.6 公交仿真平台与场景资料', 'h2')]
        platform += self.blocks([34, 35, 36])
        platform += self.add_table(38, '多线路场景的基础资料及用途', 'platform_data', [22, 41, 37])
        platform += self.blocks([39, 40, 41])
        platform += [self.old[1116], self.old[1117]]
        previous.replace_text(self.old[1117], '公交H2O+场景的SUMO路网与双向线路',
                              '多线路公交场景的SUMO路网与双向线路')
        platform += self.blocks([44, 45, 46])
        platform += self.add_table(48, '快速公交仿真与SUMO的建模对应关系', 'platform_models', [22, 39, 39], 1119)
        platform += self.blocks([49, 50])
        self.append(425, platform, '恢复平台、来源资料与路网图', before=True)
        previous.replace_text(self.old[425], '2.6 本章小结', '2.7 本章小结')
        self.replace(426, 427, self.blocks([54]), '第二章小结原文替换')
        self.replace(340, 341, [self.paragraph(
            '第2章介绍传统公交控制与强化学习相关基础，说明鲁棒评价、环境适应、迁移学习和世界模型的基本原理，并介绍公交场景资料及快速仿真与SUMO平台。各研究章再根据具体问题建立相应状态、动作、奖励与策略价值模型。')], '第一章结构说明同步')

        self.replace(432, 438, self.blocks(range(62, 68)), '第三章问题描述原文替换')
        self.append(461, self.blocks(range(72, 76)), '恢复时距传播关系')
        self.replace(500, 503, self.blocks(range(80, 87)), '补齐同车区间乘客奖励与系统评价')
        self.alias('(3-16)', 'p84')
        self.append(511, self.blocks(range(91, 94)), '恢复间隔二阶矩与平均候车关系', before=True)
        self.append(522, [self.heading('3.2.5 共享策略与鲁棒价值模型')]
                    + self.blocks(range(98, 110)), '第三章策略与Q/V完整定义', before=True)
        self.append(593, self.blocks([114]), '第三章实验平台说明', before=True)
        multiline = self.blocks([116])
        multiline += [self.paragraph('本组采用15项运行观测，按以下顺序与固定尺度送入类别嵌入和连续特征网络。线路、车辆服务标识、站点位置、时段及方向用于区分服务背景，其余量用于描述当前间隔与运行条件。')]
        multiline += self.add_table(140, '多线路验证采用的15项运行观测', 'obs3', [10, 29, 61])
        multiline += [self.paragraph('各项观测依次除以下列固定尺度，不额外按归一化分母截断。第12项为前向计划间隔减前向观测间隔，第15项采用两倍路段有效速度的编码；归一化分母不表示实际车辆数或站点总数。'),
                      self.equation(142, 'multiline3scale')]
        multiline += [self.paragraph('本组时距奖励按相邻班次各自的前后计划间隔计算，区别于双向单线路试验的共同360秒目标。两侧邻车均存在时，基础反馈与大偏差项分别为')]
        multiline += self.blocks([420, 421, 422, 423], 'multiline3p')
        multiline += [self.paragraph('只有单侧邻车时，基础项采用该侧计划偏差的负值，相应另一侧大偏差项为0；不具备有效邻车信息的事件不进入该组控制采样。乘客时间奖励则按本章定义的同车决策区间累计，评价另按全系统时钟统计。')]
        self.append(735, multiline, '第三章多线路平台边界与本地完整状态清单')

        self.replace(762, 767, self.blocks(range(124, 130)), '第四章问题描述原文替换')
        running = self.blocks(range(134, 139))
        running += self.add_table(140, '本章15维公交观测的组成', 'obs4', [10, 29, 61])
        running += self.blocks(range(141, 148))
        # Explicitly record the two speed steps present in the frozen observation interface.
        running += [self.paragraph('第15项采用两倍路段有效速度的编码，再除以20进行输入缩放；第12项为该次前向计划间隔减去前向观测间隔。上述两项均按实际观测接口形成，不改变道路速度的物理含义。')]
        self.replace(769, 771, running, '第四章15维状态、动作与条件转移')
        self.alias('(4-1)', 'p146')
        self.append(781, self.blocks([149, 150, 151]), '第四章乘客训练奖励', before=True)
        # The runtime used by these result campaigns confirms the supplement's smooth reward.
        reward_intro = self.blocks([420, 421, 422, 423, 424])
        self.replace(778, 780, [self.paragraph('本章分别采用车头时距目标和乘客时间目标训练驻站策略。')]
                     + reward_intro, '按已核实训练快照恢复动态双侧时距奖励')
        self.alias('(4-3)', 'p421')
        # Remove the obsolete fixed-threshold prose only; retain the passenger paragraph.
        old = text(self.old[780])
        passenger = old[old.index('乘客时间目标则'):]
        self.replace(780, 781, [self.paragraph('时距奖励使用相邻班次各自的计划间隔，间隔均以秒计；奖励缩放另由训练设置规定。' + passenger)], '时距奖励说明同步')
        self.append(796, [self.heading('4.2.3 近期反馈与符号约束的策略价值模型')]
                    + self.blocks(range(156, 166)) + self.blocks(range(166, 186)),
                    '第四章近期反馈及完整软符号模型', before=True)
        self.replace(818, 824, self.blocks([190]), '第四章求解部分按替换稿去重')
        self.rewrite('计算方式见第4.3.1节', '模型定义见第4.2.3节，具体更新次序见第4.3.1节')
        self.rewrite('采用附录D.1式(D-4e)给出的目标', '采用第4.2.3节给出的软符号价值目标，附录D.1式(D-4e)给出相同定义')

        self.replace(990, 994, self.blocks(range(204, 211)), '第五章问题描述原文替换')
        domain = self.blocks(range(215, 219))
        domain += self.add_table(220, '跨保真驻站控制中的主要动力学差异', 'dynamics5', [19, 43, 38])
        domain += self.blocks([221, 222])
        self.append(999, domain, '第五章源目标运行模型差异')
        # Keep the existing finite-horizon setup, with its value equation defined below.
        prefix, tail = self.split_paragraph(self.old[1031], '任务运行至')
        self.replace(1031, 1032, [prefix] + self.blocks(range(227, 231))
                     + [tail], '第五章人数积分与逐人成本等价')
        self.append(1036, self.blocks([232, 233]) + self.local_features(5), '第五章批次状态完整接口')
        self.append(1053, [self.heading('5.2.4 混合训练的策略与价值模型')]
                    + self.blocks(range(239, 257)), '第五章混合策略、Q和IQL状态价值', before=True)

        setting = [self.heading('5.4.1 仿真标定与实验设置')]
        setting += self.blocks([262, 263])
        self.replace(1113, 1121, setting, '第五章平台材料迁回第二章后补实验设置标题')
        self.append(1140, self.blocks([269, 270]), '恢复54回合固定驻站试验条件', before=True)
        self.rewrite('加上不额外驻站基线，共八种策略，分别在三档需求和十个场景组中运行，两环境各形成240个完整回合。',
                     '加上不额外驻站基线，共八种策略。')
        matched = self.blocks([352])
        matched += self.add_table(354, '更新次数匹配的无世界模型训练与评价安排', 'matched5', [24, 25, 25, 26], 1265)
        matched += self.blocks([355])
        self.replace(1264, 1267, matched, '第五章无世界模型配置表按方法拆分')

        self.replace(1315, 1319, self.blocks(range(278, 285)), '第六章问题描述原文替换')
        self.append(1329, self.local_features(6) + self.blocks(range(289, 297)),
                    '第六章本地状态清单与混合动作分布')
        self.append(1335, self.blocks(range(298, 304)), '第六章系统成本与有限时域Q/V', before=True)
        model = [self.heading('6.2.3 模型修正与策略价值协同模型')]
        model += self.blocks([308])
        model += self.add_table(310, 'MC-WM公交模型中各网络的职责', 'networks6', [27, 36, 37])
        model += self.blocks(range(311, 333))
        self.append(1344, model, '第六章世界模型、Q/V与归一化置信协同', before=True)
        self.replace(1396, 1399, [self.paragraph('公交训练采用本章模型部分给出的混合后继价值。以下说明模型拟合、置信递推和预测样本进入价值更新的执行次序。')],
                     '第六章求解部分引用模型定义')
        self.alias('(6-19)', 'p313')
        self.replace(1402, 1404, [self.paragraph('公交模型包含3个成员，局部评分按第6.2.3节的综合预测误差关系计算。')],
                     '局部评分定义仅编号一次')
        self.alias('(6-20)', 'p323')
        self.replace(1415, 1416, [], '归一化置信定义仅编号一次')
        self.alias('(6-24)', 'p325')
        self.append(1423, [self.paragraph('车头时距补充试验中的直接执行采用策略原始驻站建议，既不施加驻站幅度变换，也不启用最低驻站保障；采用完成保障的配置则依次进行幅度调整和服务条件判断。两种执行方式均保持相同的驻站动作权限。')],
                    '按实际评价器明确直接执行范围')
        self.append(1518, self.blocks([343, 344]), '第六章源参数偏差的实验含义')
        schedules = self.add_table(361, '公交世界模型各类训练安排', 'model_schedules6', [22, 29, 25, 24], 1520)
        schedules += self.add_table(363, '公交模型实验的重复与评价统计对象', 'model_evaluations6', [24, 30, 46])
        schedules += self.blocks([364])
        self.replace(1520, 1523, schedules, '不同模型批次的训练与评价登记分开')
        self.rewrite('六组共完成300个SUMO运行日，其中无世界模型混合组为两章共用参照。',
                     '六组共完成300个SUMO评价回合，运行日和需求设置由相应随机场景规定；其中混合基线与训练方式比较共用。')
        self.rewrite('算法及模型系数在第5、6章说明',
                     '本章各组的算法配置、模型系数与执行规则分别在对应实验中说明，跨章复用的基线记录另行标明')
        self.append(2316, [self.paragraph('以下完整策略与符号目标对应正文第4.2.3节的模型定义，用于本附录的导数误差与外推条件分析。')],
                    '附录D与正文新增模型小节衔接')

        # Apply the supplied factual abstract corrections, preserving advisor background wording.
        old = text(self.old[64])
        start = old.index('本文以公交到站且完成必要上下客后为决策时刻')
        previous.replace_text(self.old[64], old[start:], text(self.plan[370]))
        self.plan_insertions.append(370)
        for index, target in [(375, 65), (380, 73)]:
            p = self.formatted(self.plan[index], style(self.old[target]))
            self.replace(target, target + 1, [p], '按补充稿同步摘要编码比较对象')
            self.plan_insertions.append(index)

        # The four new model sections use the draft wording; only local section numbers change.
        for p in self.old:
            if style(p) != 'h3' or not re.match(r'5\.4\.[1-5] ', text(p)):
                continue
            hit = re.match(r'5\.4\.(\d+)', text(p))
            previous.replace_text(p, hit[0], f'5.4.{int(hit[1])+1}')
        self.final = []
        for i, p in enumerate(self.old):
            self.final.extend(self.before[i])
            self.final.extend(self.replacements.get(i, [p]))
            self.final.extend(self.after[i])
        self.preserve_bookmarks()
        self.update_section_refs()
        self.update_toc()

    def preserve_bookmarks(self):
        existing = {n.get(W + 'id') for p in self.final for n in p.iter(W + 'bookmarkStart')}
        existing_ends = {n.get(W + 'id') for p in self.final for n in p.iter(W + 'bookmarkEnd')}
        start_nodes = {n.get(W + 'id'): n for n in self.root.iter(W + 'bookmarkStart')}
        end_nodes = {n.get(W + 'id'): n for n in self.root.iter(W + 'bookmarkEnd')}
        destinations = {}
        for label, key in self.eq_aliases.items():
            original = next(p for p in self.old if self.eq_orig.get(id(p)) == label)
            replacement = next(p for p in self.final if self.new_eq.get(id(p)) == key)
            target = replacement.find(W + 'tr').findall(W + 'tc')[-1].find(W + 'p')
            for mark in original.iter(W + 'bookmarkStart'):
                destinations[mark.get(W + 'id')] = target
        for i, blocks in self.replacements.items():
            if not blocks:
                continue
            target = next((p for p in blocks if p.tag == W + 'p'), None)
            if target is not None:
                for mark in self.old[i].iter(W + 'bookmarkStart'):
                    destinations.setdefault(mark.get(W + 'id'), target)
        for ident, node in start_nodes.items():
            if ident not in existing:
                target = destinations.get(ident)
                if target is None:
                    target = self.old[349]
                target.insert(1 if target.find(W + 'pPr') is not None else 0, deepcopy(node))
                if ident not in existing_ends:
                    target.append(deepcopy(end_nodes[ident]))
                    existing_ends.add(ident)
        for ident, node in end_nodes.items():
            if ident not in existing_ends:
                self.old[349].append(deepcopy(node))

    def update_section_refs(self):
        pattern = re.compile(r'5\.4\.([1-5])(?!\d)')
        for p in self.final:
            if style(p) in {'h3', 'TOC3'}:
                continue
            for node in p.iter(W + 't'):
                if node.text:
                    node.text = pattern.sub(lambda m: f'5.4.{int(m[1])+1}', node.text)
        self.relink_local_sections()

    def update_toc(self):
        for p in self.final:
            if style(p) == 'TOC2' and text(p).startswith('2.6 本章小结'):
                previous.replace_text(p, '2.6 本章小结', '2.7 本章小结')
            if style(p) == 'TOC3' and re.match(r'5\.4\.[1-5]', text(p)):
                hit = re.match(r'5\.4\.(\d+)', text(p))
                previous.replace_text(p, hit[0], f'5.4.{int(hit[1])+1}')
        headings = {re.match(r'[\d.]+', text(p))[0]: p for p in self.final
                    if style(p) in {'h2', 'h3'} and re.match(r'\d+\.', text(p))}
        for p in self.new_headings:
            value = text(p)
            anchor = p.xpath('./w:bookmarkStart/@w:name', namespaces=NS)[0]
            section = re.match(r'[\d.]+', value)[0]
            template = deepcopy(self.old[95 if style(p) == 'h2' else 104])
            nodes = list(template.iter(W + 't'))
            old_value = ''.join(n.text or '' for n in nodes[:-1])
            previous.replace_text(template, old_value, value)
            for link in template.iter(W + 'hyperlink'):
                link.set(W + 'anchor', anchor)
            for code in template.iter(W + 'instrText'):
                code.text = re.sub(r'PAGEREF\s+\S+', 'PAGEREF ' + anchor, code.text or '')
            target = None
            for entry in self.final:
                if not style(entry).startswith('TOC'):
                    continue
                hit = re.match(r'([1-7](?:\.\d+)+) ', text(entry))
                if hit and tuple(map(int, hit[1].split('.'))) > tuple(map(int, section.split('.'))):
                    target = entry
                    break
            assert target is not None, section
            self.final.insert(self.final.index(target), template)

    def renumber(self):
        self.mapping = {}
        self.anchor_labels = {}
        chapter = 0
        counters = {kind: defaultdict(int) for kind in ['eq', 'fig', 'tab']}
        for p in self.final:
            if style(p) == 'chapter':
                hit = re.match(r'第(\d+)章', text(p))
                chapter = int(hit[1]) if hit else 0
            if chapter not in range(2, 7):
                continue
            if id(p) in self.eq_orig or id(p) in self.new_eq:
                counters['eq'][chapter] += 1
                n = counters['eq'][chapter]
                key = self.eq_orig.get(id(p)) or '@' + self.new_eq[id(p)]
                value = f'({chapter}-{n})'
                self.mapping[key] = value
                label = p.find(W + 'tr').findall(W + 'tc')[-1].find(W + 'p')
                marks = list(label.findall(W + 'bookmarkStart')) + list(label.findall(W + 'bookmarkEnd'))
                self.label(label, f'({chapter}-', f'Eq{chapter}', n, marks)
                for bm in label.iter(W + 'bookmarkStart'):
                    self.anchor_labels[bm.get(W + 'name')] = value
            if style(p) in {'figurecaption', 'tablecaption'}:
                figure = style(p) == 'figurecaption'
                kind = 'fig' if figure else 'tab'
                counters[kind][chapter] += 1
                n = counters[kind][chapter]
                old = self.cap_orig.get(id(p))
                key = old or '@' + self.new_cap[id(p)]
                new = f"{'图' if figure else '表'}{chapter}-{n}"
                self.mapping[key] = new
                tail = text(p)[len(old):] if old else ' ' + text(p)
                marks = list(p.findall(W + 'bookmarkStart')) + list(p.findall(W + 'bookmarkEnd'))
                self.label(p, f"{'图' if figure else '表'}{chapter}-", f"{'Fig' if figure else 'Tab'}{chapter}", n, marks)
                r = E.SubElement(p, W + 'r')
                E.SubElement(r, W + 't').text = tail
                for bm in p.iter(W + 'bookmarkStart'):
                    self.anchor_labels[bm.get(W + 'name')] = new
        for old, key in self.eq_aliases.items():
            self.mapping[old] = self.mapping['@' + key]
        for old, key in self.cap_aliases.items():
            self.mapping[old] = self.mapping['@' + key]
        pattern = re.compile(r'\([2-6]-\d+[a-z]?\)|[图表][2-6]-\d+')
        for p in self.final:
            if p.tag != W + 'p' or style(p) in {'figurecaption', 'tablecaption'}:
                continue
            nodes = list(p.iter(W + 't'))
            value = ''.join(n.text or '' for n in nodes)
            changes = [(m.start(), m.end(), self.mapping[m[0]]) for m in pattern.finditer(value) if m[0] in self.mapping]
            for a, b, new in reversed(changes):
                offset = 0
                inserted = False
                for node in nodes:
                    s = node.text or ''
                    right = offset + len(s)
                    if right > a and offset < b:
                        node.text = s[:max(0, a - offset)] + (new if not inserted else '') + s[min(len(s), b - offset):]
                        node.set(SPACE, 'preserve')
                        inserted = True
                    offset = right
        self.counts = {k: dict(v) for k, v in counters.items()}

    def save(self):
        chunks = []
        for p in self.final:
            i = self.index.get(id(p))
            if i is not None and E.tostring(p) == self.original[id(p)]:
                a, b = self.spans[i]
                chunks.append(self.raw[a:b])
            else:
                chunks.append(E.tostring(p))
        raw = self.raw[:self.spans[0][0]] + b''.join(chunks) + self.raw[self.spans[-1][1]:]
        with ZipFile(SOURCE) as z:
            entries = [(info, raw if info.filename == 'word/document.xml' else z.read(info.filename))
                       for info in z.infolist()]
        with ZipFile(OUTPUT, 'w') as z:
            for info, data in entries:
                z.writestr(info, data)
        (HERE / '整合记录.json').write_text(json.dumps(
            {'operations': self.logs, 'labels': self.mapping, 'counts': self.counts,
             'plan_paragraphs': self.plan_insertions}, ensure_ascii=False, indent=2))
        print(json.dumps({'output': str(OUTPUT), 'counts': self.counts}, ensure_ascii=False))


if __name__ == '__main__':
    if not (HERE / '修改前.docx').exists():
        copy2(SOURCE, HERE / '修改前.docx')
    manuscript = Supplement()
    manuscript.build()
    manuscript.renumber()
    manuscript.save()
